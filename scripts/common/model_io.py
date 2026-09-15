"""Model loading, deterministic forward passes, and activation capture.

Protocol §2/§8: one model at a time, batch 1, `use_cache=False` on causal runs, a fixed
attention implementation that exposes the tensors we patch, eval mode, and no silent
precision changes.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fixed so the attention tensors we later patch are actually exposed. Protocol §8 requires a
# fixed backend; comparing implementations is itself a G4 check, not something to leave to
# transformers' autoselection.
ATTN_IMPL = "eager"


def load_model(model_name: str, device: str = "cuda", attn_impl: str = ATTN_IMPL):
    """bf16 first; fall back to 4-bit only if another tenant has taken the GPU.

    GPU 1 on this box has previously dropped from ~80 GB free to ~15 GB mid-session when a
    second user's job started. Returns (model, tokenizer, load_mode); load_mode goes into
    every run manifest because 4-bit activations are not bit-identical to bf16 and that
    invalidates cross-run activation comparisons.
    """
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    common = dict(token=token, device_map=device, attn_implementation=attn_impl)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, **common)
        mode = "bf16"
    except torch.cuda.OutOfMemoryError as e:
        print(f"[model_io] bf16 OOM ({e}); falling back to 4-bit nf4.", flush=True)
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            **common,
        )
        mode = "4bit_nf4"
    model.eval()
    return model, tokenizer, mode


def model_revision(model) -> str:
    return getattr(model.config, "_commit_hash", None) or "unknown"


@torch.no_grad()
def forward_hidden(model, input_ids, device="cuda", layers=None, decision_pos=None):
    """Batch-1 forward. Returns (hidden_states, logits_at_decision).

    **Semantics of `out.hidden_states` in transformers 5.14.1 -- do not guess at these.**
    The tuple has length n_layers+1, but:
        hidden_states[0]        = embedding output
        hidden_states[i]        = INPUT to block i  == output of block i-1, for 1 <= i <= n_layers-1
        hidden_states[n_layers] = final_norm(output of the last block)  <-- NOT a raw residual
    The last block's raw residual output is absent from the tuple: `tie_last_hidden_states`
    defaults to True, so the final entry is replaced by the post-RMSNorm `last_hidden_state`.
    Verified empirically against forward hooks (norm ratio ~44x on the final entry). Use
    `capture_block_residuals` when true per-block residuals are needed, e.g. for §10 patching.

    `layers` selects which entries to return; §2 says retain only needed activations. All 33
    entries at 512 tokens are ~264 MiB of CPU float32 per forward on an 8B model.

    `decision_pos` is an explicit absolute index. Negative/implicit indexing is avoided on
    purpose: `-1` means the last prompt token here but would mean the last answer token in
    `answer_logprob`, and §10 warns that a shifted command must not invalidate index-based
    comparisons.
    """
    ids = torch.as_tensor([list(input_ids)], device=device)
    out = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        output_hidden_states=True,
        use_cache=False,
    )
    pos = ids.shape[1] - 1 if decision_pos is None else decision_pos
    idxs = range(len(out.hidden_states)) if layers is None else layers
    hs = {i: out.hidden_states[i][0].float().cpu() for i in idxs}
    return hs, out.logits[0, pos].float().cpu()


@torch.no_grad()
def capture_block_residuals(model, input_ids, layers, device="cuda"):
    """True post-block residual stream for the requested decoder blocks.

    `out.hidden_states` cannot supply this for the final block (see forward_hidden), so
    residuals are taken from forward hooks on `model.model.layers[i]` instead. This is the
    tensor §10 means by "residual stream ... after a block".
    """
    store: dict[int, torch.Tensor] = {}
    handles = []

    def mk(i):
        def hook(_mod, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            store[i] = h[0].float().cpu()
        return hook

    blocks = model.model.layers
    for i in layers:
        handles.append(blocks[i].register_forward_hook(mk(i)))
    try:
        ids = torch.as_tensor([list(input_ids)], device=device)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store, out.logits[0].float().cpu()


def strict_greedy_config(tokenizer, max_new_tokens: int):
    """A GenerationConfig built from scratch, inheriting nothing from the checkpoint.

    `do_sample=False` does NOT disable the logits processors that `generation_config.json`
    asks for. Qwen2.5-7B-Instruct ships `repetition_penalty: 1.05`, so a plain
    `generate(do_sample=False)` runs penalised-greedy on Qwen and true greedy on Llama --
    a model-specific decoding difference that would silently confound every cross-model
    comparison in §8/§13, and would disagree with the log-prob read of the same distribution.
    """
    from transformers import GenerationConfig
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        length_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_new_tokens=24, device="cuda"):
    """Greedy decode from pre-rendered token ids. Protocol §9 keeps primary behavioral
    outputs as unconstrained generation, so this is the primary read, not a forced A/B choice.

    Note: the KV cache is active here. If hooks are ever applied during generation they will
    only see the single new query position, not the whole sequence -- use explicit forwards
    with `use_cache=False` for interventions (§2).
    """
    ids = torch.as_tensor([list(input_ids)], device=device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        generation_config=strict_greedy_config(tokenizer, max_new_tokens),
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def answer_logprob(model, tokenizer, prompt_ids, answer_text, prompt_text=None,
                   device="cuda"):
    """Total log P(answer_text | prompt) under teacher forcing.

    Protocol §9's margin m(x) = log P(target) - log P(authorized). Returns
    (total_logprob, ans_ids) so the caller can assert equal token counts between the two
    labels -- §9 wants balanced one-token labels where possible and a length analysis
    otherwise, and an unequal-length pair biases the summed log-prob toward the shorter label
    for reasons unrelated to authority.

    The index arithmetic (`pos = len(prompt_ids) + k - 1`) was verified against two
    independent references (incremental re-forwarding, and HF's own shifted cross-entropy);
    a +/-1 shift changes the result by ~2x, so it is not a silent failure mode.

    `prompt_text` enables §9's "verify answer tokenization in the actual response context":
    tokenizing the answer in isolation can differ from tokenizing it appended to the prompt.
    """
    ans_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if prompt_text is not None:
        joint = tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
        if joint != list(prompt_ids) + list(ans_ids):
            raise ValueError(
                "answer retokenizes differently in context; the prompt does not end on a "
                "token boundary for this answer, so the teacher-forced log-prob would not "
                "correspond to the generated continuation")
    full = list(prompt_ids) + list(ans_ids)
    ids = torch.as_tensor([full], device=device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    lo, hi = len(prompt_ids) - 1, len(full) - 1
    # Slice before .float() -- a full-sequence float32 logit copy is ~500 MiB at 1024 tokens.
    lp = torch.log_softmax(logits[0, lo:hi].float(), dim=-1)
    tgt = torch.as_tensor(ans_ids, device=lp.device)
    total = lp.gather(1, tgt[:, None]).sum().item()
    return total, ans_ids
