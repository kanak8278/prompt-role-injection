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
def forward_hidden(model, input_ids, device="cuda"):
    """Batch-1 forward. Returns (hidden_states, logits_last).

    hidden_states is length n_layers+1 (embeddings, then each block's output), each
    [seq_len, d_model], detached to CPU float32 -- protocol §2 requires streaming detached
    caches to CPU rather than retaining everything on device.
    """
    ids = torch.as_tensor([input_ids], device=device)
    out = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        output_hidden_states=True,
        use_cache=False,
    )
    return (
        tuple(h[0].float().cpu() for h in out.hidden_states),
        out.logits[0, -1].float().cpu(),
    )


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, max_new_tokens=24, device="cuda"):
    """Greedy decode from pre-rendered token ids. Protocol §9 keeps primary behavioral
    outputs as unconstrained generation, so this is the primary read, not a forced A/B choice.
    """
    ids = torch.as_tensor([input_ids], device=device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def answer_logprob(model, tokenizer, prompt_ids, answer_text, device="cuda"):
    """Total log P(answer_text | prompt) under teacher forcing.

    Protocol §9's margin m(x) = log P(target) - log P(authorized). Returns
    (total_logprob, n_answer_tokens); the caller needs the token count because §9 requires
    balanced one-token labels where possible and length analysis otherwise.
    """
    ans_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    full = list(prompt_ids) + list(ans_ids)
    ids = torch.as_tensor([full], device=device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
    ).logits[0].float()
    total = 0.0
    for k, tok in enumerate(ans_ids):
        pos = len(prompt_ids) + k - 1  # logits at pos predict token pos+1
        total += torch.log_softmax(logits[pos], dim=-1)[tok].item()
    return total, len(ans_ids)
