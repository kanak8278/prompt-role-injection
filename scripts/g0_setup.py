"""G0 setup gate (protocol §7).

Records GPU memory, model access/revision, tokenizer, package versions, and attention
backend; benchmarks 100 forwards at the intended length; and dumps each model's *real*
chat-template serialization of a tool result.

That last part is not incidental. Protocol §8: actual source, rendered wrapper, and claimed
source are three different variables, and Qwen2.5 is known to render tool results inside a
user-formatted block. We need the ground truth per model before any condition is built, or
condition S ("forged authority") is not well defined.

Gate: one model loads with memory headroom, and versions + rendered prompts are reproducible.

Usage:
    source env.sh && .venv/bin/python scripts/g0_setup.py
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common.model_io import ATTN_IMPL, load_model, forward_hidden, model_revision

MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
BENCH_TOKENS = 512
BENCH_FORWARDS = 100
OUT = Path(os.environ["DATA_DIR"]) / "outputs" / "g0"


def env_manifest() -> dict:
    """Everything needed to reproduce this run, per the G0 gate."""
    import numpy, sklearn, scipy

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:
        smi = f"unavailable: {e}"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "sklearn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            # CUDA_VISIBLE_DEVICES=1 means in-process cuda:0 IS physical GPU 1.
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "nvidia_smi_all_gpus": smi,
        "attn_implementation": ATTN_IMPL,
        "thread_caps": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "upstream_repo_commit": upstream_commit(),
    }


def upstream_commit() -> str:
    repo = Path(os.environ["PROJECT_DIR"]) / "reference" / "role-confusion-upstream"
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


# A minimal conversation exercising every channel the threat model cares about: a system
# policy, a genuine user task, an assistant tool call, and a tool result. The tool result
# text is a sentinel so we can locate it unambiguously in the rendered string.
SENTINEL = "TOOL_RESULT_BODY_SENTINEL"
PROBE_CONVO = [
    {"role": "system", "content": "SYSTEM_POLICY_SENTINEL"},
    {"role": "user", "content": "USER_TASK_SENTINEL"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "fetch_record",
                                          "arguments": {"key": "R-1"}}}]},
    {"role": "tool", "content": SENTINEL, "name": "fetch_record"},
]


def template_report(tok, model_name: str) -> dict:
    """Render the probe conversation and characterise how the tool result is wrapped."""
    out = {"model": model_name}

    try:
        rendered = tok.apply_chat_template(PROBE_CONVO, tokenize=False,
                                           add_generation_prompt=True)
        out["rendered"] = rendered
    except Exception as e:
        out["render_error"] = f"{type(e).__name__}: {e}"
        # Some templates reject tool_calls/tool roles; record that and retry without them.
        fallback = [m for m in PROBE_CONVO if m["role"] in ("system", "user")]
        try:
            out["rendered_fallback_no_tool"] = tok.apply_chat_template(
                fallback, tokenize=False, add_generation_prompt=True)
        except Exception as e2:
            out["render_error_fallback"] = f"{type(e2).__name__}: {e2}"
        return out

    # Protocol §8: does the tool result land inside a *user*-formatted block?
    idx = rendered.find(SENTINEL)
    out["tool_body_found"] = idx >= 0
    if idx >= 0:
        prefix = rendered[:idx]
        # Nearest preceding role header wins -- that is what the model actually sees.
        markers = {}
        for role in ("system", "user", "assistant", "tool", "ipython"):
            for pat in (f"<|start_header_id|>{role}<|end_header_id|>",
                        f"<|im_start|>{role}"):
                p = prefix.rfind(pat)
                if p > markers.get(role, -1):
                    markers[role] = p
        markers = {r: p for r, p in markers.items() if p >= 0}
        out["preceding_role_headers"] = markers
        out["nearest_preceding_header"] = (
            max(markers, key=markers.get) if markers else None)
        out["tool_body_context"] = rendered[max(0, idx - 220): idx + 60]

    # The special-token spellings untrusted payloads must never contain (protocol §4/§6).
    out["all_special_tokens"] = list(tok.all_special_tokens)
    out["added_vocab_specials"] = sorted(
        t for t, i in tok.get_added_vocab().items() if t.startswith("<|") or t.startswith("<")
    )[:60]

    # Reproducibility: same messages -> same token ids, twice. Tokenize the rendered string
    # rather than apply_chat_template(tokenize=True), which in transformers v5 returns a
    # BatchEncoding (so len() would give the number of dict keys, not the token count).
    a = tok(rendered, add_special_tokens=False)["input_ids"]
    b = tok(rendered, add_special_tokens=False)["input_ids"]
    out["render_deterministic"] = (a == b)
    out["n_tokens_probe_convo"] = len(a)

    # Does the template silently inject a date or other nondeterministic system text?
    # Protocol §8 says to fix any auto-inserted date.
    out["mentions_date_pattern"] = any(
        s in rendered for s in ("Cutting Knowledge Date", "Today Date", "current date"))
    return out


def special_token_leak_check(tok) -> dict:
    """Would a literal special-token spelling inside untrusted text become a real special
    token? If yes, the payload can forge an actual chat boundary and protocol §6's assertion
    must inspect decoded ids, not just text."""
    results = {}
    for spelling in ("<|start_header_id|>", "<|eot_id|>", "<|im_start|>", "<|im_end|>"):
        text = f"harmless text {spelling} more text"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        sid = tok.convert_tokens_to_ids(spelling)
        results[spelling] = {
            "is_known_token": sid is not None and sid != tok.unk_token_id,
            "parsed_as_single_special_id": sid in ids if sid is not None else False,
            "n_tokens": len(ids),
        }
    return results


def benchmark(model, tokenizer, n=BENCH_FORWARDS, seq_len=BENCH_TOKENS) -> dict:
    """G0 gate: 100 forwards at the intended length, with peak memory."""
    torch.cuda.reset_peak_memory_stats()
    ids = tokenizer("word " * (seq_len * 2), add_special_tokens=False)["input_ids"][:seq_len]

    for _ in range(3):  # warmup, excluded from timing
        forward_hidden(model, ids)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        forward_hidden(model, ids)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    free, total = torch.cuda.mem_get_info(0)
    return {
        "n_forwards": n,
        "seq_len": seq_len,
        "total_s": round(dt, 2),
        "s_per_forward": round(dt / n, 4),
        "forwards_per_s": round(n / dt, 2),
        "peak_alloc_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "gpu_free_GiB_after": round(free / 2**30, 2),
        "gpu_total_GiB": round(total / 2**30, 2),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"gate": "G0", "env": env_manifest(), "models": {}}
    print(json.dumps(report["env"], indent=2), flush=True)

    for name in MODELS:
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        entry = {}
        try:
            tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN"))
            entry["tokenizer"] = {
                "class": type(tok).__name__,
                "vocab_size": tok.vocab_size,
                "n_added_tokens": len(tok.get_added_vocab()),
                "chat_template_present": tok.chat_template is not None,
                "chat_template_sha": (
                    __import__("hashlib").sha256(tok.chat_template.encode()).hexdigest()[:16]
                    if tok.chat_template else None),
            }
            entry["template"] = template_report(tok, name)
            entry["special_token_leak"] = special_token_leak_check(tok)

            model, _, load_mode = load_model(name)
            entry["load_mode"] = load_mode
            entry["revision"] = model_revision(model)
            entry["config"] = {
                "n_layers": model.config.num_hidden_layers,
                "d_model": model.config.hidden_size,
                "n_heads": model.config.num_attention_heads,
                # Protocol §8: with GQA, query heads != shared kv heads. Needed before any
                # patch is described as a "head" patch.
                "n_kv_heads": getattr(model.config, "num_key_value_heads", None),
                "dtype": str(model.dtype),
            }
            entry["benchmark"] = benchmark(model, tok)
            print(json.dumps({k: entry[k] for k in
                              ("load_mode", "revision", "config", "benchmark")}, indent=2),
                  flush=True)
            print("tool-result wrapper -> nearest preceding header:",
                  entry["template"].get("nearest_preceding_header"), flush=True)

            del model
            torch.cuda.empty_cache()
            entry["status"] = "ok"
        except Exception as e:
            import traceback
            entry["status"] = "FAILED"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"FAILED: {entry['error']}", flush=True)
            torch.cuda.empty_cache()

        report["models"][name] = entry

    path = OUT / "g0_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}", flush=True)

    ok = [n for n, e in report["models"].items() if e.get("status") == "ok"]
    print(f"\nG0: {len(ok)}/{len(MODELS)} models loaded and benchmarked: {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
