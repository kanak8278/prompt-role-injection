# Environment — daylnxecop003

Verified 2026-09-15. Account-wide gotchas live in
`/rnd_ai_datasets1/projects/raka6003/ENVIRONMENT_NOTES.md`; this file records only what
matters for this project, plus what was checked here.

## Compute

Two H100 PCIe (80 GB each), driver 535.309.01, max supported CUDA 12.2.

- **GPU 1 is ours.** GPU 0 had 72.8/81.5 GB in use by another user's `python3` (PID 3634089)
  at 16:17 on 2026-09-15. Never touch it.
- `env.sh` pins `CUDA_VISIBLE_DEVICES=1`, so in-process `cuda:0` *is* physical GPU 1.
- GPU availability is **not stable** — a neighbour's job has previously appeared on the free
  GPU mid-session and taken it from ~80 GB free to ~15 GB. Load paths should fall back from
  bf16 to 4-bit rather than hardcoding dtype. There is a working reference implementation at
  `../llm-confidence-metacognition/scripts/common/model_io.py`.

Llama-3.1-8B in bf16 is ~16 GB of weights alone; that is not the peak figure. Protocol §2
starts at batch 1 / ~512 tokens, 1024-token ceiling, `use_cache=False` for causal runs, and
detached activation caches streamed to CPU.

## Why every line of `env.sh` exists

| Setting | Reason |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=1` | GPU 0 is another user's. |
| `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE` | Outbound HTTPS goes through a Zscaler proxy doing TLS interception. The OS trusts its CA but Python's bundled `certifi` does not, so `requests`/`huggingface_hub` fail with `CERTIFICATE_VERIFY_FAILED` while plain `curl` works. |
| `UV_SYSTEM_CERTS=1` | Same root cause; `uv` uses its own rustls store. (Was `UV_NATIVE_TLS`, now deprecated.) |
| `HF_HOME=/data/cache/huggingface` | Shared 110 GB HF cache, per account rules. Do not make a per-user one. |
| `XDG_CACHE_HOME`, `UV_CACHE_DIR` | Keep all other caches off `/home` (20 GB) and `/data/home`. |
| `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=8` | Box is CPU-contended; a single sklearn `.fit()` on wide activation matrices otherwise grabs ~40 cores. Load average has been 100+ from other users' jobs alone. |

## torch must stay on the CUDA-12 series

`pip install torch` unpinned resolves to a CUDA-13 build, which the 535 driver cannot run —
`torch.cuda.is_available()` returns False with "NVIDIA driver is too old". `torch==2.5.1`
resolves to a `+cu124` wheel from plain PyPI and works. No special index needed.

The pinned set in `pyproject.toml` is copied from `../llm-confidence-metacognition`, which is
a verified-working combination on this host — not a guessed set. Protocol §2 says not to
invent version pins before testing compatibility; reusing a known-good sibling satisfies that.

## Storage — deviates from the account default, deliberately

`~/.claude/CLAUDE.md` says datasets go to dataFAIR on rnd1 or rnd2, preferring rnd1. **Both
are now past the 90% rule** and rnd1's stated 538 G free is stale:

| Mount | Size | Used | Avail | Use% |
| --- | --- | --- | --- | --- |
| `/rnd_ai_datasets1` | 3.3T | 3.1T | 137G | 96% |
| `/rnd_ai_datasets2` | 3.0T | 3.0T | 18G | **100%** |
| `/rnd_ai_datasets5` | 4.0T | 340G | 3.7T | 9% |
| `/data` | 1.5T | 1.3T | 253G | 84% |
| `/home` | 20G | 8.6G | 12G | 43% |

So this project uses **rnd5 for both code and data**:

- code → `/rnd_ai_datasets5/projects/raka6003/prompt-injection-role-defense`
- data → `/rnd_ai_datasets5/dataFAIR/raka6003/prompt-injection-role-defense`, reached through
  the `data/` symlink, split into `datasets/ outputs/ checkpoints/ logs/ scratch/`

Re-check `df -h` before any large write; rnd2 at 100% means someone else's job is already
failing on it.

## Secrets

`HF_TOKEN` and `ANTHROPIC_API_KEY` both come from `~/.bashrc`. `env.sh` warns if either is
missing but deliberately does **not** contain them, so there is no second copy on disk to
leak. `.env` is gitignored; do not create one holding keys.
