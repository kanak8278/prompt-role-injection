# Running gpt-oss-20b under vLLM via Docker on this box (the fast path)

The bare-pip vLLM attempt failed: current vLLM pulls a **CUDA-13** torch stack
(`nvidia-*-cu13`, `torch 2.10+cu128`), and this host's driver (535.x) caps at **CUDA 12.2**, so
`torch.cuda.is_available()` is False for those wheels. That is the documented driver ceiling.

**Docker solves it** because the container carries its own CUDA-12.8 userspace **plus the NVIDIA
forward-compatibility libraries**, which run on an H100 (datacenter GPU) under the older 535
driver. Verified inside the container: `torch 2.10.0+cu128, cuda_available True, vllm 0.19.1`.

## Recipe (works, verified)

A local image already had the stack: **`openenv-training:latest`** (from the sibling
`openenv-environments` project — vllm==0.19.1, torch cu128). The NVIDIA container runtime is
active (`docker info` shows `nvidia runc` and CDI `nvidia.com/gpu=1`).

```bash
docker run -d --name vllm-gptoss --gpus '"device=1"' \
  -v /data/cache/huggingface:/hfcache -e HF_HOME=/hfcache -e HF_TOKEN="$HF_TOKEN" \
  -p 8001:8000 openenv-training:latest \
  vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b --port 8000 \
  --gpu-memory-utilization 0.85 --max-model-len 12288
```

- `--gpus '"device=1"'` → physical GPU 1 (GPU 0 is another user's job). Inside the container it
  is cuda:0.
- Mount the shared HF cache read/exec so it finds the already-downloaded gpt-oss weights.
- Host port **8001** → container 8000 (8000 may be used by the dabstep server).
- Startup ~2 min (weights 2 s, KV-cache profiling + CUDA-graph capture ~1 min). Poll
  `http://localhost:8001/health` (200 = ready); `GET /v1/models` lists `gpt-oss-20b`.

Server reported: **native MXFP4** (`TRITON Mxfp4 MoE backend`) — this is the paper's setting, so
it also **removes our earlier bf16-dequant deviation**. KV cache 50.4 GiB, ~107× max concurrency
at 12k context.

## Client notes

Use `/v1/completions` with the **raw Harmony prompt** — we need exact control (forged analysis
turn, prefill), which the chat endpoint hides. vLLM parses the Harmony special tokens in the raw
string (verified: a forged-analysis prefill continued correctly).

**vLLM strips the channel tokens in the returned text**, rendering them as bare words:
`analysis…assistantfinal<answer>` for a fresh assistant turn, or a leading `final<answer>` when
the prompt already opened the assistant turn (e.g. after a forged analysis). The final-channel
extractor in `repro_vllm.py` handles both — it is different from the HF-format extractor.

## Speed

HF (batch-1 `model.generate`): ~70 s/prompt for forgery gen alone.
vLLM (concurrent, 48 workers): **all 313 forgeries in ~41 s**; full gen+attack for 8 prompts in
14 s. Roughly 40×+. The Claude judge (API) is now the bottleneck, not generation.

## What this does and doesn't cover

- Great for the **generation/attack** reproduction (throughput + native MXFP4).
- Does **not** replace HF for the activation-patching / localization work (the first project):
  vLLM does not expose the residual-stream hooks those experiments need. Two engines, two jobs.
- Single GPU: the vLLM server owns GPU-1 while running; can't also run HF-hook experiments
  concurrently. `docker rm -f vllm-gptoss` frees it.

## Teardown

```bash
docker rm -f vllm-gptoss   # stops server, frees GPU-1
```
