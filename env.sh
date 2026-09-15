# Source before running anything: `source env.sh`
# Host-specific settings for daylnxecop003. Secrets (HF_TOKEN, ANTHROPIC_API_KEY) come
# from ~/.bashrc and are deliberately NOT duplicated here.

# GPU 1 only. GPU 0 is another user's job -- never touch it.
export CUDA_VISIBLE_DEVICES=1

# Zscaler TLS interception: Python's certifi bundle lacks the org CA, so requests/
# huggingface_hub fail SSL verification even though curl works. Point at the system bundle.
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export UV_SYSTEM_CERTS=1

# Shared caches per account storage rules.
export HF_HOME=/data/cache/huggingface
export XDG_CACHE_HOME=/rnd_ai_datasets5/projects/raka6003/cache
export UV_CACHE_DIR=/rnd_ai_datasets5/projects/raka6003/uv_cache

# The box is CPU-contended by other users; BLAS otherwise grabs every core.
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Project paths. data/ is a symlink into dataFAIR on rnd5.
export PROJECT_DIR=/rnd_ai_datasets5/projects/raka6003/prompt-injection-role-defense
export DATA_DIR=/rnd_ai_datasets5/dataFAIR/raka6003/prompt-injection-role-defense
export OUTPUT_DIR="$DATA_DIR/outputs"
export LOG_DIR="$DATA_DIR/logs"

for _v in HF_TOKEN ANTHROPIC_API_KEY; do
  if [ -z "${!_v}" ]; then echo "WARNING: $_v is unset (expected from ~/.bashrc)" >&2; fi
done
unset _v
