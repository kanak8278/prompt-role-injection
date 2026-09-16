"""Download a small neutral-text sample from C4 for probe training (paper's corpus).

The role probe is trained on neutral web text wrapped in each role, so it learns the role
scaffold rather than semantics. The paper uses C4 + Dolma3 (250 seqs x 1024 tok); C4 alone is
enough for a first probe. Saved to datasets/external/c4_sample.jsonl.
"""
import json, os
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/pki/tls/certs/ca-bundle.crt")
os.environ.setdefault("SSL_CERT_FILE", "/etc/pki/tls/certs/ca-bundle.crt")
from datasets import load_dataset

OUT = Path(os.environ["DATA_DIR"]) / "datasets" / "external" / "c4_sample.jsonl"
N = int(os.environ.get("C4_N", "400"))

ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
docs, seen = [], 0
for ex in ds:
    t = ex.get("text", "").strip()
    seen += 1
    if 500 <= len(t) <= 6000:           # skip trivially short / huge docs
        docs.append({"id": f"c4-{len(docs):04d}", "text": t})
    if len(docs) >= N or seen > N * 20:
        break
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(d) for d in docs) + "\n")
print(f"wrote {len(docs)} C4 docs (scanned {seen}) -> {OUT}")
