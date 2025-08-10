# preload_pyannote.py
import os, sys
from huggingface_hub import snapshot_download

def get_hf_token():
    # 1) Env var (if Cog injects it this way)
    token = os.getenv("HF_TOKEN") or os.getenv("HF_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token.strip()
    # 2) Docker BuildKit-style secret file (common path)
    try:
        with open("/run/secrets/HF_TOKEN", "r") as f:
            return f.read().strip()
    except Exception:
        return None

token = get_hf_token()
if not token or not token.startswith("hf_"):
    sys.exit("HF token not provided at build. Run: cog build --secret id=HF_TOKEN,src=hf_token.txt")

cache_dir = os.getenv("HF_HOME") or "/root/.cache/huggingface"
repos = [
    "pyannote/speaker-diarization",
    "pyannote/segmentation",
    "pyannote/embedding",
]
for repo in repos:
    print(f"Preloading {repo} ...")
    snapshot_download(repo_id=repo, token=token, cache_dir=cache_dir)
print("✅ Pyannote repos cached to", cache_dir)
