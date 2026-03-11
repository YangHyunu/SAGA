"""Download and cache the LongMemEval dataset."""

import json
import os
import ssl
import urllib.request

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
}


def _download_file(filename: str, force: bool = False) -> str:
    """Download a single file from HuggingFace."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path) and not force:
        print(f"[download] Already cached: {path}")
        return path

    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{HF_BASE}/{filename}"
    print(f"[download] Fetching {filename} from HuggingFace...")

    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as resp:
        with open(path, "wb") as f:
            f.write(resp.read())
    print(f"[download] Saved to {path}")
    return path


def download_longmemeval(variant: str = "s", force: bool = False) -> str:
    """Download LongMemEval dataset. variant: 's' (115K tokens) or 'oracle'."""
    filename = FILES.get(variant)
    if not filename:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(FILES.keys())}")
    return _download_file(filename, force)


def load_longmemeval(variant: str = "s", path: str | None = None) -> list[dict]:
    """Load LongMemEval JSON. Downloads if needed."""
    if path is None:
        path = download_longmemeval(variant)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[download] Loaded {len(data)} instances from {variant}")
    return data


if __name__ == "__main__":
    download_longmemeval("s")
