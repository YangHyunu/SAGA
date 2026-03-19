"""Download and cache the LOCOMO dataset."""

import json
import os
import ssl
import urllib.request

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCOMO_PATH = os.path.join(DATA_DIR, "locomo10.json")


def download_locomo(force: bool = False) -> str:
    """Download locomo10.json if not already cached. Returns file path."""
    if os.path.exists(LOCOMO_PATH) and not force:
        print(f"[download] Already cached: {LOCOMO_PATH}")
        return LOCOMO_PATH

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[download] Fetching LOCOMO dataset from GitHub...")
    # macOS Python often lacks default SSL certificates
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(LOCOMO_URL)
    with urllib.request.urlopen(req, context=ctx) as resp:
        with open(LOCOMO_PATH, "wb") as f:
            f.write(resp.read())
    print(f"[download] Saved to {LOCOMO_PATH}")
    return LOCOMO_PATH


def load_locomo(path: str | None = None) -> list[dict]:
    """Load LOCOMO JSON. Downloads if needed."""
    if path is None:
        path = download_locomo()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # locomo10.json is a list of 10 conversation objects
    if isinstance(data, dict):
        data = [data]
    print(f"[download] Loaded {len(data)} conversations")
    return data


if __name__ == "__main__":
    download_locomo()
