"""MENE RP Agent Proxy — Entry point."""
import uvicorn
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    # Check config exists
    config_path = os.environ.get("MENE_CONFIG", "config.yaml")
    if not os.path.exists(config_path):
        example = "config.example.yaml"
        if os.path.exists(example):
            print(f"[MENE] config.yaml not found. Copy from {example}:")
            print(f"  cp {example} config.yaml")
        else:
            print("[MENE] config.yaml not found.")
        sys.exit(1)

    from mene.config import load_config
    config = load_config(config_path)

    # Ensure directories
    os.makedirs("db", exist_ok=True)
    os.makedirs("cache/sessions", exist_ok=True)
    os.makedirs("logs/turns", exist_ok=True)

    print(f"[MENE] Starting server on {config.server.host}:{config.server.port}")
    print(f"[MENE] Narration model: {config.models.narration}")
    print(f"[MENE] Curator: {'enabled' if config.curator.enabled else 'disabled'} (interval: {config.curator.interval})")

    uvicorn.run(
        "mene.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
