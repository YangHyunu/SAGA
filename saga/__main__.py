"""SAGA RP Agent Proxy — Entry point."""
import argparse
import uvicorn
import logging
import os
import shutil
import sys
from dotenv import load_dotenv

load_dotenv()

from saga.core.logging import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def _wipe_db():
    """Delete all DB/cache files for a clean start."""
    targets = [
        ("db/state.db", "file"),
        ("db/chroma", "dir"),
        ("cache/sessions", "dir"),
    ]
    for path, kind in targets:
        if not os.path.exists(path):
            continue
        if kind == "file":
            os.remove(path)
            logger.info(f"[SAGA] Deleted {path}")
        else:
            shutil.rmtree(path)
            logger.info(f"[SAGA] Deleted {path}/")


def main():
    parser = argparse.ArgumentParser(description="SAGA RP Agent Proxy")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Wipe all DB and cache files before starting",
    )
    args = parser.parse_args()

    # Check config exists
    config_path = os.environ.get("SAGA_CONFIG", "config.yaml")
    if not os.path.exists(config_path):
        example = "config.example.yaml"
        if os.path.exists(example):
            logger.error(f"[SAGA] config.yaml not found. Copy from {example}:")
            logger.error(f"  cp {example} config.yaml")
        else:
            logger.error("[SAGA] config.yaml not found.")
        sys.exit(1)

    # Wipe DB if requested (before any async DB connections)
    if args.reset_db:
        logger.info("[SAGA] --reset-db: wiping all databases and caches...")
        _wipe_db()
        logger.info("[SAGA] DB reset complete. Starting fresh.")

    from saga.config import load_config
    config = load_config(config_path)

    # Ensure directories
    os.makedirs("db", exist_ok=True)
    os.makedirs("cache/sessions", exist_ok=True)
    os.makedirs("logs/turns", exist_ok=True)

    logger.info(f"[SAGA] Starting server on {config.server.host}:{config.server.port}")
    logger.info(f"[SAGA] Narration model: {config.models.narration}")
    logger.info(f"[SAGA] Curator: {'enabled' if config.curator.enabled else 'disabled'} (interval: {config.curator.interval})")

    uvicorn.run(
        "saga.server:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
