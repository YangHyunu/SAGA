"""python3 -m dreaming — Phase 1 프록시 실행 (스펙 §8)."""
import uvicorn

from dreaming.proxy import Settings, create_app


def main() -> None:
    uvicorn.run(create_app(Settings.from_env()), host="127.0.0.1", port=8787)


if __name__ == "__main__":
    main()
