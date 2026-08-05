"""dreaming/storage.py — KV 문서 저장 (스펙 §8).

Storage 프로토콜은 RisuAI pluginStorage(key -> JSON blob)와 1:1 대응한다.
Phase 1 백엔드는 JSON 파일 디렉터리, Phase 2는 pluginStorage로 교체된다.
SQL·외부 DB 금지.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Protocol, Tuple

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Storage(Protocol):
    """KV 문서 저장 인터페이스. 값은 JSON 직렬화 가능한 dict."""

    def get(self, namespace: str, key: str) -> Optional[Dict]: ...

    def put(self, namespace: str, key: str, value: Dict) -> None: ...

    def delete(self, namespace: str, key: str) -> None: ...

    def scan(self, namespace: str) -> Iterator[Tuple[str, Dict]]: ...


def _check_segment(segment: str) -> str:
    if segment in (".", "..") or not _SEGMENT_RE.match(segment):
        raise ValueError(f"invalid storage path segment: {segment!r}")
    return segment


class JsonDirStorage:
    """디렉터리 기반 KV: <root>/<namespace...>/<key>.json, 원자적 쓰기."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _ns_dir(self, namespace: str) -> Path:
        parts = [_check_segment(p) for p in namespace.split("/")]
        return self.root.joinpath(*parts)

    def _path(self, namespace: str, key: str) -> Path:
        _check_segment(key)
        return self._ns_dir(namespace) / f"{key}.json"

    def get(self, namespace: str, key: str) -> Optional[Dict]:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Dict) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-temp + atomic rename: 크래시 시에도 반쪽 파일이 남지 않는다
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def delete(self, namespace: str, key: str) -> None:
        path = self._path(namespace, key)
        if path.is_file():
            path.unlink()

    def scan(self, namespace: str) -> Iterator[Tuple[str, Dict]]:
        ns_dir = self._ns_dir(namespace)
        if not ns_dir.is_dir():
            return
        for path in sorted(ns_dir.glob("*.json")):
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
