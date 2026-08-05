"""JsonDirStorage: KV 문서 저장 백엔드 (스펙 §8 — pluginStorage와 동일 모델)."""
import os

import pytest

from dreaming.storage import JsonDirStorage


# ------------------------------------------------------------------ #
# put / get / delete
# ------------------------------------------------------------------ #

def test_put_get_roundtrip(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("sess1/facts", "f1", {"claim": "포션 가격은 50골드", "pinned": False})
    assert s.get("sess1/facts", "f1") == {"claim": "포션 가격은 50골드", "pinned": False}


def test_get_missing_returns_none(tmp_path):
    s = JsonDirStorage(tmp_path)
    assert s.get("sess1/facts", "nope") is None


def test_put_overwrites(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    s.put("ns", "k", {"v": 2})
    assert s.get("ns", "k") == {"v": 2}


def test_delete(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    s.delete("ns", "k")
    assert s.get("ns", "k") is None
    # 없는 키 delete는 no-op (fail-open, 스펙 §2.6)
    s.delete("ns", "k")


# ------------------------------------------------------------------ #
# scan
# ------------------------------------------------------------------ #

def test_scan_yields_sorted_key_value_pairs(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "b", {"v": 2})
    s.put("ns", "a", {"v": 1})
    assert list(s.scan("ns")) == [("a", {"v": 1}), ("b", {"v": 2})]


def test_scan_missing_namespace_is_empty(tmp_path):
    s = JsonDirStorage(tmp_path)
    assert list(s.scan("ghost")) == []


def test_namespaces_are_isolated(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("sess1/facts", "k", {"v": 1})
    s.put("sess2/facts", "k", {"v": 2})
    assert s.get("sess1/facts", "k") == {"v": 1}
    assert s.get("sess2/facts", "k") == {"v": 2}


# ------------------------------------------------------------------ #
# 안전성
# ------------------------------------------------------------------ #

def test_put_leaves_no_tmp_files(tmp_path):
    # write-temp+rename crash 안전 (스펙 §8) — 성공 경로에 임시 파일 잔존 금지
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and not p.name.endswith(".json")]
    assert leftovers == []


def test_rejects_path_traversal_key(tmp_path):
    s = JsonDirStorage(tmp_path)
    with pytest.raises(ValueError):
        s.put("ns", "../evil", {"v": 1})
    with pytest.raises(ValueError):
        s.get("ns/..", "k")


def test_korean_content_survives_roundtrip(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"이름": "리사", "메모": "한/영 별칭"})
    assert s.get("ns", "k") == {"이름": "리사", "메모": "한/영 별칭"}
