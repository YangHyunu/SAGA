"""인스펙터 — 미니 픽스처로 LLM 0콜 e2e + 캐시 미스 차단 경로."""
import json
import types

import pytest
from benchmarks.eval import hypa
from benchmarks.fixed_haystack import inspect_hypa


def fake_embed(texts):
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _mini_hs():
    """FixedHaystack 흉내 — 필요한 속성만 (turns/run_params)."""
    turns = []
    for i in range(3):
        turns.append(types.SimpleNamespace(
            turn=i + 1, is_probe=False,
            user=f"user turn {i} " + "pad " * 40,
            assistant=f"asst turn {i} " + "pad " * 40))
    return types.SimpleNamespace(
        haystack_id="mini", turns=turns,
        run_params={"max_context": 100_000, "max_tokens": 1000})


CARD = {"greeting": "hello there"}


def test_forward_loop_zero_llm_when_cached(tmp_path):
    """캐시-온리 경로 — send 실콜 없이 완주하거나 미스를 열거한다."""
    hs = _mini_hs()
    S = hypa.HypaSettings(max_chats_per_summary=2, query_chat_count=3,
                          memory_tokens_ratio=0.2)
    result = inspect_hypa.run_inspection(
        hs, preset={}, card=CARD, impl="exp", S=S,
        max_ctx=800, embed=fake_embed, allow_llm=False,
        cache_dir=tmp_path, fixed_tokens=0,
        wire_fn=lambda kept, memory: [],       # 조립은 테스트에서 무력화
    )
    if result["misses"]:
        # 미스 경로 검증: 실콜 0으로 미스가 열거된다
        assert all("key" in m for m in result["misses"])
    assert len(result["turns"]) == 3
    assert result["meta"]["impl"] == "exp"


def test_miss_blocks_output(tmp_path):
    """미스 존재 시 write_outputs가 파일을 쓰지 않고 False."""
    res = {"meta": {"impl": "exp", "hsid": "mini", "settings": {}},
           "turns": [], "misses": [{"key": "abc", "n_msgs": 2}]}
    ok = inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert ok is False
    assert list(tmp_path.iterdir()) == []


def test_outputs_written_when_clean(tmp_path):
    res = {"meta": {"impl": "main", "hsid": "mini", "settings": {"x": 1}},
           "turns": [{"turn": 1, "memory_text": "", "kept_start_msg": 0,
                      "n_summaries": 0, "wire_tokens": 0, "events": []}],
           "misses": []}
    ok = inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert ok is True
    js = list(tmp_path.glob("hypa-inspect-*.json"))
    assert len(js) == 1
    assert json.loads(js[0].read_text())["turns"][0]["turn"] == 1


def test_render_html_selfcontained():
    res = {"meta": {"impl": "main", "hsid": "mini", "max_ctx": 45000,
                    "summary_model": "x", "settings": {"memory_tokens_ratio": 0.2},
                    "snapshot_mismatch": []},
           "turns": [
               {"turn": 1, "memory_text": "", "kept_start_msg": 0,
                "n_summaries": 0, "wire_tokens": 100, "events": [], "metrics": None,
                "error": None},
               {"turn": 2, "memory_text": "<Past Events Summary>\nA\n</Past Events Summary>",
                "kept_start_msg": 4, "n_summaries": 1, "wire_tokens": 90,
                "events": [{"ev": "batch", "idx": ["m0"], "tokens": 5}],
                "metrics": None, "error": None}],
           "misses": []}
    html = inspect_hypa.render_html(res)
    assert "const DATA" in html                 # 데이터 인라인
    assert "Past Events Summary" in html        # memory_text 포함
    assert "<script src=" not in html           # 외부 의존 0


def test_write_outputs_also_writes_html(tmp_path):
    res = {"meta": {"impl": "exp", "hsid": "mini", "max_ctx": 1, "summary_model": "x",
                    "settings": {}, "snapshot_mismatch": []},
           "turns": [], "misses": []}
    inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert list(tmp_path.glob("*.html"))
