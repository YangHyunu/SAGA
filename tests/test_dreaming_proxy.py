"""프록시 end-to-end — 주입·마킹·기록·fail-open·캐치업 (스펙 §3.1~3.2, §8)."""
import json
import time

from fastapi.testclient import TestClient

from dreaming.assembly import KNOWLEDGE_SEP
from dreaming.proxy import Settings, create_app
from dreaming.records import StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from saga.services.pair_ledger import hash_text


def _seed_ledger(storage, session, texts):
    """(user, assistant) 텍스트 쌍을 원장에 시드 — 크래시 복구 상태는
    raw뿐 아니라 원장도 남아 있다 (같은 storage). 원장 없이 플랜만 있는
    픽스처는 베이스라인 패드가 들어간 뒤로는 비현실 상태다."""
    for t, (u, a) in enumerate(texts):
        storage.put(f"{session}/ledger", f"{t:06d}", {
            "index": t, "user_hash": hash_text(u),
            "assistant_hash": hash_text(a), "status": "confirmed",
            "turn_number": t})

_EXTRACTION = json.dumps({
    "facts": [{"claim": "포션은 50골드다", "evidence_turn": 0,
               "numbers": [{"name": "가격", "value": 50}]}],
}, ensure_ascii=False)


class FakeUpstream:
    def __init__(self):
        self.payloads = []

    async def complete(self, payload, auth=None):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": "50골드다."}}]}

    async def stream(self, payload, auth=None):
        self.payloads.append(payload)
        for piece in ["50골드", "다."]:
            data = json.dumps({"choices": [{"delta": {"content": piece}}]},
                              ensure_ascii=False)
            yield f"data: {data}\n\n".encode()
        yield b"data: [DONE]\n\n"


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def complete(self, system, user):
        return self.response


def _settings(tmp_path, idle=300.0):
    return Settings(data_dir=str(tmp_path), upstream_base_url="http://up",
                    upstream_api_key="k", idle_seconds=idle)


def _body(*texts, stream=False):
    roles = ["user", "assistant"]
    msgs = [{"role": "system", "content": "너는 상인 리사다."}]
    msgs += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
    return {"model": "anthropic/claude-sonnet-4.5", "messages": msgs,
            "stream": stream}


def test_non_stream_injects_marks_and_records(tmp_path):
    up = FakeUpstream()
    storage = JsonDirStorage(tmp_path)
    MemoryStore(storage, "sess1").append_commit(
        StateCommit(slot="소지금", op="set", value=450, turn=0))
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)

    r = client.post("/v1/chat/completions", json=_body("포션 얼마야?"),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "50골드다."

    sent = up.payloads[0]["messages"]
    sys_part = sent[0]["content"][0]                 # BP1 → content part 변환
    assert sys_part["cache_control"]["type"] == "ephemeral"
    assert KNOWLEDGE_SEP in sent[-1]["content"]
    assert "소지금: 450" in sent[-1]["content"]

    raw = storage.get("sess1/raw", "000000")         # 원본 기준으로 기록됨
    assert raw["user_text"] == "포션 얼마야?"
    assert raw["assistant_text"] == "50골드다."


def test_stream_passthrough_accumulates_and_records(tmp_path):
    up = FakeUpstream()
    storage = JsonDirStorage(tmp_path)
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)

    r = client.post("/v1/chat/completions", json=_body("포션 얼마야?", stream=True),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    assert b"data:" in r.content and b"[DONE]" in r.content
    raw = storage.get("sess1/raw", "000000")
    assert raw["assistant_text"] == "50골드다."


def test_fail_open_on_sync_error(tmp_path, monkeypatch):
    import dreaming.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod.SyncPath, "process",
                        lambda self, m: (_ for _ in ()).throw(RuntimeError("boom")))
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_body("안녕"))
    assert r.status_code == 200                       # 채팅은 안 죽는다
    assert up.payloads[0]["messages"][-1]["content"] == "안녕"   # 원본 무가공


def test_upstream_error_returns_502(tmp_path):
    class DeadUpstream:
        async def complete(self, payload, auth=None):
            raise RuntimeError("connection refused")
    app = create_app(_settings(tmp_path), upstream=DeadUpstream())
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_body("안녕"))
    assert r.status_code == 502


def test_sessions_are_isolated(tmp_path):
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    client.post("/v1/chat/completions", json=_body("안녕"),
                headers={"x-dreaming-session-id": "sess-a"})
    client.post("/v1/chat/completions", json=_body("반가워"),
                headers={"x-dreaming-session-id": "sess-b"})
    storage = JsonDirStorage(tmp_path)
    assert storage.get("sess-a/raw", "000000")["user_text"] == "안녕"
    assert storage.get("sess-b/raw", "000000")["user_text"] == "반가워"


def test_catchup_dream_runs_in_background(tmp_path):
    storage = JsonDirStorage(tmp_path)
    storage.put("sess1/raw", "000000", {          # 이전 기동에서 밀린 턴
        "turn_number": 0, "user_text": "포션 얼마야?",
        "assistant_text": "50골드다.", "user_hash": "u0", "assistant_hash": "a0"})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up,
                     dream_llm=FakeLLM(_EXTRACTION))
    with TestClient(app) as client:               # with = 루프 유지
        r = client.post("/v1/chat/completions", json=_body("다음 질문"),
                        headers={"x-dreaming-session-id": "sess1"})
        assert r.status_code == 200               # 첫 요청은 즉시 통과
        for _ in range(100):                      # 백그라운드 꿈 완료 대기
            if storage.get("sess1/dreamer", "cursor"):
                break
            time.sleep(0.02)
    assert storage.get("sess1/dreamer", "cursor") is not None
    facts = MemoryStore(storage, "sess1").list_facts()
    assert any(f.claim == "포션은 50골드다" for f in facts)


def test_stored_plan_compresses_outbound_but_records_original(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_ledger(storage, "sess1", [("질문0", "답0")])
    storage.put("sess1/compression", "plan", {
        "covers_until_turn": 1,
        "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json=_body("질문0", "답0", "질문1"),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    sent = up.payloads[0]["messages"]
    joined = json.dumps(sent, ensure_ascii=False)
    assert "[지난 이야기" in joined and "질문0" not in joined
    chunk = sent[1]                                    # system 다음 = 첫 청크
    assert chunk["content"][0]["cache_control"]["type"] == "ephemeral"  # BP2
    raw = storage.get("sess1/raw", "000001")
    assert raw["user_text"] == "질문1"                  # 기록은 원본 기준


_E2E_EXTRACTION = json.dumps({"episodes": [
    {"start_turn": 0, "end_turn": 3, "title": "포션 흥정",
     "summary": "리사와 가격을 흥정했다.", "open_threads": []}]},
    ensure_ascii=False)


def test_full_loop_dream_then_compressed_prefix(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_ledger(storage, "sess1",
                 [(f"질문{t}", f"답{t}") for t in range(10)])
    for t in range(10):
        storage.put("sess1/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"질문{t}",
            "assistant_text": f"답{t}", "user_hash": f"u{t}",
            "assistant_hash": f"a{t}"})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up,
                     dream_llm=FakeLLM(_E2E_EXTRACTION))
    history = []
    for t in range(10):
        history += [f"질문{t}", f"답{t}"]

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json=_body(*history, "새 질문"),
                        headers={"x-dreaming-session-id": "sess1"})
        assert r.status_code == 200                    # 첫 요청 즉시 통과
        first = json.dumps(up.payloads[0]["messages"], ensure_ascii=False)
        assert "질문0" in first                        # 꿈 전엔 무압축
        for _ in range(100):                           # 캐치업 꿈 대기
            if storage.get("sess1/compression", "plan"):
                break
            time.sleep(0.02)
        r2 = client.post("/v1/chat/completions",
                         json=_body(*history, "새 질문", "50골드다.", "다음 질문"),
                         headers={"x-dreaming-session-id": "sess1"})
        assert r2.status_code == 200

    plan = storage.get("sess1/compression", "plan")
    assert plan["covers_until_turn"] == 4
    sent = up.payloads[1]["messages"]
    joined = json.dumps(sent, ensure_ascii=False)
    assert "포션 흥정" in joined                       # 청크 등장
    assert "질문0" not in joined and "질문4" in joined  # 선두 치환, 꼬리 보존
    marks = sum(1 for m in sent
                if isinstance(m.get("content"), list)
                and "cache_control" in m["content"][0])
    assert marks == 3                                  # BP1 + BP2 + BP3


def test_health(tmp_path):
    app = create_app(_settings(tmp_path), upstream=FakeUpstream())
    assert TestClient(app).get("/health").json() == {"ok": True}


def test_from_env_loads_dotenv_from_root(tmp_path, monkeypatch):
    import os
    (tmp_path / ".env").write_text(
        "DREAMING_UPSTREAM_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("DREAMING_UPSTREAM_KEY", raising=False)
    try:
        s = Settings.from_env(root=tmp_path)
        assert s.upstream_api_key == "from-dotenv"
        assert s.data_dir == str(tmp_path / "dreaming_data")   # 데이터도 root 앵커
    finally:
        os.environ.pop("DREAMING_UPSTREAM_KEY", None)   # load_dotenv 잔류 제거


def test_deepseek_thinking_translation(tmp_path):
    """RisuAI의 thinking_tokens를 딥시크 공식 thinking 스위치로 번역한다."""
    up = FakeUpstream()
    st = Settings(data_dir=str(tmp_path),
                  upstream_base_url="https://api.deepseek.com",
                  upstream_api_key="k")
    client = TestClient(create_app(st, upstream=up))
    body = _body("안녕")
    body["thinking_tokens"] = 0
    client.post("/v1/chat/completions", json=body)
    sent = up.payloads[0]
    assert "thinking_tokens" not in sent
    assert sent["thinking"] == {"type": "disabled"}


def test_thinking_untouched_for_other_upstreams(tmp_path):
    up = FakeUpstream()
    st = Settings(data_dir=str(tmp_path),
                  upstream_base_url="https://openrouter.ai/api/v1",
                  upstream_api_key="k")
    client = TestClient(create_app(st, upstream=up))
    body = _body("안녕")
    body["thinking_tokens"] = 0
    client.post("/v1/chat/completions", json=body)
    sent = up.payloads[0]
    assert sent["thinking_tokens"] == 0
    assert "thinking" not in sent


def test_lore_shift_and_compression_together(tmp_path):
    # PR#5 lore_shift(1안)와 Plan 4 압축 동시 가동 — 지금까지 수동 스모크로만
    # 확인했던 교차 케이스의 정식 회귀
    storage = JsonDirStorage(tmp_path)
    _seed_ledger(storage, "sess1",
                 [("질문0", "답0"), ("질문1", "답1")])
    storage.put("sess1/compression", "plan", {
        "covers_until_turn": 1,
        "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]})
    keyed = ("성문 경비는 해가 지면 두 배로 늘어나며, 통행증 없는 외지인은 "
             "동문 초소에서 하룻밤 억류된 뒤 아침에 심문을 받는다.")
    from dreaming.sync import SyncPath
    sp = SyncPath(storage, "sess1", keyed_lore=[keyed])
    msgs = [{"role": "system", "content": f"너는 리사다.\n\n{keyed}"},
            {"role": "user", "content": "질문0"},
            {"role": "assistant", "content": "답0"},
            {"role": "user", "content": "질문1"},
            {"role": "assistant", "content": "답1"},
            {"role": "user", "content": "질문2"}]
    out, v = sp.process(msgs)
    texts = [str(m.get("content")) for m in out]
    assert keyed not in texts[0]                          # 프리픽스에서 제거
    chunk_i = next(i for i, t in enumerate(texts) if "[지난 이야기" in t)
    marks = [i for i, m in enumerate(out) if "cache_control" in m]
    last_user = max(i for i, m in enumerate(out) if m["role"] == "user")
    assert "질문0" not in "".join(texts) and "질문1" in "".join(texts)
    assert marks == [0, chunk_i, len(out) - 2]            # BP1·BP2·BP3
    assert max(marks) < last_user                         # mutable 존은 캐시 밖
    assert any("<active_lorebook>" in t for t in texts)
