"""LangSmith 관측 — 선택적 활성 + fail-open (dreaming/tracing.py)."""
import threading

from dreaming import tracing


def _reset():
    tracing._client = None
    tracing._enabled = None


def test_기본은_비활성_no_op(monkeypatch):
    _reset()
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    tracing.log_run("dreaming.turn", {"q": "x"}, {"k": "y"})   # 크래시 없음
    assert tracing._enabled is False


def test_활성시_클라이언트로_전송(monkeypatch):
    _reset()
    done = threading.Event()
    calls = []

    class FakeClient:
        def create_run(self, **kw):
            calls.append(kw)
            done.set()

    tracing._enabled = True
    tracing._client = FakeClient()
    monkeypatch.setenv("DREAMING_LANGSMITH_PROJECT", "proj-x")
    tracing.log_run("dreaming.turn", {"query": "스승님"},
                    {"knowledge": "블록"}, metadata={"turn": 7})
    assert done.wait(2.0)
    kw = calls[0]
    assert kw["name"] == "dreaming.turn"
    assert kw["inputs"] == {"query": "스승님"}
    assert kw["outputs"] == {"knowledge": "블록"}
    assert kw["extra"]["metadata"] == {"turn": 7}
    assert kw["project_name"] == "proj-x"
    _reset()


def test_전송_실패는_삼킨다():
    _reset()

    class BoomClient:
        def create_run(self, **kw):
            raise RuntimeError("network down")

    tracing._enabled = True
    tracing._client = BoomClient()
    tracing.log_run("dreaming.dream", {}, {})    # 예외가 새어나오면 실패
    _reset()


def test_sync_process가_턴_트레이스를_보낸다(tmp_path, monkeypatch):
    from dreaming.storage import JsonDirStorage
    from dreaming.sync import SyncPath
    seen = []
    monkeypatch.setattr(tracing, "log_run",
                        lambda name, inputs, outputs, metadata=None:
                        seen.append((name, outputs)))
    sp = SyncPath(JsonDirStorage(tmp_path), "s-t", mark_cache_enabled=False)
    sp.process([{"role": "system", "content": "너는 리사다."},
                {"role": "user", "content": "안녕"},
                {"role": "assistant", "content": "반가워"},
                {"role": "user", "content": "잘 지냈어?"}])
    assert seen and seen[0][0] == "dreaming.turn"
    assert "knowledge" in seen[0][1]
