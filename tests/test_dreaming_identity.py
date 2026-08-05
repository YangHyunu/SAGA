"""KV pair ledger: 판정 5종 (스펙 §3.1). 순수 로직은 saga에서 승계."""
from dreaming.identity import PairLedger, Verdict
from dreaming.storage import JsonDirStorage

from saga.services.pair_ledger import extract_pairs, hash_text


def _ledger(tmp_path, session="sess1"):
    return PairLedger(JsonDirStorage(tmp_path), session_id=session)


def _msgs(*texts):
    """user/assistant 교대 메시지 생성. 홀수 개면 마지막이 trailing user."""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]


def _advance(ledger, user_text, assistant_text, history):
    """한 턴 진행: 요청 분석 → 응답 기록. history는 (u, a) 텍스트 리스트."""
    flat = [t for pair in history for t in pair] + [user_text]
    pairs, last_user = extract_pairs(_msgs(*flat))
    verdict = ledger.analyze_and_apply(pairs, last_user)
    ledger.record_turn(verdict, last_user, user_text, assistant_text,
                       turn_number=verdict.position)
    return verdict


# ------------------------------------------------------------------ #
# 판정 5종
# ------------------------------------------------------------------ #

def test_first_turn_is_new_session(tmp_path):
    ledger = _ledger(tmp_path)
    pairs, last_user = extract_pairs(_msgs("안녕"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "new_session"
    assert v.position == 0


def test_second_turn_is_next_turn(tmp_path):
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "next_turn"
    assert v.position == 1


def test_resend_without_trailing_user_is_continuation(tmp_path):
    # autoContinue류: 새 user 입력 없이 히스토리만 재전송
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와."))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert last_user is None
    assert v.kind == "continuation"


def test_tail_resend_is_reroll(tmp_path):
    # 마지막 assistant pop 후 같은 user 재전송 (RisuAI 리롤 — §0.1)
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    _advance(ledger, "포션 얼마야?", "50골드다.", history=[("안녕", "어서 와.")])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "reroll"
    assert v.position == 1
    assert v.reroll_turn_number == 1


def test_mid_history_edit_is_diverged(tmp_path):
    # 중간 턴 편집: 그 지점 재전송 → 이후 턴들은 quarantine (스펙 §3.1)
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    _advance(ledger, "포션 얼마야?", "50골드다.",
             history=[("안녕", "어서 와.")])
    _advance(ledger, "3개 줘", "150골드다.",
             history=[("안녕", "어서 와."), ("포션 얼마야?", "50골드다.")])
    # 1번 턴의 user를 편집해 그 지점에서 재전송
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "diverged"
    assert v.position == 1
    # 이후 턴(2번)은 quarantined
    statuses = {row["index"]: row["status"] for row in ledger.chain(active_only=False)}
    assert statuses[2] == "quarantined"


# ------------------------------------------------------------------ #
# 원장 상태 전이 + 원문 보존
# ------------------------------------------------------------------ #

def test_recorded_turn_is_provisional_then_confirmed(tmp_path):
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    assert ledger.chain()[0]["status"] == "provisional"
    # 다음 요청에서 같은 pair가 다시 보이면 confirmed
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    ledger.analyze_and_apply(pairs, last_user)
    assert ledger.chain()[0]["status"] == "confirmed"


def test_raw_pair_stored_for_dreamer(tmp_path):
    # Dreamer(Plan 3)의 추출 입력 — 원문이 KV에 남아야 한다
    storage = JsonDirStorage(tmp_path)
    ledger = PairLedger(storage, session_id="sess1")
    _advance(ledger, "안녕", "어서 와.", history=[])
    raw = storage.get("sess1/raw", "000000")
    assert raw["user_text"] == "안녕"
    assert raw["assistant_text"] == "어서 와."
    assert raw["user_hash"] == hash_text("안녕")


def test_fail_open_on_garbage_input(tmp_path):
    # 어떤 입력에서도 예외로 채팅을 막지 않는다 (스펙 §2.6)
    ledger = _ledger(tmp_path)
    v = ledger.analyze_and_apply([], None)
    assert isinstance(v, Verdict)
    assert v.kind == "new_session"
