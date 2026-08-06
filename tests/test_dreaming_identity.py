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


# ------------------------------------------------------------------ #
# 트림 정상상태 (corpus3·4 재생으로 실증된 붕괴의 재발 방지)
# ------------------------------------------------------------------ #

def _window(start, count, current):
    """트림된 윈도우 시뮬레이션: u{start}..u{start+count-1} pair + 현재 user."""
    pairs = [{"index": i, "user_hash": f"u{start + i}",
              "assistant_hash": f"a{start + i}"} for i in range(count)]
    return pairs, f"u{current}"


def test_trimmed_session_baseline_is_padded(tmp_path):
    # corpus3 실증: 트림된 대화 중간 합류 — 이후 윈도우가 앞으로 자라도
    # (maxContext 상향) 음수 오프셋이 안 나게 베이스라인을 띄운다
    from dreaming.identity import _BASELINE_PAD
    ledger = _ledger(tmp_path)
    pairs, cur = _window(7, 18, 25)
    v = ledger.analyze_and_apply(pairs, cur)
    assert v.position == _BASELINE_PAD + 18


def test_trim_steady_state_alignment(tmp_path):
    # corpus3 실증 붕괴: 턴 18 기록 후 다음 턴이 턴 1로 기록되던 버그
    from dreaming.identity import _BASELINE_PAD
    ledger = _ledger(tmp_path)
    pairs, cur = _window(7, 18, 25)
    v1 = ledger.analyze_and_apply(pairs, cur)
    ledger.record_turn(v1, cur, "턴25 유저", "턴25 응답",
                       turn_number=v1.position)

    pairs2 = [{"index": i, "user_hash": f"u{8 + i}",
               "assistant_hash": f"a{8 + i}"} for i in range(17)]
    pairs2.append({"index": 17, "user_hash": "u25",
                   "assistant_hash": hash_text("턴25 응답")})
    v2 = ledger.analyze_and_apply(pairs2, "u26")
    assert v2.aligned
    assert v2.position == _BASELINE_PAD + 19      # ← 버그 시절엔 1
    assert v2.offset == _BASELINE_PAD + 1         # 윈도우 첫 pair(u8)의 턴


def test_trimmed_reroll_rewinds_correct_turn(tmp_path):
    # 리롤 = 같은 윈도우 재전송 (마지막 assistant pop) — 트림 상태에서도
    # 방금 기록한 턴을 정확히 되감아야 한다 (corpus4의 "*says nothing*"
    # 3중 기록 재발 방지)
    from dreaming.identity import _BASELINE_PAD
    ledger = _ledger(tmp_path)
    pairs, cur = _window(7, 18, 25)
    v1 = ledger.analyze_and_apply(pairs, cur)
    ledger.record_turn(v1, cur, "턴25 유저", "턴25 응답",
                       turn_number=v1.position)
    v2 = ledger.analyze_and_apply(pairs, cur)      # 동일 재전송
    assert v2.kind == "reroll"
    assert v2.reroll_turn_number == _BASELINE_PAD + 18


def test_fresh_session_positions_unchanged(tmp_path):
    # 신규 세션(pair 없는 첫 메시지)은 패드 없음 — 기존 번호 체계 그대로
    ledger = _ledger(tmp_path)
    v = ledger.analyze_and_apply([], "u0")
    assert v.position == 0
