"""임계 강제압축 폴백 판정 (스펙 §6.3) — 5분 미만 간격 연속 채팅에서
유휴 타이머가 영원히 리셋돼도 히스토리가 무한 누적되지 않아야 한다."""
from dreaming.pressure import MIN_BACKLOG_TURNS, prompt_chars, should_force


def test_prompt_chars_counts_string_contents():
    msgs = [{"role": "system", "content": "12345"},
            {"role": "user", "content": "abc"}]
    assert prompt_chars(msgs) == 8


def test_prompt_chars_counts_multipart_text():
    # RisuAI 비전 형식: content가 part 리스트 — text part만 센다
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "abcd"},
        {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    assert prompt_chars(msgs) == 4


def test_prompt_chars_ignores_missing_content():
    assert prompt_chars([{"role": "assistant"}]) == 0


def test_forces_at_threshold_with_backlog():
    assert should_force(1000, threshold=1000,
                        backlog_turns=MIN_BACKLOG_TURNS) is True


def test_no_force_below_threshold():
    assert should_force(999, threshold=1000,
                        backlog_turns=MIN_BACKLOG_TURNS) is False


def test_no_force_below_min_backlog():
    # backlog가 얕으면 매 턴 Flash 콜이 돼 상각(~2%)이 깨진다 — 대기
    assert should_force(5000, threshold=1000,
                        backlog_turns=MIN_BACKLOG_TURNS - 1) is False


def test_zero_threshold_disables():
    assert should_force(999999, threshold=0, backlog_turns=100) is False
