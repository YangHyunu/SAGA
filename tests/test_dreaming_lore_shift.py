"""1안(lore_shift) — keyed 로어 이동이 프리픽스를 byte-stable하게 만드는지."""

from dreaming.lore_shift import shift_keyed

CONST = "### 발타니아\n" + "로마풍 도시국가. " * 20
KEYED_A = "### 흐라픈\n" + "뱃사공. 과묵하고 흉터가 많다. " * 10
KEYED_B = "### 바루스\n" + "죽어가는 학자. 넥타르를 연구했다. " * 10
KEYED = [KEYED_A, KEYED_B]


def wire(active):
    """RisuAI처럼 로어를 첫 system에 병합한 메시지를 만든다."""
    lore = "\n\n".join(active)
    return [
        {"role": "system", "content": f"프리셋 head\n\n{CONST}\n\n{lore}\n\nPHI tail"},
        {"role": "assistant", "content": "greeting"},
        {"role": "user", "content": "안녕"},
    ]


def test_moves_keyed_out_of_prefix():
    out, n = shift_keyed(wire([KEYED_A]), KEYED)
    assert n == 1
    assert KEYED_A.strip() not in out[0]["content"]
    assert KEYED_A.strip() in out[-1]["content"]
    assert out[-1]["content"].startswith("<active_lorebook>")
    assert out[-1]["content"].endswith("안녕")


def test_prefix_byte_stable_across_turns():
    """턴마다 다른 keyed가 켜져도 msg[0]은 동일해야 한다 — 1안의 존재 이유."""
    s1, _ = shift_keyed(wire([KEYED_A]), KEYED)
    s2, _ = shift_keyed(wire([KEYED_B]), KEYED)
    s3, _ = shift_keyed(wire([KEYED_A, KEYED_B]), KEYED)
    assert s1[0]["content"] == s2[0]["content"] == s3[0]["content"]
    assert CONST.strip() in s1[0]["content"]   # 끝 공백은 정규화에 먹힐 수 있다


def test_preserves_appearance_order():
    out, n = shift_keyed(wire([KEYED_B, KEYED_A]), KEYED)
    assert n == 2
    body = out[-1]["content"]
    assert body.index(KEYED_B.strip()) < body.index(KEYED_A.strip())


def test_noop_when_keyed_empty():
    """기능 OFF(keyed 미설정)면 완전 무가공."""
    msgs = wire([KEYED_A])
    assert shift_keyed(msgs, []) == (msgs, 0)


def test_no_match_still_normalizes():
    """매치 0이어도 정규화는 돈다 — 안 돌면 원문 \\n{3,} 때문에
    매치 있는 턴과 바이트가 갈린다 (실캡처 턴1에서 재현)."""
    out, n = shift_keyed(wire([]), KEYED)
    assert n == 0
    assert "<active_lorebook>" not in out[-1]["content"]
    stable, _ = shift_keyed(wire([KEYED_A]), KEYED)
    assert out[0]["content"] == stable[0]["content"]


def test_original_not_mutated():
    msgs = wire([KEYED_A])
    before = msgs[0]["content"]
    shift_keyed(msgs, KEYED)
    assert msgs[0]["content"] == before
