"""프로브 턴 조립 (구현 계약 §9 assemble 항목) — LLM 0콜.

무상태 3종(vanilla/trim/retrieval)은 fake 프리셋·카드로 **완전 검증**한다.
유상태 2종은 여기서 LLM을 부를 수 없으므로 계약이 요구하는 두 가지만 본다:
스냅샷 부재 시 명확한 실패, 그리고 스냅샷 원본 불변(§8-2). hypa는 요약 콜을
fake로 주입하고, dreaming은 빈 저장소 스냅샷으로 동기 경로를 태운다(0콜).
"""
import json

import pytest

from benchmarks.eval import hypa as hypa_mod
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.run2 import build_wire
from benchmarks.eval.variants import retrieve_turns
from benchmarks.eval.windowing import count
from benchmarks.fixed_haystack import assemble as A
from benchmarks.fixed_haystack import haystack as H
from dreaming import scaffold
from dreaming.assembly import KNOWLEDGE_SEP
from dreaming.records import Fact
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

# 프리셋 최소 형태: 선두 고정부 system + chat 슬롯 (preset2wire.assemble 계약)
_FIXED = "고정부 프리셋 문구가 길게 이어진다. " * 40
_PRESET = {"promptTemplate": [{"type": "plain", "text": _FIXED},
                              {"type": "chat"}]}
# 뮈토스 6.2 실물 형태: chat 뒤에 프리필 왕복이 붙어 와이어의 마지막 user가
# 프로브 발화가 아니다 (실측 106메시지 중 index 105가 영문 프리필)
_PREFILL = "프리필 꼬리 발화 — 이 앞에 붙으면 안 된다."
_PRESET_PREFILL = {"promptTemplate": [
    {"type": "plain", "text": _FIXED},
    {"type": "chat"},
    {"type": "plain", "text": "[프리필 지시]"},
    {"type": "plain", "role": "user", "text": _PREFILL}]}
_CARD = {"name": "소연", "greeting": "어서 와요."}
# 실물(100000/30000)보다 작게 — 12턴짜리 fake 히스토리에서도 트림이 물린다
_RUN_PARAMS = {"max_context": 12000, "max_tokens": 2000, "reasoning": ""}

_N_TURNS = 12
_PROBE = _N_TURNS - 1
_FILLER = "장면 묘사가 길게 이어진다. " * 160        # 발췌 1건 ≈ 2.2K자
_LOST_TURN = 1                                      # 트림에 잘릴 지뢰 턴
_VALUE = "은장도"


def _hs(question=f"품에 넣어 뒀던 {_VALUE}는 어디 있지?"):
    turns = []
    for i in range(_N_TURNS):
        # "어디"를 공유시켜 잘린 pair 전부가 발췌 후보가 되게 한다 —
        # 후보가 1건뿐이면 6000자 캡이 물리지 않아 캡 테스트가 무의미해진다
        user = f"유저 발화 {i} — 오늘은 어디로 갈까 잡담 {i}"
        if i == _LOST_TURN:
            user = f"품에 {_VALUE}를 넣어 뒀어."
        if i == _PROBE:
            user = question
        turns.append(H.FrozenTurn(turn=i, user=user,
                                  assistant=f"응답 {i}. {_FILLER}",
                                  is_probe=i == _PROBE))
    probe = H.FrozenProbe(turn=_PROBE, ptype="recall", fid="led1",
                          fact=f"품에 {_VALUE}", value=_VALUE, wrong="",
                          question=question, fact_turn=_LOST_TURN)
    return H.FixedHaystack(
        haystack_id="fake-hs", source_run="fake.json", origin_variant="dreaming",
        narrator_model="deepseek/deepseek-v4-flash-0731", preset_path="",
        card_path="", run_params=dict(_RUN_PARAMS), turns=turns,
        probes=[probe], ledger=[], validation={}, provenance={})


def _budgets():
    return A.budgets(_PRESET, _CARD, _RUN_PARAMS)


def _last_user(msgs):
    return next(m["content"] for m in reversed(msgs) if m["role"] == "user")


# ── 예산 ───────────────────────────────────────────────────────────────────

def test_budgets_reserve_fixed_part_and_response():
    fixed, budget = _budgets()
    empty = build_wire(_PRESET, _CARD, [])
    assert fixed == sum(count(m["content"]) for m in empty) > 0
    # 응답 예약(maxResponse+50)까지 빼야 실제로 보낼 수 있는 창이 나온다
    assert budget == _RUN_PARAMS["max_context"] - fixed - 2000 - 50
    assert 0 < budget < _RUN_PARAMS["max_context"]


# ── vanilla ────────────────────────────────────────────────────────────────

def test_vanilla_sends_full_history_untrimmed():
    hs = _hs()
    fixed, _ = _budgets()
    a = A.assemble_vanilla(hs, _CARD, _PRESET, _PROBE, fixed,
                           provider_ctx_limit=10 ** 9)
    assert a.overflow is False and a.win_start == 0
    assert a.use_window == H.messages_upto(hs, _PROBE, _CARD)
    assert check_wire_shape(a.msgs) == []
    # 프로브 응답 비누적 (§8-1) — 어떤 경로로도 들어가면 안 된다
    assert all(hs.turns[_PROBE].assistant not in m["content"] for m in a.msgs)


def test_vanilla_overflow_flips_exactly_at_limit():
    """경계는 **프로바이더 공간 추정치**로 잡는다.

    o200k 값으로 재면 한국어 히스토리를 과소계상해 실제로 창을 넘는 턴이
    통과한다 (실물 A의 T89: o200k 122,268 < 128,000, 추정 137,527 > 128,000).
    """
    hs = _hs()
    fixed, _ = _budgets()
    probe = A.assemble_vanilla(hs, _CARD, _PRESET, _PROBE, fixed,
                               provider_ctx_limit=10 ** 9).extra
    n = probe["provider_tokens"]
    assert n > probe["wire_tokens"]                  # 보정은 항상 부풀린다
    # 경계는 초과(>)일 때만 — 딱 맞으면 보낸다
    at = A.assemble_vanilla(hs, _CARD, _PRESET, _PROBE, fixed,
                            provider_ctx_limit=n)
    over = A.assemble_vanilla(hs, _CARD, _PRESET, _PROBE, fixed,
                              provider_ctx_limit=n - 1)
    assert at.overflow is False and at.msgs
    # 나레이터를 부르지 않는다 = 보낼 와이어 자체를 안 만든다 (§4.5)
    assert over.overflow is True and over.msgs == []
    assert over.use_window == at.use_window
    # 같은 상한을 o200k로 쟀다면 통과해 버렸을 구간 — 그게 보정의 존재 이유다
    assert A.assemble_vanilla(hs, _CARD, _PRESET, _PROBE, fixed,
                              provider_ctx_limit=probe["wire_tokens"]
                              ).overflow is True


# ── trim ───────────────────────────────────────────────────────────────────

def test_trim_keeps_window_inside_budget():
    hs = _hs()
    _, budget = _budgets()
    full = H.messages_upto(hs, _PROBE, _CARD)
    a = A.assemble_trim(hs, _CARD, _PRESET, _PROBE, budget)
    assert sum(count(m["content"]) for m in full) > budget      # 실제로 물린다
    assert sum(count(m["content"]) for m in a.use_window) <= budget
    assert a.use_window == full[len(full) - len(a.use_window):]  # 순수 suffix
    assert a.win_start > _LOST_TURN                             # 지뢰 턴은 창밖
    assert check_wire_shape(a.msgs) == []


# ── retrieval ──────────────────────────────────────────────────────────────

def test_retrieval_prepends_excerpt_to_last_user():
    hs = _hs()
    _, budget = _budgets()
    a = A.assemble_retrieval(hs, _CARD, _PRESET, _PROBE, budget)
    tail = _last_user(a.msgs)
    assert tail.startswith(A.EXCERPT_HEADER)                # 마지막 user에 prepend
    assert tail.endswith(hs.turns[_PROBE].user)             # 발화는 뒤에 남는다
    assert _VALUE in tail                                   # 잘린 지뢰 턴을 회수
    assert a.extra["excerpts_kept"] >= 1
    # 창은 발췌 포함본 — value_in_window 판정이 정직해진다
    assert _VALUE in _last_user(a.use_window)
    assert check_wire_shape(a.msgs) == []


def test_retrieval_excerpt_block_respects_cap_and_chunk_boundary():
    hs = _hs()
    _, budget = _budgets()
    a = A.assemble_retrieval(hs, _CARD, _PRESET, _PROBE, budget)
    full = H.messages_upto(hs, _PROBE, _CARD)
    found = retrieve_turns(full, full[-1]["content"], k=A.EXCERPT_K,
                           window=A.assemble_trim(hs, _CARD, _PRESET, _PROBE,
                                                  budget).use_window)
    assert a.extra["excerpt_chars"] <= A.HOT_ZONE_CHAR_BUDGET
    # 캡이 실제로 물려 발췌를 버렸다 (안 물리면 이 테스트가 의미 없다)
    assert 0 < a.extra["excerpts_kept"] < len(found)
    # 버린 단위는 청크 — 남은 발췌는 전부 원문 그대로다 (중간 절단 금지)
    block = _last_user(a.msgs).split("\n\n---\n\n")[0]
    for e in found[:a.extra["excerpts_kept"]]:
        assert e in block


def test_excerpt_block_drops_from_the_offending_chunk():
    kept, n = A._excerpt_block(["a" * 10, "b" * 10, "c" * 10],
                               len(A.EXCERPT_HEADER) + 10 + len(A.EXCERPT_SEP) + 10)
    assert n == 2 and kept == A.EXCERPT_HEADER + "a" * 10 + A.EXCERPT_SEP + "b" * 10
    # 한 발췌가 혼자 예산을 넘으면 자르지 않고 통째로 버린다
    assert A._excerpt_block(["x" * 999], 100) == ("", 0)


def test_retrieval_injects_before_the_utterance_not_the_prefill_tail():
    """프리셋 프리필 꼬리가 있어도 발췌는 프로브 발화 앞에 붙어야 한다.

    dreaming은 `scaffold.split`으로 꼬리를 벗기고 주입한다(sync.py:163,200) —
    조립 후 와이어의 마지막 user에 붙이면 두 변형의 주입 위치가 갈린다.
    """
    hs = _hs()
    _, budget = _budgets()
    a = A.assemble_retrieval(hs, _CARD, _PRESET_PREFILL, _PROBE, budget)
    assert _last_user(a.msgs) == _PREFILL              # 꼬리는 무손상
    spoken = next(m["content"] for m in a.msgs
                  if m["content"].endswith(hs.turns[_PROBE].user))
    assert spoken.startswith(A.EXCERPT_HEADER) and _VALUE in spoken
    assert check_wire_shape(a.msgs) == []


def test_retrieval_without_hits_falls_back_to_trim():
    hs = _hs(question="Nothing here overlaps at all.")
    _, budget = _budgets()
    r = A.assemble_retrieval(hs, _CARD, _PRESET, _PROBE, budget)
    t = A.assemble_trim(hs, _CARD, _PRESET, _PROBE, budget)
    assert r.extra["excerpts_kept"] == 0
    assert r.msgs == t.msgs and r.win_start == t.win_start


# ── 디스패처 ───────────────────────────────────────────────────────────────

def test_build_dispatches_stateless_variants():
    hs = _hs()
    fixed, budget = _budgets()
    for v in A.STATELESS:
        a = A.build(v, hs, _CARD, _PRESET, _PROBE, fixed_tokens=fixed,
                    trim_budget=budget, provider_ctx_limit=10 ** 9)
        assert check_wire_shape(a.msgs) == []
    assert A.build("vanilla", hs, _CARD, _PRESET, _PROBE, fixed_tokens=fixed,
                   trim_budget=budget, provider_ctx_limit=10 ** 9).win_start == 0


def test_build_rejects_unknown_variant():
    hs = _hs()
    with pytest.raises(SystemExit, match="알 수 없는 변형"):
        A.build("hypav9", hs, _CARD, _PRESET, _PROBE, fixed_tokens=1,
                trim_budget=1)


# ── 유상태 2종: 스냅샷 계약 ────────────────────────────────────────────────

def test_missing_snapshots_fail_loudly(tmp_path):
    hs = _hs()
    with pytest.raises(SystemExit, match="hypa 스냅샷 없음"):
        A.load_hypa_snapshot(str(tmp_path), _PROBE)
    with pytest.raises(SystemExit, match="dreaming 스냅샷 없음"):
        A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0")


def test_hypa_runs_on_a_copy_of_the_snapshot(tmp_path, monkeypatch):
    hs = _hs()
    fixed, _ = _budgets()
    # 요약 캐시를 tmp로 돌린다 — 안 그러면 fake 요약이 리포의 공용 캐시
    # (dreaming_data/eval/hypa-cache)에 박혀 실런이 그걸 집을 수 있다.
    # 프로덕션 리디렉션(§9-3)을 그대로 태우고 목적지만 바꾼다.
    monkeypatch.setattr(A, "BENCH_HYPA_CACHE", tmp_path / "hc")
    snap = A.hypa_snapshot_path(str(tmp_path), _PROBE)
    snap.parent.mkdir(parents=True)
    snap.write_text(json.dumps({"summaries": []}), encoding="utf-8")
    data = A.load_hypa_snapshot(str(tmp_path), _PROBE)
    a = A.assemble_hypa(hs, _CARD, _PRESET, _PROBE, fixed, data,
                        settings=hypa_mod.HypaSettings(),
                        max_context=_RUN_PARAMS["max_context"],
                        max_tokens=_RUN_PARAMS["max_tokens"],
                        summarize=lambda msgs: "요약된 과거 장면.")
    assert "hypa_error" not in a.extra
    assert a.memory_text and a.kept_start_msg > 0
    assert check_wire_shape(a.msgs) == []
    # 스냅샷 불변 (§8-2) — 원본 data도 파일도 그대로다
    assert data == {"summaries": []}
    assert json.loads(snap.read_text()) == {"summaries": []}


def test_dreaming_runs_on_a_copy_of_the_snapshot(tmp_path):
    hs = _hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), _PROBE)
    snap.mkdir(parents=True)                       # 인제스천 0턴 = 빈 저장소
    a = A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0",
                            workdir=str(tmp_path / "work"))
    assert a.msgs and a.win_start is None
    assert a.extra["storage_root"].startswith(str(tmp_path / "work"))
    # 동기 경로는 저장소에 쓴다 — 그 쓰기가 복사본에만 남아야 한다 (§8-2)
    assert list(snap.rglob("*.json")) == []
    assert list((tmp_path / "work").rglob("*.json"))


def test_dreaming_reuses_a_given_workdir_across_calls(tmp_path):
    """같은 workdir로 두 번 부를 수 있어야 한다 (--runs 2의 기본 경로).

    copytree가 dirs_exist_ok 없이 돌면 2회차에서 FileExistsError로 즉사하고,
    dirs_exist_ok로 덮으면 1회차 쓰기가 남아 복사본이 스냅샷이 아니게 된다.
    """
    hs = _hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), _PROBE)
    snap.mkdir(parents=True)
    work = str(tmp_path / "wk")
    first = A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0",
                                workdir=work)
    second = A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0",
                                 workdir=work)
    assert first.msgs == second.msgs                  # 1회차 쓰기가 안 샌다
    assert list(snap.rglob("*.json")) == []


def test_dreaming_overflow_is_flagged_not_sent(tmp_path):
    """dreaming도 압축이 안 물면 풀 히스토리를 그대로 보낸다 — 창을 넘으면
    관측 결과로 기록해야지 4xx로 런을 죽이면 안 된다 (§4.5)."""
    hs = _hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), _PROBE)
    snap.mkdir(parents=True)
    fixed, _ = _budgets()
    ok = A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0",
                             fixed_tokens=fixed, workdir=str(tmp_path / "w1"))
    assert ok.overflow is False and ok.msgs
    over = A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0",
                               fixed_tokens=fixed,
                               provider_ctx_limit=ok.extra["provider_tokens"] - 1,
                               workdir=str(tmp_path / "w2"))
    assert over.overflow is True and over.msgs == []


def test_dreaming_without_session_fails_loudly(tmp_path):
    """저장소 깊은 곳의 정규식 위반 대신 hypa_settings 누락과 같은 대접을."""
    hs = _hs()
    with pytest.raises(SystemExit, match="session이 필요하다"):
        A.build("dreaming", hs, _CARD, _PRESET, _PROBE, fixed_tokens=1,
                trim_budget=1, root=str(tmp_path))


def test_dreaming_cleans_up_its_temp_dir_when_process_raises(tmp_path,
                                                             monkeypatch):
    """복사 뒤 예외가 나면 스냅샷 통째 복사본이 /var/folders에 영구 잔류한다."""
    hs = _hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), _PROBE)
    snap.mkdir(parents=True)
    made = []
    real = A.tempfile.mkdtemp

    def spy(**kw):
        made.append(real(**kw))
        return made[-1]

    monkeypatch.setattr(A.tempfile, "mkdtemp", spy)
    monkeypatch.setattr(A.SyncPath, "process",
                        lambda self, raw: (_ for _ in ()).throw(
                            RuntimeError("원장 손상")))
    with pytest.raises(RuntimeError):
        A.assemble_dreaming(hs, _CARD, _PRESET, _PROBE, str(tmp_path), "s0")
    assert made and not A.Path(made[0]).exists()


def test_hypa_fixed_tokens_uses_the_hypa_counter():
    """hypa만 다른 자를 쓴다 — 인제스천과 조립이 같아야 배치가 안 갈린다."""
    assert (A.hypa_fixed_tokens(_PRESET, _CARD)
            == A.budgets(_PRESET, _CARD, _RUN_PARAMS)[0]
            + 3 * len(build_wire(_PRESET, _CARD, [])))


def test_dreaming_injects_knowledge_before_the_utterance(tmp_path):
    """retrieval과 같은 자리(프리필 꼬리 앞)에 지식이 붙는지 — 공정성의 근거.

    인제스천이 100턴 돌며 꼬리 지문을 배운 상태를 스냅샷에 심는다. 못 배운
    첫 요청은 꼬리째 처리돼 주입 위치가 프리필로 밀린다(sync.py:156-160).
    """
    hs = _hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), _PROBE)
    snap.mkdir(parents=True)
    storage = JsonDirStorage(snap)
    claim = f"{_VALUE}는 품에 있다"
    MemoryStore(storage, "s0").save_fact(Fact(claim=claim, status="confirmed"))
    wire = build_wire(_PRESET_PREFILL, _CARD, H.messages_upto(hs, _PROBE, _CARD))
    storage.put("s0/wire", "scaffold",
                {"prev_fp": scaffold.fingerprint(wire),
                 "tail_fp": scaffold.fingerprint(wire[-2:]), "misaligned": 0})

    a = A.assemble_dreaming(hs, _CARD, _PRESET_PREFILL, _PROBE, str(tmp_path),
                            "s0", workdir=str(tmp_path / "work"))
    spoken = next(m["content"] for m in a.msgs
                  if isinstance(m["content"], str)
                  and m["content"].endswith(hs.turns[_PROBE].user))
    assert claim in spoken and spoken != hs.turns[_PROBE].user
    assert a.extra["knowledge_chars"] == len(spoken) - len(hs.turns[_PROBE].user) \
        - len(KNOWLEDGE_SEP)
    assert 0 < a.extra["knowledge_chars"] <= A.HOT_ZONE_CHAR_BUDGET
    # 퍼널 "전달" 판정의 원재료 — 블록 원문이 그대로 노출된다
    assert a.extra["knowledge_text"] == spoken[:a.extra["knowledge_chars"]]
    assert claim in a.extra["knowledge_text"]
