"""프로브 루프 + 리포트 관통 (구현 계약 §6·§7) — 실LLM 0콜.

핵심은 **오프라인 관통**이다: 실제 fix-drm-r0 건초더미로 무상태 3변형 ×
프로브 9개를 완주시키고 결과 JSON → report.render까지 밀어 본다. 나레이터와
judge는 주입 심으로 갈아끼우고, 그 위에 httpx까지 막아 "혹시라도 나가는 콜"을
구조적으로 차단한다.

가짜 나레이터는 **컨텍스트에 값이 남아 있으면 맞히고 없으면 못 맞히는** 최소
모델이다 — 조립 알고리즘의 차이가 결과 표에 실제로 드러나야 관통 테스트가
회귀를 잡는다 (전부 정답/전부 오답이면 교차표가 상수다).
"""
import json
import pathlib

import httpx
import pytest

from benchmarks.eval import transport
from benchmarks.fixed_haystack import assemble as A
from benchmarks.fixed_haystack import haystack as H
from benchmarks.fixed_haystack import narrator
from benchmarks.fixed_haystack import report as R
from benchmarks.fixed_haystack import run as Rn
from dreaming.records import Fact
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

_HS_PATH = pathlib.Path(H.DEFAULT_OUT_DIR) / "fix-drm-r0.json"
# 실물 프리셋(~/Downloads)에 매달리지 않으려고 최소 프리셋을 쓴다 — 관통
# 대상은 프로브 루프지 와이어 충실도가 아니다(그건 assemble 테스트 소관).
_PRESET = {"promptTemplate": [{"type": "plain", "text": "고정부 문구. " * 50},
                              {"type": "chat"}]}
_CARD = {"name": "소연", "greeting": "어서 와요."}
_HIT = "에 대해 또렷하게 이야기했다"          # judge가 읽는 마커
_MISS = "소연은 잠시 눈을 감은 채 아무 말도 하지 않았다."


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """실LLM 0콜을 구조로 보장 — 심을 안 넣은 경로가 있으면 여기서 터진다."""
    def boom(*a, **k):
        raise AssertionError("실제 LLM/HTTP 호출이 일어났다")
    monkeypatch.setattr(narrator, "call", boom)
    monkeypatch.setattr(transport, "key", boom)
    monkeypatch.setattr(transport, "mk_llm", boom)
    monkeypatch.setattr(transport, "make_judge_llm", boom)
    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setattr(httpx, "Client", boom)


class FakeJudge:
    """`scoring.judge_pass`가 부르는 (system, user) -> str 심.

    마커가 프롬프트(=응답 원문 포함)에 있으면 Y. `unparsed_at`번째 호출만
    Y/N이 없는 문장을 돌려줘 파싱 실패(None) 경로를 태운다.
    """

    def __init__(self, unparsed_at: int = 0):
        self.cost, self.calls, self.unparsed_at = 0.0, 0, unparsed_at

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        self.cost += 0.002
        if self.calls == self.unparsed_at:
            return "판정을 내리기 어렵다."
        return "근거: 마커 확인.\n" + ("Y" if _HIT in user else "N")


def _fake_narrate(hs, seen=None):
    """컨텍스트에 값이 살아 있으면 맞히는 나레이터. 항상 같은 문장 틀이라
    프로브 응답이 히스토리로 새면 loop 게이트가 즉시 터진다."""
    q2v = {p.question: p.value for p in hs.probes}

    def narrate(msgs):
        texts = [m["content"] for m in msgs if isinstance(m.get("content"), str)]
        spoken = next(q for q in q2v if texts[-1].endswith(q))
        value = q2v[spoken]
        # 발화 자체는 컨텍스트가 아니지만, 그 **앞에 prepend된 블록**은
        # 컨텍스트다 (retrieval 발췌·dreaming 지식이 거기 붙는다). 메시지째
        # 빼면 두 변형이 조립을 해도 나레이터 눈엔 trim과 똑같아진다.
        ctx = [t[:-len(spoken)] if t.endswith(spoken) else t for t in texts]
        hit = any(value in t for t in ctx)
        if seen is not None:
            seen.append((spoken, hit))
        reply = (f"소연은 천천히 고개를 끄덕이며 {value}{_HIT}." if hit
                 else _MISS)
        return {"reply": reply, "finish": "stop", "prompt": 1000, "cached": 0,
                "completion": 200, "cost": 0.01, "sec": 0.1}
    return narrate


def _prov(**over):
    """`load`가 요구하는 §9-2 키를 채운 최소 provenance.

    키 목록을 베껴 적지 않는다 — 스키마가 늘면 이 fake도 같이 움직여야 한다.
    값도 채운다: 로더는 키 **존재**가 아니라 값을 보므로 전부 ""인 스텁은
    거부된다. 해시만 ""로 둔다 — §9-2가 허용하는 "미측정"이고, 값이 있으면
    `verify_inputs`가 tmp 카드 실물과 대조해 떨어진다.
    """
    prov = {k: f"fake-{k}" for k in H.REQUIRED_PROVENANCE}
    prov["source_store_reused"] = False           # §9-4 준수 선언
    prov["preset_sha256"] = prov["card_sha256"] = ""
    return {**prov, **over}


def _fake_hs(probe_turn=3, value="은장도"):
    """스냅샷 계약을 태우기 위한 소형 건초더미 (실물 파일 무의존)."""
    turns = [H.FrozenTurn(turn=i, user=f"유저 발화 {i} 잡담이 이어진다",
                          assistant=f"응답 {i}. 장면 묘사가 이어진다.",
                          is_probe=i == probe_turn)
             for i in range(probe_turn + 1)]
    q = f"품에 넣어 뒀던 그것은 어디 있지? {probe_turn}"
    turns[probe_turn] = H.FrozenTurn(turn=probe_turn, user=q,
                                     assistant="(재생성 대상)", is_probe=True)
    probe = H.FrozenProbe(turn=probe_turn, ptype="recall", fid="f1",
                          fact="품에 넣어 둔 물건", value=value, wrong="",
                          question=q, fact_turn=0)
    return H.FixedHaystack(
        haystack_id="fake-hs", source_run="fake.json",
        origin_variant="dreaming", narrator_model="fake/model",
        preset_path="", card_path="",
        run_params={"max_context": 12000, "max_tokens": 2000, "reasoning": ""},
        turns=turns, probes=[probe], ledger=[], validation={},
        provenance=_prov())


# ── 관통: 실물 건초더미 × 무상태 3변형 × 프로브 9개 ────────────────────────

@pytest.mark.skipif(not _HS_PATH.is_file(),
                    reason="freeze 산출물 없음 (dreaming_data는 gitignore)")
def test_offline_smoke_runs_three_variants_and_renders(tmp_path):
    hs = H.load(str(_HS_PATH))
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    # freeze가 채점 불가로 표시한 프로브 — 누출(값이 발화에 노출) + 값 길이
    # 미달(1글자라 마스킹·오라클 어느 쪽도 못 지킨다). 하드코딩하면 건초더미를
    # 다시 얼릴 때마다 테스트가 거짓말한다.
    excluded = (set(hs.validation.get("leaky_probes", []))
                | set(hs.validation.get("unprotectable_probes", [])))
    assert excluded                       # 실물엔 최소 1건 있다 (T99 누출)
    results = []
    for variant in A.STATELESS:
        judge = FakeJudge(unparsed_at=2)     # 2번째 채점만 파싱 실패
        r = Rn.run_variant(hs, _CARD, _PRESET, variant, 0, root=str(tmp_path),
                           fixed_tokens=fixed, trim_budget=budget,
                           narrate_fn=_fake_narrate(hs), judge_fn=judge,
                           provider_ctx_limit=H.PROVIDER_CTX_LIMIT)
        assert Rn.save_result(r, tmp_path / "results").is_file()
        results.append(r)

    for r, variant in zip(results, A.STATELESS):
        t = r["totals"]
        assert r["haystack_id"] == "fix-drm-r0" and r["variant"] == variant
        assert r["narrator_model"] == hs.narrator_model      # §8-5
        assert len(r["probes"]) == len(hs.probes) == 9
        # 제외 프로브는 조립도 나레이터도 안 탄다 (§2.2-1)
        assert [p["turn"] for p in r["probes"] if p["skipped"] == "probe_leak"] \
            == hs.validation["leaky_probes"]
        assert [p["turn"] for p in r["probes"]
                if p["skipped"] == "probe_unprotectable"] \
            == hs.validation.get("unprotectable_probes", [])
        live = 9 - len(excluded)
        # 분모 = 제외·미파싱을 뺀 나머지
        assert t["probes_scored"] == live - t["judge_unparsed"]
        assert t["judge_unparsed"] == 1
        assert t["cost"] == pytest.approx(live * 0.01)
        assert t["cost_judge"] == pytest.approx(live * 0.002)

    # 조립 계측이 프로브별로 남는다 (§6 결과 스키마)
    meta = results[0]["assembly_meta"]
    assert meta["fixed_tokens"] == fixed and meta["trim_budget"] == budget
    assert set(meta["by_probe"]) == {str(p.turn) for p in hs.probes
                                     if p.turn not in excluded}
    assert all("wire_tokens" in v for v in meta["by_probe"].values())

    # 변형이 실제로 갈린다 — trim은 창밖 사실을 못 보고, retrieval은 발췌로
    # 일부를 되찾는다. 상수 표라면 관통 테스트가 회귀를 못 잡는다.
    scored = {r["variant"]: {p["turn"]: p["judge"] for p in r["probes"]
                             if not p["skipped"]} for r in results}
    assert scored["vanilla"] != scored["trim"]
    assert any(not p["in_window"] for p in results[1]["probes"]
               if not p["skipped"])                      # trim에 창밖이 있다
    # retrieval은 trim과 같은 창에 발췌를 얹는다 — 값 생존이 줄어들 수 없다
    surv = {r["variant"]: {p["turn"]: p["value_in_window"]
                           for p in r["probes"] if not p["skipped"]}
            for r in results}
    assert all(surv["retrieval"][t] for t, v in surv["trim"].items() if v)

    md = R.render(R.aggregate(results))
    assert "## fix-drm-r0" in md
    for variant in A.STATELESS:
        assert f"| {variant} |" in md
    for section in ("변형 × 프로브 유형", "창내(LITM)", "채점 제외",
                    "비용", "프로브별 통과"):
        assert section in md
    assert "누출제외" in md                                   # T99 라벨
    # 리포트는 구조적 서술만 (§8-7) — 응답·질문 원문을 싣지 않는다
    assert _MISS not in md
    assert all(p["question"][:30] not in md for r in results
               for p in r["probes"])


# ── 오버플로 (§4.5) ────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HS_PATH.is_file(), reason="freeze 산출물 없음")
def test_vanilla_overflow_is_recorded_not_raised(tmp_path):
    """오버플로는 하네스 에러가 아니라 관측 결과다 — 분모에서만 뺀다."""
    hs = H.load(str(_HS_PATH))
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    excluded = (set(hs.validation.get("leaky_probes", []))
                | set(hs.validation.get("unprotectable_probes", [])))
    r = Rn.run_variant(hs, _CARD, _PRESET, "vanilla", 0, root=str(tmp_path),
                       fixed_tokens=fixed, trim_budget=budget,
                       narrate_fn=_fake_narrate(hs), judge_fn=FakeJudge(),
                       provider_ctx_limit=100_000)
    over = [p for p in r["probes"] if p["skipped"] == "context_overflow"]
    # T79는 o200k로 재면 93,610tok으로 통과하지만 프로바이더 추정은
    # 106,305tok으로 넘는다 — 보정 없이는 라이브로 나가 4xx가 됐을 턴이다
    assert [p["turn"] for p in over] == [79, 89]
    assert over[0]["oracle"] is None and over[0]["judge"] is None
    assert over[0]["provider_tokens"] > 100_000 >= over[0]["wire_tokens"]
    assert r["totals"]["skipped_overflow"] == 2
    assert r["totals"]["probes_scored"] == 9 - len(excluded) - 2
    assert "오버플로" in R.render(R.aggregate([r]))


# ── 프로브 응답 비누적 (§3.2 / §8-1) ───────────────────────────────────────

@pytest.mark.skipif(not _HS_PATH.is_file(), reason="freeze 산출물 없음")
def test_probe_replies_never_accumulate(tmp_path):
    """같은 응답을 8번 내리 돌려줘도 loop 리롤이 안 걸린다.

    loop 게이트는 `prior_replies`와 비교한다. 그 값이 얼린 원문의 직전
    assistant 3개라면 프로브 응답을 아무리 반복해도 안 걸리고, 프로브 응답이
    누적되고 있었다면 2번째 프로브부터 ratio 1.0으로 즉시 걸린다.
    """
    hs = H.load(str(_HS_PATH))
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    calls = []

    def flat(msgs):
        calls.append(msgs)
        return {"reply": _MISS, "finish": "stop", "prompt": 1, "cached": 0,
                "completion": 1, "cost": 0.0, "sec": 0.0}

    excluded = (set(hs.validation.get("leaky_probes", []))
                | set(hs.validation.get("unprotectable_probes", [])))
    r = Rn.run_variant(hs, _CARD, _PRESET, "trim", 0, root=str(tmp_path),
                       fixed_tokens=fixed, trim_budget=budget,
                       narrate_fn=flat, judge_fn=FakeJudge())
    # 리롤 0회 = 프로브당 정확히 1콜
    assert len(calls) == 9 - len(excluded)
    assert r["totals"]["rerolls"] == 0 and r["totals"]["flawed"] == 0
    # 어떤 와이어에도 이전 프로브의 응답이 들어 있지 않다
    assert all(_MISS not in m["content"] for msgs in calls for m in msgs)


def test_prior_replies_come_from_the_frozen_script():
    hs = _fake_hs(probe_turn=5)
    assert Rn._prior_replies(hs, 5) == [hs.turns[t].assistant for t in (2, 3, 4)]
    assert Rn._prior_replies(hs, 1) == [hs.turns[0].assistant]   # 경계


# ── dreaming: 미스 분해 → 임시 저장소 정리 (§6-4, §8-2) ────────────────────

def test_dreaming_miss_is_decomposed_before_workspace_cleanup(tmp_path):
    hs = _fake_hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), 3)
    snap.mkdir(parents=True)                     # 인제스천 0턴 = 빈 저장소
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    rec, extra = Rn.run_probe(
        hs, _CARD, _PRESET, "dreaming", hs.probes[0], root=str(tmp_path),
        session="s0", fixed_tokens=fixed, trim_budget=budget,
        narrate_fn=lambda msgs: {"reply": _MISS, "finish": "stop", "prompt": 1,
                                 "cached": 0, "completion": 1, "cost": 0.0,
                                 "sec": 0.0},
        judge_fn=FakeJudge())
    assert rec["judge"] is False
    # 저장소에 값이 없다 = 저장 실패. 임시 복사본이 지워진 뒤였다면 이 판정
    # 자체가 불가능하다 (분해가 정리보다 먼저다).
    assert rec["miss_cause"] == "storage_fail"
    assert rec["value_in_store"] is False        # 퍼널 "추출" — 무조건 측정
    assert rec["in_window"] is True              # 풀 히스토리 상한
    assert not pathlib.Path(extra["workspace"]).exists()   # 정리됐다
    assert list(snap.rglob("*.json")) == []                # 스냅샷 원본 불변


def test_dreaming_store_check_runs_even_when_judge_passes(tmp_path):
    """추출 지표는 judge와 독립 — 통과한 프로브도 저장소를 잰다."""
    hs = _fake_hs()
    snap = A.dreaming_snapshot_dir(str(tmp_path), 3)
    snap.mkdir(parents=True)
    p = hs.probes[0]
    MemoryStore(JsonDirStorage(snap), "s0").save_fact(
        Fact(claim=f"{p.value}는 품에 있다", status="confirmed"))
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    rec, _extra = Rn.run_probe(
        hs, _CARD, _PRESET, "dreaming", p, root=str(tmp_path),
        session="s0", fixed_tokens=fixed, trim_budget=budget,
        narrate_fn=lambda msgs: {"reply": p.value + _HIT, "finish": "stop",
                                 "prompt": 1, "cached": 0, "completion": 1,
                                 "cost": 0.0, "sec": 0.0},
        judge_fn=FakeJudge())
    assert rec["judge"] is True
    assert rec["value_in_store"] is True
    assert rec["miss_cause"] == "-"              # 미스가 아니니 분해 없음


# ── 스킵 라벨 · 집계 규약 ──────────────────────────────────────────────────

def test_hypa_error_skips_the_probe(tmp_path, monkeypatch):
    hs = _fake_hs()
    snap = A.hypa_snapshot_path(str(tmp_path), 3)
    snap.parent.mkdir(parents=True)
    snap.write_text(json.dumps({"summaries": []}), encoding="utf-8")
    monkeypatch.setattr(A, "assemble_hypa",
                        lambda *a, **k: A.Assembly(
                            msgs=[], use_window=[],
                            extra={"hypa_error": "요약 불가"}))
    rec, _ = Rn.run_probe(hs, _CARD, _PRESET, "hypa", hs.probes[0],
                          root=str(tmp_path), session="s0", fixed_tokens=1,
                          trim_budget=1, narrate_fn=None, judge_fn=None,
                          hypa_settings=object())
    assert rec["skipped"] == "hypa_error" and rec["judge"] is None
    assert rec["hypa_error"] == "요약 불가"


def test_in_window_rule_per_variant():
    """창내/창외 판정은 변형마다 다른 자로 잰다 (§6-5)."""
    p = _fake_hs().probes[0]                       # fact_turn = 0
    a = A.Assembly(msgs=[], use_window=[], win_start=2, kept_start_msg=5)
    # 풀 히스토리 — dreaming은 프록시 압축 때문에 상한값이다
    assert Rn._in_window("vanilla", p, a, _CARD) is True
    assert Rn._in_window("dreaming", p, a, _CARD) is True
    assert Rn._in_window("trim", p, a, _CARD) is False
    assert Rn._in_window("retrieval", p, a, _CARD) is False
    # hypa는 턴이 아니라 메시지 인덱스로 자른다 — greeting 있으면 턴0=index1
    assert Rn._in_window("hypa", p, a, _CARD) is False
    kept0 = A.Assembly(msgs=[], use_window=[], kept_start_msg=0)
    assert Rn._in_window("hypa", p, kept0, _CARD) is True


def test_unparsed_judge_leaves_the_denominator(tmp_path):
    hs = _fake_hs()
    fixed, budget = A.budgets(_PRESET, _CARD, hs.run_params)
    r = Rn.run_variant(hs, _CARD, _PRESET, "vanilla", 0, root=str(tmp_path),
                       fixed_tokens=fixed, trim_budget=budget,
                       narrate_fn=_fake_narrate(hs),
                       judge_fn=FakeJudge(unparsed_at=1))
    assert r["totals"]["judge_unparsed"] == 1
    assert r["totals"]["probes_scored"] == 0
    assert R.aggregate([r])["fake-hs"]["variants"]["vanilla"]["total"] == (0, 0)


def test_report_keeps_haystacks_in_separate_sections(tmp_path):
    """두 건초더미를 절대 평균 내지 않는다 (설계 §2.3)."""
    a = _fake_hs()
    fixed, budget = A.budgets(_PRESET, _CARD, a.run_params)
    rows = []
    for hid in ("fix-drm-r0", "fix-van-r0"):
        hs = H.FixedHaystack(**{**a.__dict__, "haystack_id": hid})
        rows.append(Rn.run_variant(
            hs, _CARD, _PRESET, "vanilla", 0, root=str(tmp_path),
            fixed_tokens=fixed, trim_budget=budget,
            narrate_fn=_fake_narrate(hs), judge_fn=FakeJudge()))
    agg = R.aggregate(rows)
    assert set(agg) == {"fix-drm-r0", "fix-van-r0"}
    md = R.render(agg)
    assert md.count("### 변형 × 프로브 유형 (judge 통과/분모)") == 2


def test_report_grid_keeps_scored_runs_when_only_some_are_skipped():
    """3반복 중 1회만 스킵된 셀이 "전부 실패"로 보이면 안 된다.

    오버플로는 결정론이라 전 반복이 같이 스킵되지만 hypa_error는 아니다.
    """
    def _r(run, skipped):
        p = {"turn": 3, "ptype": "recall", "distance_turns": 3,
             "judge": None if skipped else True, "oracle": True,
             "skipped": skipped, "in_window": True}
        return {"haystack_id": "h", "variant": "hypa", "run": run,
                "probes": [p], "totals": {}}

    agg = R.aggregate([_r(0, "hypa_error"), _r(1, ""), _r(2, "")])
    md = R.render(agg)
    assert agg["h"]["variants"]["hypa"]["grid"][3] == (2, 2, 1, "hypa_error")
    assert "2/2 (hypa실패1)" in md


def test_report_finds_ingest_meta_for_results_written_elsewhere(tmp_path,
                                                               monkeypatch):
    """`run --out`으로 결과를 옮겨도 인제스천 상각 비용이 사라지면 안 된다."""
    root = tmp_path / "fix-x"
    root.mkdir()
    (root / "ingest-meta.json").write_text(
        json.dumps({"hypa": {"summary_calls": 7, "summary_cost": 0.12}}),
        encoding="utf-8")
    res = tmp_path / "elsewhere" / "vanilla-run0.json"
    res.parent.mkdir()
    res.write_text(json.dumps({"haystack_id": "fix-x"}), encoding="utf-8")
    monkeypatch.setattr(R, "DEFAULT_OUT_DIR", str(tmp_path))
    assert R.load_ingest_meta([str(res)])["fix-x"]["hypa"]["summary_calls"] == 7


def test_report_shows_ingest_amortized_cost():
    """유상태 변형의 인제스천 비용이 빠지면 실제보다 싸 보인다."""
    md = R.render({"h1": {"meta": {"origin_variant": "dreaming",
                                   "narrator_model": "m", "fixed_tokens": 1,
                                   "provider_ctx_limit": 2},
                          "axis": [], "variants": {}}},
                  {"h1": {"dreaming": {"dream_cycles": 4, "dream_cost": 0.0,
                                       "quarantine_count": 0, "dream_sec": 3},
                          "hypa": {"summary_calls": 7, "summary_cost": 0.12,
                                   "summary_truncated": 0}}})
    assert "꿈 4사이클" in md and "미측정" in md      # 0.0은 미측정이라 명시
    assert "hypa 요약 7콜" in md


# ── CLI 배선 ───────────────────────────────────────────────────────────────

def test_cli_rejects_unknown_variant():
    with pytest.raises(SystemExit, match="알 수 없는 변형"):
        Rn.main(["--haystack", "fix-drm-r0", "--variants", "hypav9"])


def test_cli_writes_one_result_per_variant_and_run(tmp_path, monkeypatch):
    """--variants/--runs/--out 배선 — 실물 프리셋·카드·LLM 없이 관통한다."""
    card = tmp_path / "card.json"
    card.write_text(json.dumps(_CARD), encoding="utf-8")
    base = _fake_hs()
    hs = H.FixedHaystack(**{**base.__dict__, "card_path": str(card)})
    H.save(hs, str(tmp_path / "fake-hs.json"))
    monkeypatch.setattr(Rn, "DEFAULT_OUT_DIR", str(tmp_path))
    monkeypatch.setattr(Rn, "decode_risup", lambda path: _PRESET)
    monkeypatch.setattr(Rn, "default_narrate", _fake_narrate)
    monkeypatch.setattr(transport, "make_judge_llm", FakeJudge)

    out = tmp_path / "res"
    assert Rn.main(["--haystack", "fake-hs", "--variants", "vanilla,trim",
                    "--runs", "2", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.glob("*.json")) == [
        "trim-run0.json", "trim-run1.json",
        "vanilla-run0.json", "vanilla-run1.json"]
    doc = json.loads((out / "vanilla-run0.json").read_text(encoding="utf-8"))
    assert doc["haystack_id"] == "fake-hs" and doc["run"] == 0
    assert doc["assembly_meta"]["session"] == Rn.ingest.SESSION
    # §9-2 귀속 사슬의 마지막 칸 — 결과 JSON만 손에 남아도 어느 코드·어느
    # 조립 입력·어느 재freeze로 잰 것인지 복원할 수 있어야 한다
    assert doc["provenance"] == hs.provenance


def test_cli_refuses_to_write_results_into_a_session_directory(
        tmp_path, monkeypatch):
    """§9-3: `--out`이 무검증이면 네임스페이스가 플래그 하나로 무력해진다."""
    card = tmp_path / "card.json"
    card.write_text(json.dumps(_CARD), encoding="utf-8")
    hs = H.FixedHaystack(**{**_fake_hs().__dict__, "card_path": str(card)})
    H.save(hs, str(tmp_path / "fake-hs.json"))
    monkeypatch.setattr(Rn, "DEFAULT_OUT_DIR", str(tmp_path))
    monkeypatch.setattr(Rn, "decode_risup", lambda path: _PRESET)
    with pytest.raises(SystemExit, match="§9-3"):
        Rn.main(["--haystack", "fake-hs", "--variants", "vanilla",
                 "--out", str(H.ROOT / "dreaming_data" / "fix-drm-r0")])


def test_cli_writes_into_the_dedicated_runs_namespace(tmp_path, monkeypatch):
    """스펙 §9-3: 산출물은 `fixed_haystack/{id}/runs/`에만 쓴다.

    기존 세션 디렉터리에 쓰면 §9가 막으려는 오염이 반대 방향으로 생긴다 —
    벤치 산출물이 다음 벤치의 입력 후보로 섞여 들어간다.
    """
    card = tmp_path / "card.json"
    card.write_text(json.dumps(_CARD), encoding="utf-8")
    base = _fake_hs()
    hs = H.FixedHaystack(**{**base.__dict__, "card_path": str(card)})
    H.save(hs, str(tmp_path / "fake-hs.json"))
    monkeypatch.setattr(Rn, "DEFAULT_OUT_DIR", str(tmp_path))
    monkeypatch.setattr(Rn, "decode_risup", lambda path: _PRESET)
    monkeypatch.setattr(Rn, "default_narrate", _fake_narrate)
    monkeypatch.setattr(transport, "make_judge_llm", FakeJudge)

    assert Rn.main(["--haystack", "fake-hs", "--variants", "vanilla"]) == 0
    assert [p.name for p in (tmp_path / "fake-hs" / "runs").glob("*.json")] \
        == ["vanilla-run0.json"]
    assert not (tmp_path / "fake-hs" / "results").exists()


def test_report_cli_without_args_returns_usage(capsys):
    assert R.main([]) == 2
    assert "usage" in capsys.readouterr().out
