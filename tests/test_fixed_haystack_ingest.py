"""인제스천 스냅샷 불변식 (구현 계약 §9 ingest 항목) — LLM 0콜.

실LLM은 fake로 대체하고(꿈=추출 0건, hypa 요약=고정 문자열) 프리셋·카드는
최소 dict로 만든다. 검증 대상은 두 가지뿐이다:
 1. 스냅샷이 **프로브 턴 직전** 상태인가 (그 턴의 원문이 안 들어갔는가)
 2. 스냅샷의 복사본에서 프로브를 돌려도 원본 스냅샷이 안 변하는가 (계약 §8-2)
"""
import asyncio
import dataclasses
import json
import pathlib
import shutil

import httpx
import pytest

from benchmarks.eval import hypa
from benchmarks.eval.run2 import build_wire
from benchmarks.fixed_haystack import assemble
from benchmarks.fixed_haystack import haystack as H
from benchmarks.fixed_haystack import ingest
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath

# 최소 프리셋 — promptTemplate의 plain(고정부) + chat(히스토리 자리)만.
# 실물 .risup 없이도 run2.build_wire가 그대로 돈다.
PRESET = {"promptTemplate": [{"type": "plain", "text": "고정부 지시문 " * 20},
                             {"type": "chat"}]}
CARD = {"name": "소연", "greeting": "어서 와요."}
_BODY = "장면 서술이 이어진다. " * 12          # 요약 발동에 쓸 부피


def _prov(**over):
    """CLI 경로가 save→load를 관통한다 — load는 §9-2 키를 전부 요구한다.

    키 목록은 베껴 적지 않는다(스키마가 늘면 fake도 같이 움직인다). 값도
    채운다 — 로더는 키 **존재**가 아니라 값을 보므로 전부 ""인 스텁은
    거부된다(그건 git 없는 머신의 깨진 freeze가 만드는 산물과 구분이 안 된다).
    해시만 ""로 둔다: §9-2가 허용하는 "미측정"이고, 값이 있으면
    `verify_inputs`가 tmp 카드/프리셋 실물과 대조해 떨어진다.
    """
    prov = {k: f"fake-{k}" for k in H.REQUIRED_PROVENANCE}
    prov["source_store_reused"] = False           # §9-4 준수 선언
    prov["preset_sha256"] = prov["card_sha256"] = ""
    return {**prov, **over}


def _hs(n_turns=5, probe_turns=(2, 4), max_context=100000):
    turns = [H.FrozenTurn(i, f"유저 발화 {i} {_BODY}",
                          f"응답 본문 {i} {_BODY}", i in probe_turns)
             for i in range(n_turns)]
    probes = [H.FrozenProbe(turn=t, ptype="recall", fid="", fact="사실",
                            value="값", wrong="", question=f"유저 발화 {t}",
                            fact_turn=0) for t in probe_turns]
    return H.FixedHaystack(
        haystack_id="fake-hs", source_run="", origin_variant="dreaming",
        narrator_model="fake/model", preset_path="", card_path="",
        run_params={"max_context": max_context, "max_tokens": 1000,
                    "reasoning": ""},
        turns=turns, probes=probes, ledger=[], validation={},
        provenance=_prov())


class _FakeDreamLLM:
    """추출 0건 — 꿈 사이클은 돌되 지식은 안 만든다 (LLM 0콜 대체)."""

    def __init__(self):
        self.calls = 0

    async def complete(self, system, user):
        self.calls += 1
        return '{"episodes":[],"facts":[],"commits":[],"actors":[]}'


def _ingest_meta(hs, **over):
    """`require_meta`가 요구하는 §9-2 꿈 축 메타 (키 목록은 파생)."""
    meta = {k: f"fake-{k}" for k in ingest.REQUIRED_META}
    meta["haystack_id"] = hs.haystack_id
    meta["session"] = ingest.SESSION
    meta["source_run_sha256"] = hs.provenance["source_run_sha256"]
    return {**meta, **over}


def _fingerprint(root: pathlib.Path):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*.json"))}


def _raw_turns(store_root: pathlib.Path):
    return {json.loads(p.read_text())["turn_number"]
            for p in (store_root / ingest.SESSION / "raw").glob("*.json")}


# ── dreaming ──────────────────────────────────────────────────────────────

def test_dreaming_snapshot_is_pre_probe_state(tmp_path):
    hs = _hs()
    llm = _FakeDreamLLM()
    # 픽스처가 5턴이라 프로덕션 양자(10턴)로는 꿈이 영영 안 찬다 — 양자 2
    meta = ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path, dream_llm=llm,
                                  batch_turns=2)

    snap = tmp_path / "dreaming" / "snap"
    # 프로브 턴 직전 = 그 턴의 원문이 아직 저장소에 없다
    assert _raw_turns(snap / "t2") == {0, 1}
    assert _raw_turns(snap / "t4") == {0, 1, 2, 3}
    assert _raw_turns(tmp_path / "dreaming" / "live") == {0, 1, 2, 3, 4}
    assert meta["snapshots"] == [2, 4]
    assert meta["quarantine_count"] == 0
    # backlog가 프로브마다 소진됐다 — 꿈 횟수가 프로브 간 통제된다(계약 §5.1)
    assert llm.calls == meta["dream_cycles"] > 0


def test_dreaming_snapshot_survives_probe_run(tmp_path):
    """불변식 2 — 프로브는 복사본에서 돌고 스냅샷 원본은 안 변한다."""
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    snap_t2 = tmp_path / "dreaming" / "snap" / "t2"
    before = _fingerprint(snap_t2)

    work = tmp_path / "work"
    shutil.copytree(snap_t2, work)
    sp = SyncPath(JsonDirStorage(work), ingest.SESSION,
                  mark_cache_enabled=True)
    raw = build_wire(PRESET, CARD, H.messages_upto(hs, 2, CARD))
    _out, verdict = sp.process(raw)
    sp.record_response(verdict, raw, "라이브 재생성 응답")

    # 복사본은 실제로 변했고(검증이 공회전이 아님을 보장), 원본은 그대로다
    assert _raw_turns(work) == {0, 1, 2}
    assert _fingerprint(snap_t2) == before


def test_dreaming_rerun_skips_without_force(tmp_path):
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    llm2 = _FakeDreamLLM()
    meta = ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path, dream_llm=llm2)
    # 인제스천은 비싸다 — 완주한 스냅샷이 있으면 LLM을 한 번도 안 부른다
    assert meta["skipped"] is True and llm2.calls == 0


def test_empty_snapshot_dir_does_not_count_as_complete(tmp_path):
    """존재 검사만 하면 **내용 없는** 스냅샷이 "완주"로 통과한다.

    copytree 중 kill이나 동기화 사고로 디렉터리 껍데기만 남으면 → 인제스천
    스킵 → 조립이 빈 스토어를 복사 → dreaming이 기억 0으로 측정된다. 그건
    알고리즘 성능이 아니라 하네스 산물인데 결과 JSON엔 구분이 안 남는다.
    (실물 `fix-drm-r0/dreaming/snap/t79`가 그 상태로 디스크에 있었다.)
    """
    hs = _hs()
    for t in (2, 4):
        (tmp_path / "dreaming" / "snap" / f"t{t}").mkdir(parents=True)
    assert ingest._snapped_dirs(tmp_path) == set()
    # → 스킵이 아니라 실제 인제스천이 돈다 (양자 2 — 픽스처 5턴)
    llm = _FakeDreamLLM()
    meta = ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path, dream_llm=llm,
                                  batch_turns=2)
    assert not meta.get("skipped") and llm.calls > 0
    assert _raw_turns(tmp_path / "dreaming" / "snap" / "t2") == {0, 1}


def test_live_residue_refuses_instead_of_appending(tmp_path):
    """부분 스냅샷은 --force를 요구하면서 부분 live는 조용히 이어붙였다.

    첫 프로브 턴 전에 죽으면 snap은 0개(가드 통과)인데 live엔 앞 턴 원장이
    남는다 → 재실행이 같은 저장소에 전 턴을 다시 리플레이해 원장·쌍 해시가
    이중화되고, 그 위에서 뜬 스냅샷이 전 프로브에 쓰인다.
    """
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    shutil.rmtree(tmp_path / "dreaming" / "snap")      # live만 남은 상태 재현
    assert ingest._snapped_dirs(tmp_path) == set()     # 스냅샷 가드는 통과
    with pytest.raises(SystemExit, match="live"):
        ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                               dream_llm=_FakeDreamLLM())


# ── §9-2 꿈 축 메타 (require_meta) ────────────────────────────────────────

def test_reused_snapshots_require_dream_axis_metadata(tmp_path):
    """스킵은 "옛 산출물을 그대로 쓰겠다"는 뜻 — 그때 조건 기록을 확인한다.

    §9-2의 "생성 설정" 중 꿈 축(모델·트리거·git)은 동결 파일이 아니라 이
    산출물이 진다. 동결 파일엔 로더 거부가 있는데 여기 없으면 의무가 무상태
    축에만 걸리고, 정작 병리(1턴 꿈·IDLE 타이머)가 사는 유상태 축은 무방비다.
    """
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    meta_path = tmp_path / "ingest-meta.json"
    doc = json.loads(meta_path.read_text())
    # 실물 `fix-drm-r0/ingest-meta.json`이 정확히 이 상태로 남아 있었다
    for k in ("git_commit", "git_dirty", "dream_trigger", "dream_cap"):
        doc["dreaming"].pop(k)
    meta_path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="dream_trigger"):
        ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                               dream_llm=_FakeDreamLLM())


def test_snapshot_from_a_different_frozen_text_is_refused(tmp_path):
    """동결 파일과 스냅샷을 잇는 식별자 — 없으면 조합이 어긋나도 모른다.

    실물에선 동결 파일(14:15)이 스냅샷 메타(14:04)보다 나중에 만들어졌는데
    둘을 잇는 값이 하나도 없었다. `frozen_at`은 재freeze마다 바뀌어 못박는
    자로 못 쓰므로 원문 해시로 잇는다.
    """
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    other = dataclasses.replace(
        hs, provenance=_prov(source_run_sha256="다른원문해시"))
    with pytest.raises(SystemExit, match="source_run_sha256"):
        ingest.require_meta(tmp_path, other)
    assert ingest.require_meta(tmp_path, hs)["dream_trigger"] == (
        ingest.DREAM_TRIGGER)


def test_dreaming_partial_snapshot_refuses(tmp_path):
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    shutil.rmtree(tmp_path / "dreaming" / "snap" / "t4")
    with pytest.raises(SystemExit, match="부분만"):
        ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                               dream_llm=_FakeDreamLLM())


# ── hypa ──────────────────────────────────────────────────────────────────

@pytest.fixture
def hypa_offline(tmp_path, monkeypatch):
    """실물 export(~/Downloads)와 리포의 요약 캐시에서 테스트를 떼어낸다.

    ingest는 HYPA_EXPORT를 자기 네임스페이스에서 읽으므로 여기를 갈아끼운다
    (config 쪽을 패치하면 import-시점 별칭이라 no-op이다).
    캐시는 프로덕션 경로(`assemble.bench_summary_cache`)를 그대로 태우고
    목적지만 tmp로 돌린다 — summarize를 대신 갈아끼우면 그 리디렉션이
    테스트에서 통째로 우회돼 §9-3 배선이 검증되지 않는다.
    """
    export = tmp_path / "export.json"
    export.write_text(json.dumps(
        {"data": {"settings": {"maxChatsPerSummary": 2,
                               "summarizationPrompt": "요약해라"}}}))
    monkeypatch.setattr(ingest, "HYPA_EXPORT", str(export))
    monkeypatch.setattr(assemble, "BENCH_HYPA_CACHE", tmp_path / "hypa-cache")


def _fake_summarize(calls):
    def send(messages):
        calls.append(messages)
        return f"요약 {len(calls)}"
    return send


def _hypa_hs():
    # 9턴 @900 — 요약이 T4부터 실제로 발동해 두 스냅샷이 모두 비어 있지 않은
    # 최소 조합(실측). 짧거나 예산이 크면 스냅샷이 전부 빈 요약이라
    # "직전 상태" 검증이 공회전한다.
    return _hs(n_turns=9, probe_turns=(6, 8), max_context=900)


def test_hypa_snapshot_is_pre_probe_state(tmp_path, hypa_offline):
    hs = _hypa_hs()
    calls = []
    meta = ingest.ingest_hypa(hs, PRESET, CARD, tmp_path,
                             summarize=_fake_summarize(calls))
    snap = tmp_path / "hypa" / "snap"
    t6 = json.loads((snap / "t6.json").read_text())
    t8 = json.loads((snap / "t8.json").read_text())

    memos6 = [m for s in t6["summaries"] for m in s["chatMemos"]]
    # memo = 히스토리 인덱스 (greeting 포함) → 턴 t의 user는 m{1+2t}.
    # 프로브 턴(6)의 발화는 스냅샷 시점에 아직 히스토리에 없다.
    assert memos6 and "m13" not in memos6
    assert 0 < len(t6["summaries"]) < len(t8["summaries"])
    assert meta["summary_calls"] == 0        # 프로덕션 콜은 0 (fake send)
    assert len(calls) > 0 and meta["error"] == ""


def test_hypa_snapshot_survives_probe_run(tmp_path, hypa_offline):
    """불변식 2(hypa판) — 프로브는 deepcopy에서 돌고 스냅샷 파일은 그대로."""
    import copy

    hs = _hypa_hs()
    ingest.ingest_hypa(hs, PRESET, CARD, tmp_path,
                       summarize=_fake_summarize([]))
    snap_t6 = tmp_path / "hypa" / "snap" / "t6.json"
    before = snap_t6.read_bytes()

    S = hypa.load_hypa_settings(ingest.HYPA_EXPORT)
    loaded = json.loads(snap_t6.read_text())
    data = copy.deepcopy(loaded)
    fixed = sum(hypa.tok_chat(m) for m in build_wire(PRESET, CARD, []))
    _mem, _kept, _ksm, data, err = hypa.hypa_step(
        H.messages_upto(hs, 6, CARD), fixed, S, data,
        _fake_summarize([]), hs.run_params["max_context"],
        hs.run_params["max_tokens"])

    assert err is None
    # 프로브 실행이 data를 실제로 늘렸는데도(검증이 공회전이 아님) 파일은 그대로
    assert len(data["summaries"]) > len(loaded["summaries"])
    assert snap_t6.read_bytes() == before


def test_hypa_rerun_skips_without_force(tmp_path, hypa_offline):
    hs = _hypa_hs()
    ingest.ingest_hypa(hs, PRESET, CARD, tmp_path,
                       summarize=_fake_summarize([]))
    calls = []
    meta = ingest.ingest_hypa(hs, PRESET, CARD, tmp_path,
                             summarize=_fake_summarize(calls))
    assert meta["skipped"] is True and calls == []


def test_ingest_meta_holds_both_variants(tmp_path, hypa_offline):
    hs = _hs()
    ingest.ingest_dreaming(hs, PRESET, CARD, tmp_path,
                           dream_llm=_FakeDreamLLM())
    ingest.ingest_hypa(hs, PRESET, CARD, tmp_path,
                       summarize=_fake_summarize([]))
    meta = json.loads((tmp_path / "ingest-meta.json").read_text())
    assert set(meta) == {"dreaming", "hypa"}
    # 스냅샷 저장소의 세션 id는 assemble/run이 읽어야 한다 — 메타에 박제
    assert meta["dreaming"]["session"] == ingest.SESSION
    assert meta["dreaming"]["turn_sec"].keys() == {str(i) for i in range(5)}
    # 스펙 §9-2의 "생성 설정" 중 꿈 축은 동결 파일이 아니라 여기 산출물의
    # 속성이다 — 코드 버전과 트리거 정책이 없으면 나중에 산출물만 보고
    # IDLE 병리 데이터와 구분할 수 없다.
    for v in ("dreaming", "hypa"):
        assert set(meta[v]) >= {"git_commit", "git_dirty"}
    assert meta["dreaming"]["dream_trigger"] == ingest.DREAM_TRIGGER
    assert meta["dreaming"]["dream_cap"] == ingest.DREAM_CAP


def test_summary_cache_writes_into_the_bench_namespace(tmp_path, monkeypatch):
    """§9-3은 값이 아니라 **쓰기 위치**에 대한 규칙이다.

    기본값이면 요약 캐시가 `dreaming_data/eval/hypa-cache/`에 떨어져 벤치
    산출물이 오염 지대에서 eval v2 하네스와 같은 디렉터리를 공유한다.
    `hypa.summarize`의 cache_dir은 def 시점에 바인딩된 기본인자라 모듈 속성
    패치로는 안 바뀐다 — 호출부를 감싸 인자로 넘기는 게 유일한 손잡이다.
    """
    bench = tmp_path / "ns-cache"
    monkeypatch.setattr(assemble, "BENCH_HYPA_CACHE", bench)
    before = sorted(hypa.CACHE_DIR.glob("*.txt"))
    with assemble.bench_summary_cache() as target:
        assert target == bench
        hypa.summarize(lambda msgs: "요약 본문", [{"role": "user", "memo": "m0",
                                               "content": "장면"}],
                       hypa.HypaSettings())
    assert [p.suffix for p in bench.glob("*.txt")] == [".txt"]
    assert sorted(hypa.CACHE_DIR.glob("*.txt")) == before   # eval 캐시 무접촉
    # 되돌린다 — 전역 오염은 같은 프로세스의 eval 하네스까지 끌고 간다
    assert hypa.summarize.__name__ == "summarize"


def test_cli_rejects_stateless_variant(tmp_path):
    with pytest.raises(SystemExit, match="dreaming/hypa"):
        ingest.main(["--haystack", "nope", "--variants", "vanilla",
                     "--out-dir", str(tmp_path)])


@pytest.mark.skipif(not pathlib.Path(H.DEFAULT_PRESET).exists(),
                    reason="실물 프리셋 없음")
def test_cli_wiring_on_complete_snapshots(tmp_path, capsys):
    """CLI 배선(load → decode_risup → 가드)을 스킵 경로로만 밟는다 — LLM 0콜.

    인제스천 본체는 실LLM이라 여기서 못 돌린다. 스냅샷을 미리 완성해 두면
    가드가 먼저 걸려 경로·인자 오타만 검출된다.
    """
    card = tmp_path / "card.json"
    card.write_text(json.dumps(CARD))
    hs = dataclasses.replace(_hs(), preset_path=H.DEFAULT_PRESET,
                             card_path=str(card))
    H.save(hs, str(tmp_path / "fake-hs.json"))
    root = tmp_path / "fake-hs"
    for t in (2, 4):
        d = root / "dreaming" / "snap" / f"t{t}"
        d.mkdir(parents=True)
        # 내용을 넣는다 — 빈 디렉터리는 이제 "완주"로 안 센다(`_snapped_dirs`).
        (d / "state.json").write_text("{}")
        (root / "hypa" / "snap").mkdir(parents=True, exist_ok=True)
        (root / "hypa" / "snap" / f"t{t}.json").write_text('{"summaries":[]}')
    # 스킵 경로도 §9-2 꿈 축 메타를 요구한다 (`require_meta`)
    (root / "ingest-meta.json").write_text(json.dumps(
        {"dreaming": _ingest_meta(hs)}), encoding="utf-8")

    ingest.main(["--haystack", "fake-hs", "--out-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert out.count("스냅샷 이미 있음") == 2


# ── MeteredDreamLLM (§8-7 비용 실측 계측판) ──────────────────────────────

class _FakeResp:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return self._body


def _usage_body(text="꿈", cost=0.001, pin=100, cout=20):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"cost": cost, "prompt_tokens": pin,
                      "completion_tokens": cout}}


def _metered(monkeypatch, responses):
    """post를 대본대로 돌려주는 MeteredDreamLLM — 네트워크 0."""
    llm = ingest.MeteredDreamLLM("http://up", "key-not-logged", "m/1")
    monkeypatch.setattr(llm, "BACKOFF_SEC", 0)
    seq = list(responses)

    async def post(_path, json=None):
        r = seq.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm._client, "post", post)
    return llm


def test_metered_llm_collects_usage(monkeypatch):
    """OpenAICompatLLM이 버리는 usage를 수거한다 — 0.0(미측정)이 아니어야 한다."""
    llm = _metered(monkeypatch, [_FakeResp(200, _usage_body(cost=0.002)),
                                 _FakeResp(200, _usage_body(cost=0.003))])
    asyncio.run(llm.complete("s", "u"))
    asyncio.run(llm.complete("s", "u"))
    assert llm.calls == 2
    assert llm.cost == pytest.approx(0.005)
    assert (llm.prompt_tokens, llm.completion_tokens) == (200, 40)


def test_metered_llm_sends_usage_include(monkeypatch):
    """OpenRouter는 usage:{include:true}가 있어야 cost를 실어 보낸다."""
    sent = {}
    llm = ingest.MeteredDreamLLM("http://up", "key-not-logged", "m/1")

    async def post(_path, json=None):
        sent.update(json)
        return _FakeResp(200, _usage_body())

    monkeypatch.setattr(llm._client, "post", post)
    asyncio.run(llm.complete("s", "u"))
    assert sent["usage"] == {"include": True}


def test_metered_llm_retries_5xx_but_not_4xx(monkeypatch):
    """502 한 번이 그 뒤 스냅샷의 기억을 통째로 비우는 걸 막는다."""
    llm = _metered(monkeypatch, [_FakeResp(502), _FakeResp(200, _usage_body())])
    assert asyncio.run(llm.complete("s", "u")) == "꿈"
    assert llm.calls == 1                      # 실패한 시도는 안 센다

    llm2 = _metered(monkeypatch, [_FakeResp(400), _FakeResp(200, _usage_body())])
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(llm2.complete("s", "u"))   # 4xx는 우리 잘못 — 즉시 전파
