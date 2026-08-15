"""고정 건초더미 동결 스키마 (구현 계약 §9 haystack 항목) — LLM 0콜.

핵심 로직은 fake run JSON + fake wire_fn으로 검증한다(프리셋·카드 파일 불필요).
실물 건초더미에 의존하는 검증만 skipif로 감싼다.
"""
import dataclasses
import json
import pathlib

import pytest

from benchmarks.eval.config import ROOT
from benchmarks.eval.windowing import count
from benchmarks.fixed_haystack import haystack as H

_FIXED = "고정부 프리셋 카드 " * 40      # 와이어 앞머리 대역폭 흉내
_CARD = {"name": "소연", "greeting": "어서 와요."}


def _wire(window):
    return [{"role": "system", "content": _FIXED}] + list(window)


def _run(probe_qs=("유저 발화 3", "유저 발화 5"), n_turns=6):
    """프로브 2개(T3 recall, T5 false)를 가진 6턴 fake 런."""
    turns = [{"turn": i, "user": f"유저 발화 {i}", "reply": f"응답 본문 {i}",
              "prompt": 0} for i in range(n_turns)]
    turns[3]["user"], turns[5]["user"] = probe_qs
    ledger = [{"fid": "led3", "kind": "exact", "value": "250골드",
               "text": "잔액은 250골드", "turn": 1, "probed": True},
              {"fid": "led5", "kind": "exact", "value": "은장도",
               "text": "품에 은장도", "turn": 0, "probed": True}]
    probes = [{"turn": 3, "ptype": "recall", "fact": "잔액은 250골드",
               "value": "250골드", "wrong": "", "question": probe_qs[0],
               "distance_turns": 2},
              # fact/value가 원장과 안 맞는 쪽 — distance_turns 역산 경로
              {"turn": 5, "ptype": "false", "fact": "원장에 없는 사실",
               "value": "은장도", "wrong": "목검", "question": probe_qs[1],
               "distance_turns": 4}]
    return {"variant": "dreaming", "session": "fake", "run": 0,
            "model": "deepseek/deepseek-v4-flash-0731",
            "turns": turns, "probes": probes, "ledger": ledger}


def _with_t0_prompt(run):
    """T0 와이어 실측치를 기록값으로 심는다 — 정합성 검사가 통과하도록.

    이미 채워져 있으면 그대로 둔다 — 불일치 케이스를 테스트가 직접 심는다.
    """
    if run["turns"][0].get("prompt"):
        return run
    turns = [H.FrozenTurn(t["turn"], t["user"], t["reply"], False)
             for t in run["turns"]]
    msgs = _wire(H._messages_upto(turns, 0, _CARD))
    run["turns"][0]["prompt"] = sum(count(m["content"]) for m in msgs)
    return run


def _freeze(tmp_path, run=None, **kw):
    run = _with_t0_prompt(run or _run())
    p = tmp_path / "run.json"
    p.write_text(json.dumps(run))
    card = tmp_path / "card.json"
    card.write_text(json.dumps(_CARD))
    return H.freeze(str(p), "fake-hs", card_path=str(card),
                    out_dir=str(tmp_path), wire_fn=_wire, **kw)


def test_freeze_roundtrip(tmp_path):
    hs = _freeze(tmp_path)
    loaded = H.load(str(tmp_path / "fake-hs.json"))
    assert loaded == hs
    assert loaded.narrator_model == "deepseek/deepseek-v4-flash-0731"
    assert loaded.origin_variant == "dreaming"
    assert loaded.run_params == H.RUN_PARAMS
    assert [t.turn for t in loaded.turns if t.is_probe] == [3, 5]


def test_messages_upto_excludes_probe_reply(tmp_path):
    hs = _freeze(tmp_path)
    msgs = H.messages_upto(hs, 3, _CARD)
    # greeting이 맨 앞 assistant (run2.py:447-448), 마지막은 프로브 user
    assert msgs[0] == {"role": "assistant", "content": _CARD["greeting"]}
    assert msgs[-1] == {"role": "user", "content": hs.turns[3].user}
    # 프로브 턴의 원문 응답은 어떤 경로로도 들어가면 안 된다 (계약 §8-1)
    assert all(m["content"] != hs.turns[3].assistant for m in msgs)
    assert len(msgs) == 1 + 3 * 2 + 1


def test_messages_upto_without_greeting(tmp_path):
    hs = _freeze(tmp_path)
    msgs = H.messages_upto(hs, 0, {"name": "소연"})
    assert msgs == [{"role": "user", "content": hs.turns[0].user}]


def test_messages_upto_rejects_out_of_range(tmp_path):
    hs = _freeze(tmp_path)
    with pytest.raises(IndexError):
        H.messages_upto(hs, len(hs.turns), _CARD)


def test_ledger_match_and_distance_fallback(tmp_path):
    hs = _freeze(tmp_path)
    by_turn = {p.turn: p for p in hs.probes}
    # (value, text) 쌍이 원장에 있으면 fid·fact_turn을 원장에서 가져온다
    assert (by_turn[3].fid, by_turn[3].fact_turn) == ("led3", 1)
    # 못 찾으면 fid는 비우고 turn - distance_turns로 역산
    assert (by_turn[5].fid, by_turn[5].fact_turn) == ("", 1)
    assert hs.validation["ledger_matched"] == 1
    assert hs.validation["distance_median"] == 3      # dist 2, 4


def test_ledger_collision_resolved_by_distance(tmp_path):
    """(value, text) 쌍은 원장에서 유일하지 않다 (fix-van-r0에 2쌍 충돌).

    마지막 행으로 덮어쓰면 fact_turn이 조용히 뒤쪽 행으로 잡혀 거리·창내
    판정·미스 분해까지 전부 어긋난다. distance_turns와 대조해 고른다.
    """
    run = _with_t0_prompt(_run())
    # 같은 (value, text)가 turn 1과 turn 4에 두 번 — 프로브는 turn 1을 가리킨다
    run["ledger"].append({"fid": "led3b", "kind": "exact", "value": "250골드",
                          "text": "잔액은 250골드", "turn": 4, "probed": True})
    hs = _freeze(tmp_path, run)
    p3 = next(p for p in hs.probes if p.turn == 3)
    assert (p3.fid, p3.fact_turn) == ("led3", 1)      # 뒤쪽 turn 4가 아니다
    assert hs.validation["ledger_ambiguous"] == 1


def test_wire_gate_covers_every_recorded_turn_for_full_history_origin(tmp_path):
    """정합성 게이트가 T0만 보면 오버플로가 내려지는 후반 턴을 못 지킨다."""
    run = _with_t0_prompt(_run())
    run["variant"] = "vanilla"
    turns = [H.FrozenTurn(t["turn"], t["user"], t["reply"], False)
             for t in run["turns"]]
    for t in run["turns"][1:]:                        # 전 턴에 기록을 심는다
        t["prompt"] = sum(count(m["content"])
                          for m in _wire(H._messages_upto(turns, t["turn"],
                                                          _CARD)))
    hs = _freeze(tmp_path, run)
    assert hs.validation["wire_gated_turns"] == len(run["turns"])
    # 후반 턴 하나만 어긋나도 잡힌다 (T0은 그대로 통과하는데도)
    run["turns"][5]["prompt"] = int(run["turns"][5]["prompt"] * 0.5)
    with pytest.raises(SystemExit, match="T5"):
        _freeze(tmp_path, run)


def test_leaky_probe_recorded_but_not_dropped(tmp_path):
    hs = _freeze(tmp_path, _run(probe_qs=("그때 250골드 맞았죠?", "유저 발화 5")))
    assert hs.validation["leaky_probes"] == [3]
    # 조용히 버리지 않는다 — 프로브는 남고 run.py가 excluded로 스킵한다
    assert [p.turn for p in hs.probes] == [3, 5]


def test_young_probe_warned(tmp_path):
    hs = _freeze(tmp_path)
    # dist 2·4 < MIN_PROBE_AGE(15) — 경고로만 남기고 채택은 계속
    assert hs.validation["distance_young"] == [3, 5]
    assert hs.validation["distance_min"] == 2


def test_wire_mismatch_raises(tmp_path):
    run = _with_t0_prompt(_run())
    run["turns"][0]["prompt"] = int(run["turns"][0]["prompt"] * 1.3)
    with pytest.raises(SystemExit, match="와이어 정합성 실패"):
        _freeze(tmp_path, run)


def test_overflow_turns_flagged_at_limit(tmp_path):
    """오버플로 판정은 **프로바이더 공간** 추정치로 한다.

    o200k 값을 창 상한과 직접 비교하면 한국어 히스토리가 최대 12%
    과소계상돼 실제로 넘는 턴이 가드를 통과한다(실물 A의 T89).
    """
    run = _with_t0_prompt(_run())
    turns = [H.FrozenTurn(t["turn"], t["user"], t["reply"], False)
             for t in run["turns"]]
    probes = H._build_probes(run["probes"], run["ledger"])
    t5 = sum(count(m["content"])
             for m in _wire(H._messages_upto(turns, 5, _CARD)))
    prompts = {t["turn"]: t["prompt"] for t in run["turns"]}
    est5 = H.validate(turns, probes, run["ledger"], _CARD, _wire,
                      prompts)["probe_provider_tokens"]["5"]
    assert est5 > t5                                  # 보정은 항상 부풀린다
    v = H.validate(turns, probes, run["ledger"], _CARD, _wire, prompts,
                   provider_ctx_limit=est5 - 1)
    assert v["overflow_turns"] == [5]
    assert v["probe_wire_tokens"]["5"] == t5          # o200k는 그대로 남긴다
    # 같은 상한을 o200k로 재면 통과해 버린다 — 그게 이 보정의 존재 이유다
    assert t5 <= est5 - 1


def test_unprotectable_probe_recorded_but_not_dropped(tmp_path):
    """값이 1글자면 마스킹·누출 게이트로 보호할 수 없다 (lucid.py:84-93)."""
    run = _run()
    run["probes"][0]["value"] = "렌"[0]              # 한글 1글자
    run["ledger"][0]["value"] = "렌"[0]
    hs = _freeze(tmp_path, run)
    assert hs.validation["unprotectable_probes"] == [3]
    assert [p.turn for p in hs.probes] == [3, 5]     # 조용히 버리지 않는다


# ── provenance (스펙 §9-2) ─────────────────────────────────────────────────

def test_freeze_records_provenance(tmp_path):
    """"어떤 조건에서 생성됐는지 기록이 전무"한 상태를 막는 게 §9-2다."""
    hs = _freeze(tmp_path)
    pv = hs.provenance
    assert not [k for k in H.REQUIRED_PROVENANCE if k not in pv]
    assert pv["source_session"] == "fake"          # 원 런 JSON의 session
    assert pv["source_run"].endswith("run.json")
    # 해시는 실제 파일 내용에서 나온다 (상수 박제가 아니다)
    assert pv["source_run_sha256"] == H._sha256(str(tmp_path / "run.json"))
    assert pv["card_sha256"] == H._sha256(str(tmp_path / "card.json"))
    assert pv["frozen_at"].endswith("Z") and pv["frozen_at"][:2] == "20"
    # 생성 설정은 한 블록에 모인다 — 최상위 필드와 값이 갈리면 안 된다
    assert (pv["narrator_model"], pv["run_params"]) == (hs.narrator_model,
                                                        hs.run_params)
    # §9-4 준수 선언 + 게이트는 복제가 아니라 포인터
    assert pv["source_store_reused"] is False
    assert getattr(hs, pv["gates"]) is hs.validation


def test_freeze_survives_missing_git(tmp_path, monkeypatch):
    """git이 없거나 실패해도 freeze는 죽지 않는다 — 기록만 미상으로 남는다."""
    def _boom(*a, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(H.subprocess, "run", _boom)
    pv = _freeze(tmp_path).provenance
    # "" 는 False("깨끗했다")가 아니라 "모른다" — 모르는 걸 단언하지 않는다
    assert (pv["git_commit"], pv["git_dirty"]) == ("", "")


def test_git_state_reports_head_of_this_repo():
    commit, dirty = H.git_state()
    assert commit == "" or (len(commit) == 40 and commit.isalnum())
    assert dirty == "" or isinstance(dirty, bool)


def test_load_rejects_haystack_without_provenance(tmp_path):
    """§9-2: 메타데이터 없는 haystack은 로더가 거부한다 (조용한 폴백 금지)."""
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    del doc["provenance"]
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as e:
        H.load(str(path))
    # 무엇이 없는지 + 어떻게 되살리는지까지 말해 준다
    assert "source_session" in str(e.value) and "freeze" in str(e.value)
    assert "--id fake-hs" in str(e.value)


def test_load_rejects_partial_provenance(tmp_path):
    """일부만 있는 것도 거부 — 빠진 키를 기본값으로 메우지 않는다."""
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    dropped = H.REQUIRED_PROVENANCE[0]
    doc["provenance"].pop(dropped)
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match=dropped):
        H.load(str(path))


def test_load_rejects_provenance_whose_values_are_all_blank(tmp_path):
    """키만 있고 값이 전부 ""인 provenance는 기록이 아니다 — 거부한다.

    이론적 케이스가 아니라 **프로덕션 freeze가 만들 수 있는 산물**이다:
    `git_state`는 실패 시 `("", "")`, `_sha256`는 OSError에 ""를 돌려주도록
    설계돼 있어, git 없는 머신에서 경로가 깨진 채 얼리면 정확히 이 모양이
    나온다. 키 존재만 보면 그 파일이 초록불을 받고, §9-2가 막으려던 "어떤
    조건에서 생성됐는지 기록이 전무한 건초더미"가 형식만 갖춘 채 들어온다.
    """
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    doc["provenance"] = {k: "" for k in H.REQUIRED_PROVENANCE}
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as e:
        H.load(str(path))
    assert "빈 값 source_run_sha256" in str(e.value)


def test_load_tolerates_blank_only_where_unknown_is_legitimate(tmp_path):
    """반대편 — git·해시가 미상인 정직한 freeze 산물은 통과해야 한다.

    ""를 전부 막으면 "git 없는 머신에서 얼린 파일"이 통째로 못 쓰게 되고,
    그러면 freeze가 미상을 False로 메우는 유혹이 생긴다(§10.4가 경고한 방향).
    """
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    for k in H.PROVENANCE_MAY_BE_UNKNOWN:
        doc["provenance"][k] = ""
    path.write_text(json.dumps(doc))
    assert H.load(str(path)).provenance["git_commit"] == ""


@pytest.mark.parametrize("prov", [["source_session"], "unknown", 7])
def test_load_rejects_non_dict_provenance_with_a_refreeze_hint(tmp_path, prov):
    """dict가 아닌 provenance도 **거부**로 끝나야 한다 — 스택트레이스 말고.

    `(prov or {}).get(...)`은 truthy 비-dict에서 AttributeError로 죽어 재freeze
    안내가 사라진다.
    """
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    doc["provenance"] = prov
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="freeze"):
        H.load(str(path))


def test_load_rejects_hand_edited_source_store_reused(tmp_path):
    """§9-4 준수 선언은 값 검증이 있어야 선언 구실을 한다.

    freeze는 항상 False를 적고 인제스천은 새로 리플레이한다 — True인 파일은
    손편집뿐이고, 그건 병리 세션 스토어를 썼다는 자기 신고다.
    """
    _freeze(tmp_path)
    path = tmp_path / "fake-hs.json"
    doc = json.loads(path.read_text())
    doc["provenance"]["source_store_reused"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="§9-4"):
        H.load(str(path))


def test_load_accepts_freshly_frozen_haystack(tmp_path):
    hs = _freeze(tmp_path)
    assert H.load(str(tmp_path / "fake-hs.json")).provenance == hs.provenance


def test_run_params_records_whether_the_operator_said_so(tmp_path):
    """run_params는 측정값이 아니라 가정이다 — 그 사실을 provenance가 진다.

    원 런 JSON의 top-level엔 max_context/max_tokens/reasoning이 **없어서**
    freeze엔 대조할 진실이 없다. 조작자가 플래그를 빠뜨리면 모듈 상수가
    사실처럼 박히므로, 값만 적고 출처를 안 적으면 §10.4가 경고한 "기록이 아니라
    추측"이 된다.
    """
    assert _freeze(tmp_path).provenance["run_params_source"] == (
        "assumed-default")
    hs = _freeze(tmp_path, run_params={**H.RUN_PARAMS, "max_context": 45000},
                 run_params_source="cli")
    assert hs.provenance["run_params_source"] == "cli"
    assert hs.run_params["max_context"] == 45000


def test_cli_labels_omitted_run_params_as_assumed(monkeypatch):
    """CLI가 "명시했는가"를 실제로 구분하는지 — 배선만 본다(디스크 무접촉)."""
    seen = {}

    def _spy(run_json, haystack_id, *a, **kw):
        seen.update(kw)
        raise SystemExit(0)                    # 저장까지 갈 필요 없다

    monkeypatch.setattr(H, "freeze", _spy)
    for argv, want in ((["freeze", "r.json", "--id", "x"], "assumed-default"),
                       (["freeze", "r.json", "--id", "x",
                         "--max-context", "45000"], "cli")):
        with pytest.raises(SystemExit):
            H.main(argv)
        assert seen["run_params_source"] == want
    assert seen["run_params"]["max_context"] == 45000
    # 명시하지 않은 나머지는 상수로 채워지되, 그 사실이 출처에 남는다
    assert seen["run_params"]["max_tokens"] == H.RUN_PARAMS["max_tokens"]


# ── 측정 시점 입력 대조 (§9-2) ─────────────────────────────────────────────

def test_verify_inputs_catches_a_changed_card(tmp_path):
    """preset·card는 동결 파일에 안 실리고 매 런 디스크에서 다시 읽힌다.

    지문을 freeze 때 한 번 찍고 아무도 다시 안 재면, A는 어제 프리셋 B는 오늘
    프리셋으로 돌아도 결과 JSON만 보고는 알 수 없다 — A/B 교차 검증의 전제가
    측정 시점엔 보증이 아니게 된다.
    """
    hs = _freeze(tmp_path)
    H.verify_inputs(hs)                       # 그대로면 통과
    (tmp_path / "card.json").write_text(json.dumps({"name": "다른카드"}))
    with pytest.raises(SystemExit, match="카드"):
        H.verify_inputs(hs)


def test_verify_inputs_skips_unmeasured_fingerprints(tmp_path):
    """지문이 ""(미측정)면 대조를 건너뛴다 — 없는 지문으로 판정하지 않는다."""
    hs = _freeze(tmp_path)
    blank = {**hs.provenance, "preset_sha256": "", "card_sha256": ""}
    H.verify_inputs(dataclasses.replace(hs, provenance=blank))


# ── 이식성·네임스페이스 (§9-3) ─────────────────────────────────────────────

def test_frozen_paths_are_repo_relative_when_inside_root(tmp_path):
    """워크트리 절대 경로를 박으면 머지 후 워크트리를 지운 순간 실행 불가다."""
    assert H.repo_path(str(ROOT / "dreaming_data" / "eval" / "c.json")) == (
        "dreaming_data/eval/c.json")
    assert H.repo_path("/tmp/outside.json") == "/tmp/outside.json"
    # 왕복이 성립해야 소비자가 상대 경로를 다시 열 수 있다
    assert H.resolve_path("dreaming_data/eval/c.json") == str(
        ROOT / "dreaming_data" / "eval" / "c.json")


def test_require_namespace_refuses_a_session_directory():
    """§9-3: `dreaming_data/` 안에서 `fixed_haystack/` 밖은 세션 디렉터리다.

    `--out-dir <…>/dreaming_data --id fix-drm-r0`이면 root가 원 세션 스토어가
    되고 거기에 `dreaming/live`를 쓴다 — `--force`면 그 안에서 rmtree까지 돈다.
    """
    with pytest.raises(SystemExit, match="§9-3"):
        H.require_namespace(str(ROOT / "dreaming_data"), "출력")
    with pytest.raises(SystemExit, match="§9-3"):
        H.require_namespace(str(ROOT / "dreaming_data" / "fix-drm-r0"), "출력")
    # 네임스페이스 안과 데이터 루트 바깥(tmp·스크래치)은 통과
    H.require_namespace(H.DEFAULT_OUT_DIR, "출력")
    H.require_namespace(str(ROOT / "dreaming_data" / "fixed_haystack" / "x"),
                        "출력")
    H.require_namespace("/tmp/scratch-run", "출력")


def test_freeze_refuses_to_write_outside_the_namespace(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_with_t0_prompt(_run())))
    with pytest.raises(SystemExit, match="§9-3"):
        H.freeze(str(run), "fake-hs", wire_fn=_wire,
                 out_dir=str(ROOT / "dreaming_data" / "fix-drm-r0"))


# ── 실물 건초더미 (없으면 스킵) ────────────────────────────────────────────

_REAL_RUN = ROOT / "dreaming_data" / "eval" / "v2-fix-drm-r0-run0.json"
_REAL_CARD = pathlib.Path(H.DEFAULT_CARD)
_REAL_PRESET = pathlib.Path(H.DEFAULT_PRESET)


@pytest.mark.skipif(not (_REAL_RUN.exists() and _REAL_CARD.exists()
                         and _REAL_PRESET.exists()),
                    reason="실물 런/카드/프리셋 없음")
def test_freeze_real_drm_haystack(tmp_path):
    hs = H.freeze(str(_REAL_RUN), "fix-drm-r0", out_dir=str(tmp_path))
    assert len(hs.turns) == 100 and len(hs.probes) == 9
    assert hs.origin_variant == "dreaming"
    assert hs.narrator_model == "deepseek/deepseek-v4-flash-0731"
    assert hs.validation["ledger_matched"] == 9
    assert abs(hs.validation["wire_delta_pct"]) <= 5
    # 프로바이더 공간 추정으로는 **T89도** 창을 넘는다 (o200k 122,268 →
    # 추정 137,527). 계약 §0의 "T89는 125K로 아슬아슬"은 o200k 값이라
    # 단위가 달랐다 — 그대로 뒀으면 T89이 라이브로 나가 4xx로 런을 죽인다.
    assert hs.validation["overflow_turns"] == [89, 99]
    assert hs.validation["probe_wire_tokens"]["89"] < 128000
    assert hs.validation["probe_provider_tokens"]["89"] > 128000
    # 양 건초더미 T29(false)의 값이 한글 1글자 — 채점 불가라 제외 표시
    assert hs.validation["unprotectable_probes"] == [29]
    # §9-2: 실물 프리셋·카드도 지문이 남는다 (fake는 "" = 미측정이었다)
    pv = hs.provenance
    assert pv["source_session"] == "fix-drm-r0"
    assert len(pv["preset_sha256"]) == 64 and len(pv["card_sha256"]) == 64
    card = json.loads(_REAL_CARD.read_text())
    msgs = H.messages_upto(hs, 99, card)
    assert msgs[-1]["content"] == hs.turns[99].user
    assert all(m["content"] != hs.turns[99].assistant for m in msgs)
