"""동결 스키마 + freeze CLI + 채택 검증 (구현 계약 §2 — 스키마 소유자).

원 런 JSON(`benchmarks/eval/run2.py` 산출물)의 (user, assistant) 쌍을 통째로
얼려 4+1변형이 **같은 텍스트**를 서로 다른 방식으로만 조립하게 만든다.
프로브 턴에서만 라이브 재생성하므로, 재생성된 응답이 히스토리로 새어 들어가면
"고정 건초더미"가 무의미해진다 — `messages_upto`가 haystack만 읽는 순수
함수인 것이 그 불변식의 구조적 보증이다(계약 §8-1).

`benchmarks/eval/**`는 읽기 전용 — 전부 import 재사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval.config import ROOT
from benchmarks.eval.lucid import MIN_PROBE_AGE, DirFact, probe_leaks_value
from benchmarks.eval.preset2wire import decode_risup
from benchmarks.eval.run2 import build_wire
from benchmarks.eval.windowing import count

# 계약 §0 표의 실물 경로 — 이 조합으로 원 런이 돌았음이 T0 와이어 재조립
# (12,427tok vs 기록 12,725tok, −2.3%)으로 확인됐다. scripts/night_run.sh:11-12.
DEFAULT_PRESET = ("/Users/yanghyeon-u/Downloads/뮈토스6.2/"
                  "🏺뮈토스 프롬프트 V6.2/"
                  "🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup")
DEFAULT_CARD = str(ROOT / "dreaming_data" / "eval" / "card-soyeon-v2.json")
# 스펙 §9-3: 벤치 산출물은 `fixed_haystack/` **전용 네임스페이스**에만 쓴다.
# DATA_ROOT는 세션 16개가 코드 버전·설정 제각각으로 섞여 있는 오염 지대라,
# 그 아래 fixed_haystack 밖으로 쓰는 경로는 `require_namespace`가 막는다.
DATA_ROOT = ROOT / "dreaming_data"
BENCH_NAMESPACE = DATA_ROOT / "fixed_haystack"
DEFAULT_OUT_DIR = str(BENCH_NAMESPACE)
# 원 런이 실제로 쓴 값 — config.MAX_CONTEXT(45000)/MAX_TOKENS(4000) 기본값이
# 아니다. night_run.sh:16,21,24 + fix-run.log "100턴 @100000" 실측(계약 §0).
RUN_PARAMS = {"max_context": 100000, "max_tokens": 30000, "reasoning": "4000"}
PROVIDER_CTX_LIMIT = 128000    # DeepSeek 창 — 오버플로 판정선(§2.2-4)
WIRE_TOLERANCE = 0.05          # 재조립 허용 오차(§2.2-3). 넘으면 카드/프리셋 불일치
# lucid.eligible/probe_plan이 값 길이 2 미만 사실을 출제 풀에서 빼는 것과 같은
# 문턱(lucid.py:84-93, 199-209). 마스킹 못 하는 값은 누출 게이트로도 보호할 수
# 없고, 오라클이 부분 문자열 포함 검사라 1글자면 문자 기저율이 판정을 지배한다.
MIN_VALUE_LEN = 2

# ── 프로바이더 토크나이저 보정 ────────────────────────────────────────────
# `windowing.count`는 o200k인데 창 상한 128K는 **프로바이더(DeepSeek)
# 토크나이저 공간**의 숫자다. 그대로 비교하면 최대 12% 과소계상된다 —
# v2-fix-van-r0 T99 실측: o200k 107,094 vs 기록 prompt 120,011 (비율 1.121).
# 비율이 턴에 따라 1.100→1.121로 움직이는 이유는 고정부(영문 프리셋)와
# 한국어 히스토리의 팽창률이 달라서다 → 두 항으로 나눠 잰다.
# 계수는 그 런의 기록 prompt 10점(T0 + 프로브 9턴) 최소자승 적합, 최대 오차
# 0.30%. **프리셋/언어/프로바이더가 바뀌면 재측정 대상** — freeze가 매번
# 기록 prompt와 대조해 이탈을 validation에 박제하고 허용치를 넘으면 죽는다.
TOK_RATIO_FIXED = 1.020
TOK_RATIO_HISTORY = 1.136

WireFn = Callable[[List[Dict]], List[Dict]]   # window -> 최종 와이어


def provider_tokens(fixed_tokens: int, history_tokens: int) -> int:
    """o200k 실측치 → 프로바이더 토크나이저 공간 추정치.

    오버플로 판정은 반드시 이 값으로 한다. o200k 값을 128000과 직접 비교하면
    단위가 달라, 실제로는 창을 넘는 턴(A의 T89 = o200k 122,268 → 추정
    137,527)이 가드를 통과해 나레이터를 부르고 4xx로 런을 죽인다.
    """
    return round(fixed_tokens * TOK_RATIO_FIXED
                 + history_tokens * TOK_RATIO_HISTORY)


@dataclass(frozen=True)
class FrozenTurn:
    turn: int          # 0-기준
    user: str
    assistant: str     # 원 런의 reply 원문
    is_probe: bool


@dataclass(frozen=True)
class FrozenProbe:
    turn: int          # 0-기준, 이 턴의 user가 프로브 발화
    ptype: str         # recall|relation|false|update|recent
    fid: str           # ledger 대조용 (없으면 "")
    fact: str          # fact.text
    value: str         # fact.value
    wrong: str         # false 프로브 오염값, 그 외 ""
    question: str      # == turns[turn].user
    fact_turn: int     # ledger에서 찾은 원 기록 턴 (distance_turns 계산용)


@dataclass(frozen=True)
class FixedHaystack:
    haystack_id: str
    source_run: str
    origin_variant: str         # "dreaming" | "vanilla" — 출처 편향 라벨
    narrator_model: str         # 프로브 재생성도 이 모델로 (config.MODEL 참조 금지)
    preset_path: str
    card_path: str
    run_params: dict
    turns: List[FrozenTurn]
    probes: List[FrozenProbe]
    ledger: List[dict]
    validation: dict
    # 스펙 §9-2 생성 이력 — 이게 없으면 `load`가 거부한다. 게이트 **결과**는
    # 여기 복제하지 않는다: `gates="validation"` 포인터로 위 필드를 가리킨다
    # (두 사본을 두면 재freeze 때 갈린다). `_provenance` 참고.
    provenance: dict


# ── 히스토리 재구성 (계약 §2.1) ────────────────────────────────────────────

def _messages_upto(turns: List[FrozenTurn], probe_turn: int,
                   card: Dict) -> List[Dict]:
    if not 0 <= probe_turn < len(turns):
        raise IndexError(
            f"probe_turn {probe_turn}가 턴 범위(0~{len(turns) - 1}) 밖")
    msgs: List[Dict] = []
    if card.get("greeting"):
        # run2.py:447-448과 동일 — greeting은 history[0]의 assistant다.
        msgs.append({"role": "assistant", "content": card["greeting"]})
    for t in turns[:probe_turn]:
        msgs.append({"role": "user", "content": t.user})
        msgs.append({"role": "assistant", "content": t.assistant})
    msgs.append({"role": "user", "content": turns[probe_turn].user})
    return msgs


def messages_upto(hs: FixedHaystack, probe_turn: int,
                  card: Dict) -> List[Dict]:
    """greeting + turns[0..probe_turn-1]의 (user, assistant) 쌍
    + turns[probe_turn].user

    프로브 응답은 절대 포함되지 않는다 — 이 함수가 haystack만 읽는 순수 함수라
    스펙 §3.2("프로브 응답은 채점에만 쓰고 history에 반영 금지")가 구조적으로
    보장된다.
    """
    return _messages_upto(hs.turns, probe_turn, card)


# ── 채택 검증 (계약 §2.2) ──────────────────────────────────────────────────

def _ledger_index(ledger: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """(value, text) → 행 **목록**. 키는 유일하지 않다.

    실측: fix-van-r0 원장 343행 중 2쌍이 충돌한다(turn 24↔90, 93↔98).
    dict 컴프리헨션으로 덮어쓰면 뒤쪽 행이 이기고 fact_turn이 조용히 최대
    66턴 어긋난다 — 그 값이 distance_turns·_in_window·decompose_miss까지
    그대로 전파돼 창밖 실패가 창내 실패로 뒤집힌다.
    """
    index: Dict[Tuple[str, str], List[Dict]] = {}
    for r in ledger:
        index.setdefault((r["value"], r["text"]), []).append(r)
    return index


def _build_probes(raw_probes: List[Dict],
                  ledger: List[Dict]) -> List[FrozenProbe]:
    """§2.2-5 ledger 대조 — (value, text) 쌍으로 원 사실을 찾아 fid·fact_turn을 채운다.

    원 런 JSON의 `probes[].distance_turns`는 원장과 **독립적인 진실**이다.
    §2.2-5가 요구하는 "대조"는 단방향 조회가 아니라 이 두 출처를 맞춰 보는
    것이므로, 기대 턴(`turn - distance_turns`)과 일치하는 행만 채택한다.
    후보가 갈리면 fid를 비우고 역산값을 쓴다 — 조용한 오답보다 무기록이 낫다.
    """
    index = _ledger_index(ledger)
    out: List[FrozenProbe] = []
    for p in raw_probes:
        rows = index.get((p["value"], p["fact"]), [])
        back = (p["turn"] - p["distance_turns"]
                if "distance_turns" in p else None)
        hits = rows if back is None else [r for r in rows if r["turn"] == back]
        row = hits[0] if len(hits) == 1 else None
        out.append(FrozenProbe(
            turn=p["turn"], ptype=p["ptype"],
            fid=row["fid"] if row else "", fact=p["fact"],
            value=p["value"], wrong=p.get("wrong", ""), question=p["question"],
            fact_turn=(row["turn"] if row
                       else p["turn"] - p.get("distance_turns", 0))))
    return out


def _history_tokens(turns: List[FrozenTurn], card: Dict) -> List[int]:
    """turn i의 히스토리(greeting + 0..i-1쌍 + i의 user) 토큰 — 누적합.

    `count`가 메시지별 합이라 프리픽스 합이 `_messages_upto` 직접 계산과
    정확히 같다(실측 9프로브 diff 0). 매 턴 재계산은 O(n²) tiktoken 호출이라
    100턴 전체 보정 대조를 돌릴 수 없다.
    """
    g = count(card["greeting"]) if card.get("greeting") else 0
    out: List[int] = []
    acc = g
    for t in turns:
        u = count(t.user)
        out.append(acc + u)
        acc += u + count(t.assistant)
    return out


def validate(turns: List[FrozenTurn], probes: List[FrozenProbe],
             ledger: List[Dict], card: Dict, wire_fn: WireFn,
             recorded_prompts: Dict[int, int], origin_variant: str = "",
             provider_ctx_limit: int = PROVIDER_CTX_LIMIT) -> Dict:
    """계약 §2.2의 5개 검증. 3번(와이어 정합성)만 실패 시 SystemExit.

    나머지는 조용히 버리지 않고 전부 반환 dict에 박제한다 — leaky·값 길이
    미달 프로브도 FrozenProbe로는 남기고 run.py가 스킵 라벨로 뺀다(§2.2-1).

    wire_fn은 조립 심(seam)이다: freeze는 `build_wire(preset, card, ...)`를
    넣고, 테스트는 프리셋·카드 파일 없이 fake를 넣어 LLM·실물 0의존으로 돈다.
    `recorded_prompts`는 {turn: 기록 prompt_tokens} — 0/누락은 미기록으로 본다.
    """
    # 1. 프로브 누출 소급 검증 (§8-10) — 발화가 정답을 그대로 담았는지
    leaky = [p.turn for p in probes
             if probe_leaks_value(p.question,
                                  DirFact(fid=p.fid, kind="", value=p.value,
                                          text=p.fact, turn=p.fact_turn))]
    # 1b. 값 길이 게이트 — lucid가 출제 단계에서 적용하는 문턱을 freeze가
    #     소급 재적용한다. 누출 게이트만 소급하고 이건 빠뜨리면 채점 불가
    #     프로브가 그대로 실린다. 실측: 양 건초더미 T29(false)의 값이 한글
    #     1글자라 얼린 원문 92/100·82/100턴에 이미 등장한다 →
    #     `value_in_window`가 항상 True이고 오라클도 문자 기저율에 지배된다.
    unprotectable = [p.turn for p in probes if len(p.value) < MIN_VALUE_LEN]
    # 2. 거리 분포 (G2 대응)
    dists = [p.turn - p.fact_turn for p in probes]
    # 3. 와이어 정합성 (§8-9) — **보정 추정치** vs 기록 prompt_tokens.
    #    T0만 보던 것을 넓혔다: T0는 히스토리 비중이 최소라 두 토크나이저의
    #    격차가 최소인 지점(-2.34%)이고, 오버플로 판정이 실제로 내려지는
    #    후반 턴(-10.8%)은 아무것도 보증하지 못했다. 풀 히스토리를 그대로
    #    보낸 출처(vanilla)면 기록이 있는 모든 턴을 잰다 — dreaming 출처는
    #    프록시 압축 때문에 기록이 raw보다 정당하게 작아 T0만 문이다.
    fixed_tokens = sum(count(m["content"]) for m in wire_fn([]))
    hist = _history_tokens(turns, card)
    est = [provider_tokens(fixed_tokens, h) for h in hist]
    checked = sorted(t for t, n in recorded_prompts.items()
                     if n > 0 and 0 <= t < len(turns))
    gated = (checked if origin_variant == "vanilla"
             else [t for t in checked if t == 0])
    dev = {t: (est[t] - recorded_prompts[t]) / recorded_prompts[t]
           for t in checked}
    worst = max(gated, key=lambda t: abs(dev[t])) if gated else None
    # 4. 오버플로 지점 — 프로바이더 공간에서 판정한다(o200k 그대로 비교 금지)
    probe_tokens = {str(p.turn): fixed_tokens + hist[p.turn] for p in probes}
    probe_provider = {str(p.turn): est[p.turn] for p in probes}
    out = {
        "leaky_probes": leaky,
        "unprotectable_probes": unprotectable,
        "min_value_len": MIN_VALUE_LEN,
        "distance_median": statistics.median(dists) if dists else 0,
        "distance_min": min(dists) if dists else 0,
        "distance_young": [p.turn for p, d in zip(probes, dists)
                           if d < MIN_PROBE_AGE],
        "min_probe_age": MIN_PROBE_AGE,
        "wire_t0_rebuilt": fixed_tokens + hist[0] if turns else 0,
        "wire_t0_estimated": est[0] if turns else 0,
        "wire_t0_recorded": recorded_prompts.get(0, 0),
        "wire_delta_pct": round(dev.get(0, 0.0) * 100, 2),
        "wire_checked_turns": len(checked),
        "wire_gated_turns": len(gated),
        "wire_worst_turn": worst,
        "wire_worst_delta_pct": (round(dev[worst] * 100, 2)
                                 if worst is not None else 0.0),
        "tokenizer_ratio": {"fixed": TOK_RATIO_FIXED,
                            "history": TOK_RATIO_HISTORY},
        "fixed_tokens": fixed_tokens,
        "provider_ctx_limit": provider_ctx_limit,
        "probe_wire_tokens": probe_tokens,           # o200k (트림 예산과 같은 자)
        "probe_provider_tokens": probe_provider,     # 보정 추정 (창 판정용)
        "overflow_turns": [int(t) for t, n in probe_provider.items()
                           if n > provider_ctx_limit],
        # 5. ledger 대조 결과
        "ledger_size": len(ledger),
        "ledger_matched": sum(1 for p in probes if p.fid),
        "ledger_ambiguous": sum(1 for rows in _ledger_index(ledger).values()
                                if len(rows) > 1),
    }
    if worst is not None and abs(dev[worst]) > WIRE_TOLERANCE:
        raise SystemExit(
            f"와이어 정합성 실패: T{worst} 추정 {est[worst]}tok vs 기록 "
            f"{recorded_prompts[worst]}tok ({out['wire_worst_delta_pct']}%, "
            f"허용 ±{WIRE_TOLERANCE * 100}%) — 카드/프리셋이 원 런과 다르거나 "
            f"토크나이저 보정({TOK_RATIO_FIXED}/{TOK_RATIO_HISTORY})이 낡았다")
    return out


# ── provenance (스펙 §9-2) ─────────────────────────────────────────────────
# `dreaming_data/`에는 코드 버전·설정·카드가 제각각인 세션이 섞여 있고 어떤
# 조건에서 나왔는지 기록이 없다. 동결 파일이 그 기록을 들고 다니지 않으면
# 오염이 조용히 측정에 들어간다 — 그래서 아래 키 전부가 의무이고, 하나라도
# 없으면 `load`가 거부한다.
REQUIRED_PROVENANCE = (
    "source_session",       # 원 런 JSON의 session (예 "fix-drm-r0")
    # 원 런 파일. `_relpath`가 심링크를 따라가므로 ROOT 밖의 실물을 가리키면
    # 절대 경로가 남는다 — 실물 두 건이 그 경우다(dreaming_data/eval의 런
    # JSON이 본 저장소로 향하는 심링크). 어느 쪽이든 sha256이 내용을 못박는다.
    "source_run",
    "source_run_sha256",
    "frozen_at",            # ISO8601 UTC
    "git_commit",           # freeze 시점 HEAD ("" = 미상)
    "git_dirty",            # 워킹트리 오염 (bool, "" = 미상)
    "preset_path", "preset_sha256",
    "card_path", "card_sha256",
    "narrator_model",       # 최상위 필드와 중복 — 생성 설정을 한 곳에 모은다
    "run_params",
    # run_params는 원 런 JSON에서 **읽어낸 값이 아니다** — 그 JSON의 top-level엔
    # max_context/max_tokens/reasoning이 없어 대조할 진실이 아예 없다. 조작자가
    # 플래그를 빠뜨리면 모듈 상수 RUN_PARAMS가 사실처럼 박힌다. 그래서 값이 아니라
    # **출처**를 같이 적는다: "cli"(조작자가 명시) vs "assumed-default"(상수 가정).
    "run_params_source",
    "source_store_reused",  # §9-4 준수 선언 (아래 주석)
    "gates",                # 게이트 결과가 있는 필드 이름
)

# `""`가 정당한 "미상"인 자리 — 그 외 키는 값이 비면 freeze 파이프라인이 만든
# 산물이 아니다(freeze는 이 값들을 못 채울 수 없다). §10.4는 "모르는 값을
# 기본값으로 메우는 순간 메타데이터는 기록이 아니라 추측"이라는 교훈을 freeze
# 쪽에만 적용했는데, 로더가 그 "미상"을 무제한 통과시키면 반쪽이다: git 없는
# 머신에서 프리셋 경로가 깨진 채 얼린 파일은 전 필드가 ""인 채로 초록불을 받고,
# §9-2가 막으려던 "어떤 조건에서 생성됐는지 기록이 전무한 건초더미"가 형식만
# 갖춘 채 측정에 들어간다. 키 **존재**가 아니라 값을 본다.
PROVENANCE_MAY_BE_UNKNOWN = frozenset({
    "git_commit",      # git이 없거나 실패 (`git_state`가 ""를 돌려준다)
    "git_dirty",       # 같음. "" ≠ False (미상 ≠ 깨끗함)
    "preset_sha256",   # 파일을 못 읽음 (`_sha256`가 OSError에 ""를 돌려준다)
    "card_sha256",
    "source_session",  # 원 런 JSON에 session이 없으면 ""가 정확한 기록이다
})


def _sha256(path: str) -> str:
    """파일 해시. 읽을 수 없으면 "" = 미측정 — freeze를 죽이지 않는다.

    프리셋/카드가 실물과 다른 경우는 해시가 아니라 와이어 정합성 게이트
    (`validate` 3번)가 잡는다. 해시는 "그때 그 파일"을 나중에 대조하기 위한
    지문이지 이 자리의 통과 조건이 아니다.
    """
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def git_state(root: Optional[pathlib.Path] = None) -> Tuple[str, object]:
    """(HEAD, 워킹트리 오염). git이 없거나 실패하면 `("", "")`.

    dirty의 `""`는 `False`와 뜻이 다르다 — 모르는 것을 "깨끗했다"고 적으면
    provenance가 거짓말을 한다. freeze는 이 실패로 죽지 않는다(기록이 조금
    빈 동결 파일이, 얼리지 못한 건초더미보다 낫다).
    """
    def _git(*args: str) -> str:
        return subprocess.run(("git", "-C", str(root or ROOT)) + args,
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout

    try:
        return (_git("rev-parse", "HEAD").strip(),
                bool(_git("status", "--porcelain").strip()))
    except (OSError, subprocess.SubprocessError):
        return "", ""


def _provenance(run_json_path: str, run: Dict, preset_path: str,
                card_path: str, narrator_model: str,
                run_params: Dict, run_params_source: str) -> Dict:
    """§9-2 의무 메타데이터를 한 블록에 모은다.

    `source_store_reused=False`는 §9-4("병리 세션의 facts/·episodes/ 재사용
    금지")를 기계 판독 가능하게 박제한 것이다 — dreaming 변형의 인제스천은
    `ingest.py`가 `fixed_haystack/{id}/dreaming/live`에 **새로 리플레이**하고
    원 세션 스토어를 여는 경로가 없다. 이 값이 True인 동결 파일은 없어야 한다.
    """
    commit, dirty = git_state()
    return {
        "source_session": run.get("session", ""),
        "source_run": _relpath(run_json_path),
        "source_run_sha256": _sha256(run_json_path),
        "frozen_at": (datetime.now(timezone.utc)
                      .isoformat(timespec="seconds").replace("+00:00", "Z")),
        "git_commit": commit,
        "git_dirty": dirty,
        # 경로는 이식 가능한 형태로(repo_path), 지문은 실물에서 잰다.
        "preset_path": repo_path(preset_path),
        "preset_sha256": _sha256(preset_path),
        "card_path": repo_path(card_path),
        "card_sha256": _sha256(card_path),
        "narrator_model": narrator_model,
        "run_params": dict(run_params),
        "run_params_source": run_params_source,
        "source_store_reused": False,
        "gates": "validation",
    }


def _blank(v: object) -> bool:
    """빈 문자열·빈 컨테이너·None. `False`는 빈 값이 아니다.

    `source_store_reused=False`·`git_dirty=False`가 정상값이라 truthy 검사로는
    구분이 안 된다.
    """
    return v is None or (isinstance(v, (str, dict, list)) and not v)


def _require_provenance(d: Dict, path: pathlib.Path) -> None:
    """§9-2 "메타데이터 없는 haystack은 로더가 거부한다".

    빠진 키를 기본값으로 메우면 어떤 코드·설정·원본에서 나온 건초더미인지
    모른 채 측정이 돌아간다 — §9가 막으려는 오염 그 자체다. 조용한 폴백 금지.

    키 **존재**만 보면 전 필드가 ""인 provenance(= git 없는 머신에서 경로가
    깨진 채 얼린 산물, 또는 손으로 만든 스텁)가 통과한다. 그래서 세 가지를
    본다: 누락 / 허용되지 않은 자리의 빈 값 / §9-4 준수 선언의 값.
    """
    prov = d.get("provenance")
    prov = prov if isinstance(prov, dict) else {}   # 비-dict는 "없음"과 같다
    problems = [f"누락 {k}" for k in REQUIRED_PROVENANCE if k not in prov]
    problems += [f"빈 값 {k}" for k in REQUIRED_PROVENANCE
                 if k in prov and k not in PROVENANCE_MAY_BE_UNKNOWN
                 and _blank(prov[k])]
    if "source_store_reused" in prov and prov["source_store_reused"] is not False:
        # §9-4: 병리 세션의 facts/·episodes/ 재사용 금지. freeze는 항상 False를
        # 적고 인제스천은 새로 리플레이한다 — True인 파일은 손편집뿐이다.
        problems.append(
            f"source_store_reused={prov['source_store_reused']!r} "
            f"(§9-4 위반 — 병리 세션 스토어 재사용 선언)")
    if not problems:
        return
    src = prov.get("source_run") or d.get("source_run") or "<원 런 JSON>"
    raise SystemExit(
        f"{path}: provenance {problems} — 스펙 §9-2가 요구하는 생성 이력이 "
        f"없는 동결 파일은 쓸 수 없다(어느 코드·설정·원본에서 나왔는지 모르는 "
        f"건초더미는 오염을 측정에 들여보낸다). 재freeze: python3 -m "
        f"benchmarks.fixed_haystack.haystack freeze {src} "
        f"--id {d.get('haystack_id') or '<id>'}")


def verify_inputs(hs: FixedHaystack) -> None:
    """측정 시점에 조립 입력(preset/card)이 freeze 때 그 파일인지 대조한다.

    §9-1의 "입력은 오직 동결 파일뿐"은 **런 JSON에 대해서만** 참이다 —
    preset과 card는 동결 파일에 실리지 않고 매 런·매 인제스천마다 디스크에서
    새로 읽힌다(`decode_risup(hs.preset_path)`). provenance의 지문은 freeze
    시점 스냅샷일 뿐이라, 아무도 다시 재보지 않으면 "두 건초더미가 같은
    preset/card를 썼다"는 A/B 교차 검증의 전제가 측정 시점엔 아무 보증도
    아니다. 여기가 그 유일한 검문소다.

    freeze가 지문을 못 잰 자리("")는 건너뛴다 — 없는 지문으로 판정할 수는
    없다(§10.4: 모르는 값을 기본값으로 메우지 않는다).
    """
    for kind, path_field, sha_field in (
            ("프리셋", "preset_path", "preset_sha256"),
            ("카드", "card_path", "card_sha256")):
        want = hs.provenance.get(sha_field) or ""
        if not want:
            continue
        path = resolve_path(getattr(hs, path_field))
        got = _sha256(path)
        if got != want:
            raise SystemExit(
                f"{hs.haystack_id}: {kind} 파일이 freeze 시점과 다르다 — "
                f"{path}\n  기록 {want[:12]} vs 현재 "
                f"{got[:12] or '<읽기 실패>'}. 조립 입력이 바뀌면 이 건초더미의 "
                f"와이어가 원 런과 갈리고, 두 건초더미를 같은 조건으로 비교한다는 "
                f"전제(§2.3 A/B 교차 검증)가 깨진다. 파일을 되돌리거나 재freeze")


# ── 동결·직렬화 ────────────────────────────────────────────────────────────

def save(hs: FixedHaystack, path: str) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(hs), ensure_ascii=False, indent=1))


def load(path: str) -> FixedHaystack:
    p = pathlib.Path(path)
    d = json.loads(p.read_text())
    _require_provenance(d, p)
    return FixedHaystack(**{**d,
                           "turns": [FrozenTurn(**t) for t in d["turns"]],
                           "probes": [FrozenProbe(**p) for p in d["probes"]]})


def freeze(run_json_path: str, haystack_id: str,
           preset_path: str = DEFAULT_PRESET, card_path: str = DEFAULT_CARD,
           out_dir: str = DEFAULT_OUT_DIR, *,
           run_params: Optional[Dict] = None,
           run_params_source: str = "",
           wire_fn: Optional[WireFn] = None) -> FixedHaystack:
    """원 런 JSON → FixedHaystack. `{out_dir}/{haystack_id}.json`에 저장하고 반환.

    wire_fn은 validate의 조립 심 — 미지정이면 프리셋을 디코드해 실제
    `run2.build_wire`를 쓴다.

    원 세션 디렉터리를 읽는 유일한 지점이 여기다(스펙 §9-1) — 그것도 런 JSON
    파일 1개뿐이고, 그 사실을 `provenance`에 박제한다(§9-2).
    """
    require_namespace(out_dir, "freeze 출력 디렉터리")
    run = json.loads(pathlib.Path(run_json_path).read_text())
    card = json.loads(pathlib.Path(card_path).read_text())
    probe_turns = {p["turn"] for p in run["probes"]}
    turns = [FrozenTurn(turn=t["turn"], user=t["user"], assistant=t["reply"],
                        is_probe=t["turn"] in probe_turns)
             for t in run["turns"]]
    probes = _build_probes(run["probes"], run["ledger"])
    if wire_fn is None:
        preset = decode_risup(preset_path)

        def wire_fn(window: List[Dict]) -> List[Dict]:
            return build_wire(preset, card, window)

    # 기록 prompt 전량을 넘긴다 — T0만 넘기면 정합성 게이트가 격차 최소
    # 지점만 보고 후반 턴을 보증하지 못한다(§2.2-3 주석 참고).
    validation = validate(turns, probes, run["ledger"], card, wire_fn,
                          {t["turn"]: t.get("prompt", 0) for t in run["turns"]},
                          run["variant"])
    params = dict(run_params or RUN_PARAMS)
    # 원 런 JSON엔 이 세 값이 없어 대조할 진실이 없다 — 그래서 "측정값"이 아니라
    # 조작자 가정임을 출처로 라벨한다(§9-2 "생성 설정"의 정직한 형태).
    source = run_params_source or ("explicit" if run_params
                                   else "assumed-default")
    hs = FixedHaystack(
        haystack_id=haystack_id, source_run=_relpath(run_json_path),
        origin_variant=run["variant"],
        # 얼린 원문을 쓴 모델 그대로 — 프로브 재생성도 이 값으로 해야 문체
        # 일관성이 성립한다(계약 §0: 스펙의 config.MODEL은 실물과 다르다).
        narrator_model=run["model"],
        # ROOT 아래면 상대로 — 워크트리 절대 경로를 박으면 머지 후 워크트리를
        # 지운 순간 동결 파일이 실행 불가가 된다(`repo_path` 주석).
        preset_path=repo_path(preset_path), card_path=repo_path(card_path),
        run_params=params,
        turns=turns, probes=probes, ledger=list(run["ledger"]),
        validation=validation,
        provenance=_provenance(run_json_path, run, preset_path, card_path,
                               run["model"], params, source))
    save(hs, str(pathlib.Path(out_dir) / f"{haystack_id}.json"))
    return hs


def _relpath(path: str) -> str:
    p = pathlib.Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def repo_path(path: str) -> str:
    """ROOT 아래면 ROOT 상대, 아니면 절대. **심링크는 따라가지 않는다.**

    `_relpath`는 `resolve()`로 심링크를 따라가므로 워크트리 밖 실물을 가리키면
    절대 경로가 남는다(§10.4). 조립 입력(preset/card)엔 그 반대가 필요하다 —
    `card_path`에 워크트리 절대 경로를 박으면 브랜치를 머지하고 워크트리를
    지운 순간 동결 파일이 통째로 실행 불가가 된다(`source_run`은 sha256이
    내용을 못박아 대조라도 되지만 card엔 그런 폴백이 없다).
    """
    p = pathlib.Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def resolve_path(path: str) -> str:
    """`repo_path`의 역 — 상대 경로는 현재 ROOT 기준으로 되돌린다."""
    p = pathlib.Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def require_namespace(path: str, what: str) -> pathlib.Path:
    """§9-3 전용 네임스페이스 가드. 위반이면 SystemExit.

    `--out`/`--out-dir`이 검증 없이 통과하면 `--out-dir <…>/dreaming_data
    --id fix-drm-r0`이 root를 **원 세션 디렉터리**(facts/·ledger/·episodes/를
    가진 JsonDirStorage 루트)로 만들고 거기에 `dreaming/live`·`snap`·
    `ingest-meta.json`을 쓴다. `--force`면 그 안에서 rmtree까지 돈다 — §9-3이
    금지한 "기존 세션 디렉터리에 쓰기" 그 자체이고, §9가 병리 데이터로 지목한
    바로 그 디렉터리다.

    막는 것은 **`dreaming_data/` 안에서 `fixed_haystack/` 밖**이다. 데이터
    루트 바깥(스크래치·tmp·테스트)은 세션을 오염시킬 수 없어 대상이 아니다.
    """
    p = pathlib.Path(path).resolve()
    data, bench = DATA_ROOT.resolve(), BENCH_NAMESPACE.resolve()
    if (p == data or data in p.parents) and not (p == bench
                                                 or bench in p.parents):
        raise SystemExit(
            f"{what} {p} 는 스펙 §9-3의 전용 네임스페이스({bench}) 밖이면서 "
            f"세션 디렉터리가 사는 {data} 안이다 — 기존 세션 디렉터리에 쓰면 "
            f"벤치 산출물이 다음 벤치의 입력 후보로 섞이고, 최악에는 원 세션 "
            f"스토어를 덮어쓴다")
    return pathlib.Path(path)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="고정 건초더미 동결")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze", help="원 런 JSON을 FixedHaystack으로 동결")
    fz.add_argument("run_json")
    fz.add_argument("--id", required=True, dest="haystack_id")
    fz.add_argument("--preset", default=DEFAULT_PRESET)
    fz.add_argument("--card", default=DEFAULT_CARD)
    fz.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    # 원 런이 다른 세팅으로 돌았으면 여기서 맞춰 준다 — 기본값만 있으면
    # run_params가 조용히 오라벨되고 트림 예산·프로브 콜이 원본과 갈린다.
    # 기본값을 None으로 두는 이유: 조작자가 **명시했는지**를 알아야 provenance가
    # "cli"와 "assumed-default"를 구분해 적을 수 있다(§9-2 run_params_source).
    fz.add_argument("--max-context", type=int, default=None)
    fz.add_argument("--max-tokens", type=int, default=None)
    fz.add_argument("--reasoning", default=None)
    a = ap.parse_args(argv)
    given = {k: v for k, v in (("max_context", a.max_context),
                               ("max_tokens", a.max_tokens),
                               ("reasoning", a.reasoning)) if v is not None}
    hs = freeze(a.run_json, a.haystack_id, a.preset, a.card, a.out_dir,
                run_params={**RUN_PARAMS, **given} if given else None,
                run_params_source="cli" if given else "assumed-default")
    v = hs.validation
    print(f"{hs.haystack_id}: 턴 {len(hs.turns)}개 · 프로브 {len(hs.probes)}개 · "
          f"원장 {v['ledger_size']}개 (출처 {hs.origin_variant}, "
          f"나레이터 {hs.narrator_model})")
    print(f"  누출 프로브 {len(v['leaky_probes'])}건 {v['leaky_probes']} · "
          f"값<{v['min_value_len']}자 제외 {v['unprotectable_probes']} · "
          f"거리 중앙값 {v['distance_median']} 최소 {v['distance_min']} · "
          f"MIN_PROBE_AGE 미만 {v['distance_young']}")
    print(f"  와이어 T0 추정 {v['wire_t0_estimated']}tok vs 기록 "
          f"{v['wire_t0_recorded']}tok ({v['wire_delta_pct']}%) · "
          f"보정 대조 {v['wire_gated_turns']}/{v['wire_checked_turns']}턴 "
          f"최악 T{v['wire_worst_turn']} {v['wire_worst_delta_pct']}% · "
          f"고정부 {v['fixed_tokens']}tok")
    print(f"  오버플로(추정>{v['provider_ctx_limit']}) "
          f"{v['overflow_turns']} · ledger 대조 {v['ledger_matched']}"
          f"/{len(hs.probes)} (충돌쌍 {v['ledger_ambiguous']})")
    pv = hs.provenance
    print(f"  provenance: 원세션 {pv['source_session']} · "
          f"run {pv['source_run_sha256'][:12] or '?'} · "
          f"git {pv['git_commit'][:12] or '?'}"
          f"{'+dirty' if pv['git_dirty'] is True else ''} · "
          f"preset {pv['preset_sha256'][:12] or '?'} · "
          f"card {pv['card_sha256'][:12] or '?'} · {pv['frozen_at']}")
    # 원 런 JSON에 대조할 진실이 없는 값이라 출처를 같이 보여 준다 —
    # assumed-default면 "이 런이 정말 그 세팅이었나"를 사람이 확인해야 한다.
    print(f"  run_params {hs.run_params} (출처 {pv['run_params_source']})")
    print(f"  → {pathlib.Path(a.out_dir) / (a.haystack_id + '.json')}")


if __name__ == "__main__":
    main()
