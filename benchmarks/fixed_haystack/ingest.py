"""유상태 변형(dreaming/hypa) 1회 리플레이 + 프로브 턴 스냅샷 (구현 계약 §5).

얼린 히스토리를 턴 0부터 순차 소비하되 **프로브 턴을 소비하기 직전**에
저장소 상태를 뜬다. 프로브 실행(assemble.py)은 그 스냅샷의 *복사본*에서
돌아야 반복(runs)·프로브 간 상태가 갈라지지 않는다(계약 §8-2).

인제스천은 haystack당 1회만 하고 반복은 스냅샷을 재사용한다(계약 §5.1) —
조립 알고리즘의 순수 성능을 재기 위한 선택이고, 실사용은 유휴가 짧아
backlog가 남는다는 한계는 리포트에 라벨한다.

`benchmarks/eval/**`·`dreaming/**`는 읽기 전용 — 전부 import 재사용한다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import httpx

from benchmarks.eval import hypa, transport
from benchmarks.eval.config import HYPA_EXPORT, UPSTREAM
from benchmarks.eval.preset2wire import decode_risup
from benchmarks.eval.run2 import build_wire
from benchmarks.fixed_haystack import assemble
from benchmarks.fixed_haystack.haystack import (DEFAULT_OUT_DIR, FixedHaystack,
                                                git_state, load,
                                                require_namespace,
                                                resolve_path, verify_inputs)
from dreaming.dreamer import BATCH_TURNS, Dreamer
from dreaming.llm import LLMClient
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath

# 스냅샷 저장소의 세션 id — assemble.py/run.py가 그대로 읽는다(스냅샷 루트가
# 이미 haystack별이라 값 자체엔 의미가 없지만, 경로가 되므로 고정한다).
# ingest-meta.json에도 박제해 두 모듈이 하드코딩 없이 집어갈 수 있게 한다.
SESSION = "fixed"
DREAM_CAP = 30                                  # 계약 §5.1 backlog 소진 상한
# 원 런의 꿈 모델 (scripts/night_run.sh:50). config.MODEL(나레이터)과 별개 축이다.
DREAM_MODEL = "google/gemini-3-flash-preview"
# 꿈 트리거 정책 라벨 (스펙 §9-2가 요구하는 "생성 설정" 중 꿈 축).
# **`DREAMING_IDLE_SECONDS` 타이머를 쓰지 않는다** — 프로브 스냅샷 직전에
# backlog를 완전히 소진하는 상한 조건이다(계약 §5.1). 이 라벨이 없으면
# 나중에 산출물만 보고 IDLE 병리 데이터와 구분할 수 없다.
DREAM_TRIGGER = "probe-drain-quanta"   # 배치 결정론: 완전 양자만, 잔량 원문 유지
# 스펙 §9-2가 요구하는 "생성 설정" 중 **꿈 축**은 동결 파일이 아니라 이
# 산출물의 속성이라 여기가 진다. 동결 파일 쪽엔 로더 거부(`_require_provenance`)가
# 있는데 이쪽에 없으면 의무가 무상태 축에만 걸리고, 정작 병리(1턴 꿈·IDLE
# 타이머 설정)가 사는 유상태 축은 무방비다 — 그래서 소비자도 이 목록을 강제한다.
# `source_run_sha256`은 동결 파일과 이 스냅샷을 잇는 유일한 식별자다: 어느
# 원문을 리플레이한 스냅샷인지 이게 없으면 조합이 어긋나도 아무도 모른다
# (frozen_at은 재freeze마다 바뀌므로 못박는 자로 쓸 수 없다).
REQUIRED_META = ("git_commit", "git_dirty", "dream_trigger", "dream_cap",
                 "dream_model", "dream_batch_turns", "haystack_id", "session",
                 "source_run_sha256")

SummarizeFn = Callable[[List[Dict[str, str]]], str]


# ── 공통 ──────────────────────────────────────────────────────────────────

def _seed_history(card: Dict) -> List[Dict]:
    # greeting은 history[0]의 assistant (run2.py:447-448 = messages_upto와 동일)
    return ([{"role": "assistant", "content": card["greeting"]}]
            if card.get("greeting") else [])


def _probe_turns(hs: FixedHaystack) -> Set[int]:
    # turns[].is_probe가 단일 진실 — 루프도 이 플래그로 돈다
    return {t.turn for t in hs.turns if t.is_probe}


def _done_or_die(kind: str, have: Set[int], want: Set[int],
                 root: pathlib.Path) -> bool:
    """재실행 안전 — 완주한 스냅샷이 있으면 True(스킵), 부분이면 중단.

    인제스천은 유상태 변형당 LLM 수십 콜이라 비싸다. 그렇다고 부분 스냅샷
    위에 이어 붙이면 live 저장소와 스냅샷의 턴 정합이 깨지므로, 조용히
    재시작하지 않고 --force(=기존 산출물 삭제)를 요구한다.
    """
    if want and want <= have:
        return True
    if have:
        raise SystemExit(
            f"{kind} 스냅샷이 부분만 있다 ({sorted(have)} / 필요 "
            f"{sorted(want)}) — 이어붙이면 턴 정합이 깨진다. "
            f"--force로 {root / kind} 를 지우고 다시 돌려라")
    return False


def _merge_meta(root: pathlib.Path, variant: str, meta: Dict) -> None:
    path = root / "ingest-meta.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    doc[variant] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def _skipped(root: pathlib.Path, variant: str) -> Dict:
    path = root / "ingest-meta.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {**doc.get(variant, {"variant": variant}), "skipped": True}


def require_meta(root, hs: FixedHaystack) -> Dict:
    """dreaming 스냅샷을 쓰기 전 §9-2 꿈 축 메타를 강제한다 (없으면 SystemExit).

    동결 파일 로더와 같은 규약이다: **모르는 값을 메우지 않고 거부한다.** 옛
    메타에 빠진 git_commit·dream_trigger를 지금 값으로 채우면 "그때 그 조건"이
    아니라 "지금 추측"이 기록된다(§10.4) — 그래서 재인제스천(=현행 코드로 새로
    리플레이, §9-4)을 요구한다. 인제스천은 유상태 변형당 LLM 수십 콜이라 비싸지만,
    조건 미상 스냅샷 위에서 나온 측정치는 그보다 비싸다.
    """
    root = pathlib.Path(root)
    path = root / "ingest-meta.json"
    doc = (json.loads(path.read_text(encoding="utf-8"))
           if path.is_file() else {})
    meta = doc.get("dreaming") or {}
    problems = [f"누락 {k}" for k in REQUIRED_META if k not in meta]
    want = hs.provenance.get("source_run_sha256") or ""
    got = meta.get("source_run_sha256") or ""
    if want and got and got != want:
        problems.append(f"source_run_sha256 {got[:12]} ≠ 동결 {want[:12]} "
                        f"(다른 원문을 리플레이한 스냅샷)")
    if not problems:
        return meta
    raise SystemExit(
        f"{path}: dreaming 인제스천 메타 {problems} — 스펙 §9-2가 요구하는 "
        f"생성 설정(코드 버전·꿈 트리거 정책)이 없는 스냅샷으로는 측정할 수 "
        f"없다. 이 스냅샷이 IDLE 병리 조건에서 만들어졌는지 현행 규칙에서 "
        f"만들어졌는지 산출물만 보고는 구분이 안 된다. 재인제스천: python3 -m "
        f"benchmarks.fixed_haystack.ingest --haystack {hs.haystack_id} "
        f"--variants dreaming --force")


# ── dreaming (계약 §5.1) ──────────────────────────────────────────────────

def _snapped_dirs(root: pathlib.Path) -> Set[int]:
    """완주한 dreaming 스냅샷 턴. **내용이 있는 디렉터리만** 센다.

    존재 검사(`is_dir()`)만 하면 빈 디렉터리가 "완주"로 통과한다. copytree
    중간에 죽거나 동기화 사고로 디렉터리 껍데기만 남으면 → 인제스천 스킵 →
    `assemble_dreaming`이 빈 스토어를 복사해 프로덕션 경로를 태움 → dreaming이
    기억 0으로 측정된다. 그건 알고리즘 성능이 아니라 하네스 산물인데 결과
    JSON엔 구분이 안 남는다. 실물 `fix-drm-r0/dreaming/snap/t79`가 지금 파일
    0개(0B)인 채로 남아 있다 — 이론이 아니라 디스크에 있는 상태다.

    첫 스냅샷(T19)도 앞선 19턴을 이미 소비한 뒤라 저장소가 비어 있을 수 없다.
    """
    return {int(p.name[1:]) for p in (root / "dreaming" / "snap").glob("t*")
            if p.is_dir() and p.name[1:].isdigit()
            and any(f.is_file() for f in p.rglob("*"))}


def _require_clean_live(live: pathlib.Path) -> None:
    """리플레이는 **빈 저장소**에서 시작해야 한다 (없으면 SystemExit).

    `_done_or_die`는 `snap/`만 본다. 첫 프로브 턴 전에 프로세스가 죽으면
    스냅샷은 0개(→ 가드 통과)인데 `live`엔 0~18턴 원장이 남고, 재실행이 같은
    저장소에 0~99턴을 **다시** 리플레이해 원장·쌍 해시가 이중화된다. 그 위에서
    뜬 스냅샷이 전 프로브에 쓰인다.

    부분 스냅샷은 --force를 요구하면서 부분 live는 조용히 이어붙이면, 가드가
    막으려던 "이어붙이면 턴 정합이 깨진다"가 반대편 문으로 그대로 들어온다.
    """
    if live.is_dir() and any(f.is_file() for f in live.rglob("*")):
        raise SystemExit(
            f"dreaming live 저장소에 기존 내용이 있다 ({live}) — 같은 저장소에 "
            f"다시 리플레이하면 원장·쌍 해시가 이중화돼 스냅샷이 어느 턴 상태인지 "
            f"의미를 잃는다. --force로 {live.parent} 를 지우고 다시 돌려라")


class MeteredDreamLLM:
    """꿈 콜의 usage를 수거하는 LLMClient (§8-7 인제스천 비용 실측).

    `dreaming.llm.OpenAICompatLLM`은 응답 텍스트만 돌려주고 usage를 버려
    `dream_cost`가 항상 0(=미측정)이 된다 — 이 파일럿의 목적 자체가 비용
    실측이라 그러면 답이 안 나온다. `dreaming/**`는 읽기 전용이므로 벤치
    쪽에 계측판을 댄다. OpenRouter는 `usage:{"include":True}`를 줘야 cost를
    실어 보낸다(transport.py:99와 같은 규약).

    5xx·전송오류는 재시도한다: `Dreamer.dream`이 예외를 삼키고 None을
    돌려주면 `_drain_dreams`가 backlog를 남긴 채 나가버려, 한 번의 502가
    그 스냅샷 이후의 기억을 통째로 비운다(transport.py:61-62의 같은 사고).
    """

    ATTEMPTS = 3
    BACKOFF_SEC = 10

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 180.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"})
        self._model = model
        self.cost = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def complete(self, system: str, user: str) -> str:
        for attempt in range(self.ATTEMPTS):
            try:
                return await self._complete_once(system, user)
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                if (isinstance(e, httpx.HTTPStatusError)
                        and e.response.status_code < 500):
                    raise                      # 4xx는 우리 잘못 — 즉시 전파
                if attempt == self.ATTEMPTS - 1:
                    raise
                await asyncio.sleep(self.BACKOFF_SEC * (attempt + 1))
        raise RuntimeError("unreachable")

    async def _complete_once(self, system: str, user: str) -> str:
        r = await self._client.post("/chat/completions", json={
            "model": self._model, "temperature": 0,
            "usage": {"include": True},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        d = r.json()
        u = d.get("usage") or {}
        self.cost += u.get("cost") or 0.0
        self.calls += 1
        self.prompt_tokens += u.get("prompt_tokens") or 0
        self.completion_tokens += u.get("completion_tokens") or 0
        return d["choices"][0]["message"]["content"] or ""


def _default_dream_llm() -> LLMClient:
    """원 런과 같은 꿈 모델 (night_run.sh:48-50: OpenRouter + gemini flash).

    키는 변수로만 다룬다 — 로그·메타·리포트 어디에도 값을 남기지 않는다.
    """
    return MeteredDreamLLM(
        os.environ.get("DREAMING_DREAM_BASE") or UPSTREAM,
        os.environ.get("DREAMING_DREAM_KEY") or transport.key(),
        os.environ.get("DREAMING_DREAM_MODEL") or DREAM_MODEL)


async def _drain_dreams(dreamer: Dreamer, cap: int) -> Tuple[int, float]:
    """완전 양자 전부 소진 (계약 §5.1 — 배치 결정론 개정 후 트리거 타이밍은
    배치 내용에 영향이 없으므로, 프로브 직전 소진은 임의가 아닌 대표 시점이다).

    배치 결정론 후 dream()은 완전 양자만 소화하고 잔량(<BATCH_TURNS)은 원문로
    남긴다 — has_backlog가 True여도 None이 오면 정상 종료다. 사이클이
    폐기되면 커서가 그대로다(dreamer.py) — 재시도해도 같은 실패라
    무한루프·재결제가 된다. 한 번 실패하면 남은 backlog를 두고 나간다.
    사이클 수는 호출 횟수가 아니라 report["cycles"]로 센다 (dream() 1회가
    여러 양자를 내부 루프로 소화한다).
    """
    n, t0 = 0, time.time()
    while dreamer.has_backlog(SESSION) and n < cap:
        report = await dreamer.dream(SESSION)
        if report is None:
            break
        n += report.get("cycles", 1)
    return n, round(time.time() - t0, 1)


async def _ingest_dreaming(hs: FixedHaystack, preset: Dict, card: Dict,
                           root: pathlib.Path, dream_llm: LLMClient,
                           dream_cap: int,
                           batch_turns: int = BATCH_TURNS) -> Dict:
    live, snap = root / "dreaming" / "live", root / "dreaming" / "snap"
    _require_clean_live(live)
    live.mkdir(parents=True, exist_ok=True)
    storage = JsonDirStorage(live)
    sp = SyncPath(storage, SESSION, mark_cache_enabled=True)
    # 프로덕션과 같은 양자 — batch_turns는 소형 픽스처 테스트용 심이다
    dreamer = Dreamer(storage, dream_llm, batch_turns=batch_turns)

    hist = _seed_history(card)
    cycles, dream_sec = 0, 0.0
    cycles_at: Dict[str, int] = {}
    quarantined: List[int] = []
    turn_sec: Dict[str, float] = {}
    t_all = time.time()
    for t in hs.turns:
        if t.is_probe:
            # 소진 → 스냅샷 → 소비. 프로브 턴 자신의 원문은 스냅샷 뒤에
            # 소비되므로, 이 스냅샷엔 T{turn} 이전 원문만 들어 있다.
            n, sec = await _drain_dreams(dreamer, dream_cap)
            cycles, dream_sec = cycles + n, dream_sec + sec
            cycles_at[str(t.turn)] = n
            shutil.copytree(live, snap / f"t{t.turn}", dirs_exist_ok=True)
        t0 = time.time()
        hist.append({"role": "user", "content": t.user})
        raw = build_wire(preset, card, hist)      # 프록시가 받는 전체 와이어
        _out, verdict = sp.process(raw)
        # process에 넘긴 **원본 raw**를 넘긴다 (proxy.py:149-150) — 주입·마킹
        # 후 와이어를 넘기면 쌍 해시가 달라져 원장이 어긋난다.
        sp.record_response(verdict, raw, t.assistant)
        hist.append({"role": "assistant", "content": t.assistant})
        if verdict.quarantine:
            quarantined.append(t.turn)
        turn_sec[str(t.turn)] = round(time.time() - t0, 2)

    commit, dirty = git_state()
    meta = {
        "variant": "dreaming", "haystack_id": hs.haystack_id,
        "session": SESSION, "turns": len(hs.turns),
        "snapshots": sorted(_probe_turns(hs)),
        # 이 스냅샷이 **어느 동결 원문**을 리플레이한 것인지 — 없으면 나중에
        # 만들어진 동결 파일과 옛 스냅샷의 조합이 조용히 측정에 나간다.
        # sha256은 재freeze에 불변이라(frozen_at과 달리) 못박는 자로 쓸 수 있다.
        "source_run_sha256": hs.provenance.get("source_run_sha256", ""),
        "haystack_frozen_at": hs.provenance.get("frozen_at", ""),
        # 스펙 §9-2의 "생성 설정"은 꿈 축에선 동결 파일이 아니라 **인제스천
        # 산출물**의 속성이다(동결은 원문만 얼린다). 그래서 그 의무를 여기서
        # 채운다 — 코드 버전(git)과 꿈 트리거 정책.
        "git_commit": commit, "git_dirty": dirty,
        "dream_trigger": DREAM_TRIGGER, "dream_cap": dream_cap,
        # 배치 양자는 기억의 밀도·에피소드 경계를 정한다 (같은 100턴이 사실
        # 353 vs 39로 갈린 실측) — 이 숫자 없이는 스냅샷을 해석할 수 없다.
        "dream_batch_turns": batch_turns,
        # 주입 심(fake)은 모델명을 안 들고 있을 수 있다 — 그때는 빈 문자열
        "dream_model": getattr(dream_llm, "_model", ""),
        "dream_cycles": cycles, "dream_cycles_at": cycles_at,
        "dream_sec": round(dream_sec, 1),
        # 비용은 usage를 수거하는 클라이언트(MeteredDreamLLM)일 때만 채워진다.
        # 주입 심(fake)이나 OpenAICompatLLM이면 0.0 = "미측정"이다.
        "dream_cost": round(float(getattr(dream_llm, "cost", 0.0)), 6),
        "dream_calls": int(getattr(dream_llm, "calls", 0)),
        "dream_prompt_tokens": int(getattr(dream_llm, "prompt_tokens", 0)),
        "dream_completion_tokens":
            int(getattr(dream_llm, "completion_tokens", 0)),
        # 소진 실패로 남은 backlog — 0이 아니면 그 뒤 스냅샷의 기억이
        # 조용히 비어 있다(사이클 폐기 시 커서 불변, dreamer.py:360).
        "backlog_left": len(dreamer.snapshot(SESSION)),
        "quarantine_turns": quarantined,
        "quarantine_count": len(quarantined),
        "turn_sec": turn_sec, "sec": round(time.time() - t_all, 1),
    }
    _merge_meta(root, "dreaming", meta)
    if quarantined:
        # night2-drm-r0 사고(105/106턴 격리)의 재발 감지선 — 격리 턴은
        # record_turn을 안 타 원장 체인이 자라지 않는다(sync.py:28-31).
        print(f"[!] 경고: dreaming 인제스천 격리 {len(quarantined)}턴 "
              f"{quarantined[:10]} — 이 스냅샷은 기억이 비어 있을 수 있다",
              file=sys.stderr)
    return meta


def ingest_dreaming(hs: FixedHaystack, preset: Dict, card: Dict, root,
                    *, dream_llm: Optional[LLMClient] = None,
                    dream_cap: int = DREAM_CAP,
                    batch_turns: int = BATCH_TURNS) -> Dict:
    """dreaming 저장소를 얼린 히스토리로 리플레이하고 프로브 턴 스냅샷을 남긴다.

    dream_llm은 주입 심 — 미지정이면 원 런과 같은 꿈 모델을 만든다.
    턴 루프 전체가 하나의 코루틴이라 asyncio.run은 여기서 1회만 돈다.
    """
    root = pathlib.Path(root)
    if _done_or_die("dreaming", _snapped_dirs(root), _probe_turns(hs), root):
        # 스킵은 "옛 산출물을 그대로 쓰겠다"는 뜻이다 — §9-2 메타를 여기서
        # 확인하지 않으면 미충족 스냅샷이 "이미 있음"이라는 초록불만 받는다.
        require_meta(root, hs)
        return _skipped(root, "dreaming")
    return asyncio.run(_ingest_dreaming(
        hs, preset, card, root, dream_llm or _default_dream_llm(), dream_cap,
        batch_turns=batch_turns))


# ── hypa (계약 §5.2) ──────────────────────────────────────────────────────

def ingest_hypa(hs: FixedHaystack, preset: Dict, card: Dict, root,
                *, summarize: Optional[SummarizeFn] = None) -> Dict:
    """hypa data(요약 누적)를 리플레이하고 프로브 턴 직전 상태를 JSON으로 뜬다.

    summarize는 주입 심 — 미지정이면 프로덕션 요약 콜. 요약 디스크 캐시
    (hypa-cache/)를 그대로 쓰므로 재실행이 싸다(계약 §5.2).
    """
    root = pathlib.Path(root)
    snap = root / "hypa" / "snap"
    have = {int(p.stem[1:]) for p in snap.glob("t*.json")
            if p.stem[1:].isdigit()}
    if _done_or_die("hypa", have, _probe_turns(hs), root):
        return _skipped(root, "hypa")
    snap.mkdir(parents=True, exist_ok=True)

    S = hypa.load_hypa_settings(HYPA_EXPORT)
    send = summarize or hypa._summarize_call
    max_ctx = hs.run_params["max_context"]
    max_tok = hs.run_params["max_tokens"]
    # 프리셋+카드 고정 비용은 hypa 카운터로 잰다. 조립(assemble.build)과
    # **같은 함수**를 써야 한다 — 자가 갈리면 배치 경계에 걸린 턴에서 프로브가
    # 인제스천과 다른 요약 배치를 만든다(assemble.hypa_fixed_tokens 주석).
    fixed_tokens = assemble.hypa_fixed_tokens(preset, card)

    hist = _seed_history(card)
    data: Dict = {"summaries": []}
    calls0, cost0, trunc0 = (hypa.SUMMARY_CALLS, hypa.SUMMARY_COST,
                             hypa.SUMMARY_TRUNCATED)
    turn_sec: Dict[str, float] = {}
    err, err_turn, consumed = "", None, 0
    t_all = time.time()
    # 요약 캐시 쓰기를 벤치 네임스페이스로 (§9-3, assemble.bench_summary_cache)
    with assemble.bench_summary_cache():
        for t in hs.turns:
            if t.is_probe:
                # 소비 직전 상태 — 파일로 떨어뜨리는 순간이 곧 불변 사본이다
                (snap / f"t{t.turn}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")
            t0 = time.time()
            hist.append({"role": "user", "content": t.user})
            _mem, _kept, _ksm, data, herr = hypa.hypa_step(
                hist, fixed_tokens, S, data, send, max_ctx, max_tok)
            turn_sec[str(t.turn)] = round(time.time() - t0, 2)
            if herr:
                # 원본도 요청 자체를 실패시킨다 (hypav3.ts:263-274) — 병리
                # 재현이라 부분 결과를 남기고 중단한다.
                err, err_turn = herr, t.turn
                break
            hist.append({"role": "assistant", "content": t.assistant})
            consumed += 1

    commit, dirty = git_state()
    meta = {
        "variant": "hypa", "haystack_id": hs.haystack_id,
        "turns": consumed, "snapshots": sorted(_probe_turns(hs)),
        "git_commit": commit, "git_dirty": dirty,   # §9-2 코드 버전
        "fixed_tokens": fixed_tokens,
        "summary_calls": hypa.SUMMARY_CALLS - calls0,
        "summary_cost": round(hypa.SUMMARY_COST - cost0, 6),
        "summary_truncated": hypa.SUMMARY_TRUNCATED - trunc0,
        "summaries": len(data.get("summaries", [])),
        "error": err, "error_turn": err_turn,
        "turn_sec": turn_sec, "sec": round(time.time() - t_all, 1),
    }
    _merge_meta(root, "hypa", meta)
    if err:
        print(f"[!] 경고: hypa 요약 실패 T{err_turn} — {err} "
              f"(스냅샷은 T{err_turn}까지만 유효)", file=sys.stderr)
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="유상태 변형 인제스천 (1회)")
    ap.add_argument("--haystack", required=True, help="haystack_id")
    ap.add_argument("--variants", default="dreaming,hypa")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dream-cap", type=int, default=DREAM_CAP)
    ap.add_argument("--force", action="store_true",
                    help="기존 인제스천 산출물을 지우고 다시 돈다 (비싸다)")
    a = ap.parse_args(argv)

    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in ("dreaming", "hypa")]
    if bad:
        raise SystemExit(f"인제스천 대상은 dreaming/hypa뿐이다 (무상태 변형은 "
                         f"인제스천 없음, 계약 §5.3): {bad}")
    # §9-3 — 세션 디렉터리를 out-dir로 주면 `root/dreaming/live`가 원 세션
    # 스토어 안에 생기고 --force는 그 안에서 rmtree까지 돈다
    out_dir = require_namespace(a.out_dir, "인제스천 출력 디렉터리")
    hs = load(str(out_dir / f"{a.haystack}.json"))
    # 얼려지지 않은 조립 입력 2개(preset·card)를 지문과 대조한다 (§9-2)
    verify_inputs(hs)
    root = out_dir / a.haystack
    preset = decode_risup(resolve_path(hs.preset_path))
    card = json.loads(pathlib.Path(resolve_path(hs.card_path))
                      .read_text(encoding="utf-8"))

    for v in variants:
        if a.force and (root / v).exists():
            shutil.rmtree(root / v)
        meta = (ingest_dreaming(hs, preset, card, root, dream_cap=a.dream_cap)
                if v == "dreaming" else ingest_hypa(hs, preset, card, root))
        if meta.get("skipped"):
            print(f"{v}: 스냅샷 이미 있음 — 스킵 (--force로 재실행)")
            continue
        if v == "dreaming":
            print(f"dreaming: 턴 {meta['turns']} · 스냅샷 "
                  f"{len(meta['snapshots'])}개 · 꿈 {meta['dream_cycles']}사이클 "
                  f"({meta['dream_sec']}s, ${meta['dream_cost']}, "
                  f"in {meta['dream_prompt_tokens']}/out "
                  f"{meta['dream_completion_tokens']}tok) · 격리 "
                  f"{meta['quarantine_count']}턴 · 잔여backlog "
                  f"{meta['backlog_left']} · 총 {meta['sec']}s")
        else:
            print(f"hypa: 턴 {meta['turns']} · 스냅샷 "
                  f"{len(meta['snapshots'])}개 · 요약 {meta['summary_calls']}콜 "
                  f"(${meta['summary_cost']}) · 누적요약 {meta['summaries']}개 "
                  f"· 총 {meta['sec']}s")
    print(f"  → {root / 'ingest-meta.json'}")


if __name__ == "__main__":
    main()
