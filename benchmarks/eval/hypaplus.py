"""HypaPlus v1.0.9 재현 — 아카 플러그인(내장 HypaV3의 드롭인 후계) 포트.

규범 스펙: docs/superpowers/specs/2026-08-28-hypaplus-port-spec.md
원본 sha256 81b6c4e3f6c395811ff0950f548833cd809d5e4f8dbe7adbde9ee4f692c7da26.

HypaV3(`hypa.py`)와의 근본 차이: **maxContext를 한 번도 읽지 않는다**. 압축은
미요약 메시지 개수로, 주입 예산은 자체 문자 가중 토큰 추정으로만 결정된다.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from benchmarks.eval import hypa

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PLUGIN_SRC = str(pathlib.Path.home() / "Documents" / "arca-modules-2026"
                         / "hypaplus" / "HypaPlus v1_0_9.ts")

# ── 결정론 고정 (스펙 §2) ────────────────────────────────────────────────
# 원본은 `Date.now()`(노드 createdAt, 752/782)와 `nowTimeString()`(facts 시각
# 폴백, 777), 그리고 `new Date().getFullYear()`(MM-DD 파싱의 연도 폴백, 1739)로
# 실시각을 세 군데서 읽는다. 전부 이 상수로 얼린다 — 원본의 "지금" 의미를
# 유지하면서(작중 과거 날짜보다 뒤) 리런 결정론을 얻는다.
FROZEN_NOW = datetime(2026, 8, 28, 0, 0)
FACTS_FALLBACK_TIME = "2026-08-28 00:00"
FALLBACK_YEAR = FROZEN_NOW.year
_EPOCH = datetime(1970, 1, 1)
_FROZEN_NOW_MS = (FROZEN_NOW - _EPOCH).total_seconds() * 1000.0

# 원본 기본값 (HypaPlus v1_0_9.ts:95-97)
DEFAULT_EVENT_FORMAT = ('[기억: time="[[time]]" | location="[[location]]" | '
                        'characters="[[characters]]"\n[[content]]\n[[dialogues]]]')
DEFAULT_FACTS_FORMAT = '[사실: time="[[time]]" | characters="[[characters]]"\n[[content]]]'


@dataclass
class PlusSettings:
    """원본 전역 설정 (252-271). 값은 전부 배포 기본값."""
    chunk_size: int = 30                     # DEFAULT_CHUNK_SIZE (90)
    summarize_threshold: int = 40            # DEFAULT_SUMMARIZE_THRESHOLD (91)
    max_memory_tokens: int = 20000           # DEFAULT_MAX_MEMORY_TOKENS (92)
    include_user_messages: bool = True       # (257)
    embedding_context_messages: int = 5      # (259)
    time_decay_days: float = 15.0            # (262)
    time_scoring_mode: str = "minute"        # (263) minute|day|none
    time_weight: float = 1.0                 # (264)
    similarity_weight: float = 1.0           # (265)
    event_format: str = DEFAULT_EVENT_FORMAT
    facts_format: str = DEFAULT_FACTS_FORMAT
    summary_prompt: str = ""                 # load_default_prompt로 주입


def new_state() -> Dict[str, Any]:
    """createEmptyState (231-238) + 포트 전용 관측 카운터."""
    return {"chunks": [], "seq": 0, "facts_fallback": 0}


# ── 토크나이저 (520-533) ─────────────────────────────────────────────────

def est_tokens(text: str) -> int:
    """heuristicEstimateTokens — 문자 종류별 가중 합. 원본과 같은 순회 순서."""
    tokens = 0.0
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7AF:
            tokens += 0.8          # 한글
        elif 0x4E00 <= code <= 0x9FFF:
            tokens += 0.9          # CJK
        elif 0x3040 <= code <= 0x30FF:
            tokens += 0.9          # 일본어
        else:
            tokens += 0.25
    return math.ceil(tokens)


# ── 시각 파서 (647-680, 1693-1750) ───────────────────────────────────────

_FULL_RE = re.compile(r"^(\d{1,4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})$")
_SHORT_RE = re.compile(r"^(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})$")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{1,2})$")
_RANGE_RE = re.compile(r"^(.+?)(\s*(?:→|->)\s*)(.+)$", re.S)


def _normalize_part(part: str) -> Optional[str]:
    t = part.strip()
    m = _FULL_RE.match(t)
    if m:
        return (f"{m[1].zfill(4)}-{m[2].zfill(2)}-{m[3].zfill(2)} "
                f"{m[4].zfill(2)}:{m[5].zfill(2)}")
    m = _SHORT_RE.match(t)
    if m:
        return f"{m[1].zfill(2)}-{m[2].zfill(2)} {m[3].zfill(2)}:{m[4].zfill(2)}"
    m = _TIME_RE.match(t)
    if m:
        return f"{m[1].zfill(2)}:{m[2].zfill(2)}"
    return None


def normalize_time(ts: str) -> str:
    """normalizeTimeString — 파싱 실패 시 **원문 그대로 반환** (680)."""
    value = ts.strip()
    rm = _RANGE_RE.match(value)
    if rm:
        start = _normalize_part(rm[1])
        end = _normalize_part(rm[3])
        if start and end:
            return f"{start}{rm[2]}{end}"
    got = _normalize_part(value)
    return got if got is not None else value


_N_FULL_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$")
_N_SHORT_RE = re.compile(r"^(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$")
_N_TIME_RE = re.compile(r"^(\d{2}):(\d{2})$")


def _parse_part(value: str, base: Optional[datetime] = None) -> Optional[datetime]:
    m = _N_FULL_RE.match(value)
    if m:
        try:                    # 원본 1725-1733의 필드 재확인과 같은 효과
            return datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]))
        except ValueError:
            return None
    m = _N_SHORT_RE.match(value)
    if m:
        year = base.year if base is not None else FALLBACK_YEAR
        try:
            return datetime(year, int(m[1]), int(m[2]), int(m[3]), int(m[4]))
        except ValueError:
            return None
    m = _N_TIME_RE.match(value)
    if m and base is not None:
        try:
            return base.replace(hour=int(m[1]), minute=int(m[2]),
                                second=0, microsecond=0)
        except ValueError:
            return None
    return None


def parse_time(ts: str) -> Optional[datetime]:
    """parseTimeString — 3형식만 통과. 그 밖은 None (= 시간축에서 탈락)."""
    normalized = normalize_time(ts)
    rm = _RANGE_RE.match(normalized)
    if rm:
        start = _parse_part(rm[1].strip())
        if not start:
            return None
        return _parse_part(rm[3].strip(), start)
    return _parse_part(normalized)


def _time_key(node: Dict[str, Any]) -> float:
    """compareNodesByTime (1751-1760) — 파싱 실패면 createdAt 폴백."""
    d = parse_time(node["time"])
    if d is not None:
        return (d - _EPOCH).total_seconds() * 1000.0
    return node["createdAt"]


# ── 노드 파서 (682-787) ──────────────────────────────────────────────────

_EVENT_BLOCK_RE = re.compile(r"<event>([\s\S]*?)</event>", re.I)
_FACTS_BLOCK_RE = re.compile(r"<facts>([\s\S]*?)</facts>", re.I)
_TIME_FIELD_RE = re.compile(r"(?:시간|Time)\s*:\s*(.+)")
_LOC_FIELD_RE = re.compile(r"(?:장소|Loc)\s*:\s*(.+)")
_CHAR_FIELD_RE = re.compile(r"(?:인물|Chars)\s*:\s*(.+)")
_EVENT_CONTENT_RE = re.compile(
    r"(?:내용|Content)\s*:\s*([\s\S]*?)(?:---\s*(?:대사|Dialogues)\s*---|\Z)")
_FACTS_CONTENT_RE = re.compile(r"(?:내용|Content)\s*:\s*([\s\S]*)")
_DIALOGUE_SECTION_RE = re.compile(r"---\s*(?:대사|Dialogues)\s*---\s*([\s\S]*)")
_DIALOGUE_LINE_RE = re.compile(r'^(.+?)\s*:\s*"(.+)"$')


def _characters(m: Optional[re.Match]) -> List[str]:
    if not m:
        return []
    return [c.strip() for c in m[1].split(",") if c.strip()]


def _created_at(state: Dict[str, Any]) -> float:
    """원본의 Date.now()를 얼린 값 + 삽입 순 카운터 (스펙 §2 #1)."""
    state["seq"] += 1
    return _FROZEN_NOW_MS + state["seq"]


def _parse_event(block: str, node_id: str, state: Dict[str, Any]
                 ) -> Optional[Dict[str, Any]]:
    tm = _TIME_FIELD_RE.search(block)
    cm = _EVENT_CONTENT_RE.search(block)
    if not tm or not cm:
        return None                       # 시간 없는 event는 버린다 (724)
    dialogues: List[Dict[str, str]] = []
    ds = _DIALOGUE_SECTION_RE.search(block)
    if ds:
        for line in ds[1].strip().split("\n"):
            dm = _DIALOGUE_LINE_RE.match(line)
            if dm:
                dialogues.append({"speaker": dm[1].strip(), "text": dm[2].strip()})
    lm = _LOC_FIELD_RE.search(block)
    return {"id": node_id, "type": "event",
            "time": normalize_time(tm[1]),
            "location": lm[1].strip() if lm else None,
            "characters": _characters(_CHAR_FIELD_RE.search(block)),
            "content": cm[1].strip(),
            "dialogues": dialogues or None,
            "createdAt": _created_at(state), "favorite": False}


def _parse_facts(block: str, node_id: str, state: Dict[str, Any]
                 ) -> Optional[Dict[str, Any]]:
    cm = _FACTS_CONTENT_RE.search(block)
    if not cm:
        return None
    tm = _TIME_FIELD_RE.search(block)
    if tm:
        t = normalize_time(tm[1])
    else:
        t = FACTS_FALLBACK_TIME           # 원본은 실시각 (777) — 스펙 §2 #3
        state["facts_fallback"] += 1
    return {"id": node_id, "type": "facts", "time": t, "location": None,
            "characters": _characters(_CHAR_FIELD_RE.search(block)),
            "content": cm[1].strip(), "dialogues": None,
            "createdAt": _created_at(state), "favorite": False}


def get_nodes(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """getNodes (543-551) — 청크 순회 평탄화가 단일 진실 소스."""
    return [n for c in state["chunks"] for n in c["nodes"]]


def parse_nodes(raw: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """parseNodesFromSummary — event 전량 먼저, 그 다음 facts (682-714)."""
    base = 0
    for n in get_nodes(state):
        try:
            num = int(n["id"])
        except (TypeError, ValueError):
            continue
        if num > base:
            base = num
    nxt = base + 1

    nodes: List[Dict[str, Any]] = []
    for m in _EVENT_BLOCK_RE.finditer(raw):
        node = _parse_event(m[1].strip(), str(nxt), state)
        if node:
            nodes.append(node)
            nxt += 1
    for m in _FACTS_BLOCK_RE.finditer(raw):
        node = _parse_facts(m[1].strip(), str(nxt), state)
        if node:
            nodes.append(node)
            nxt += 1
    return nodes


# ── 포매터 (1651-1658, 1940-1967) ────────────────────────────────────────

def node_text(node: Dict[str, Any]) -> str:
    """buildNodeText — 임베딩 입력."""
    return " ".join([node["content"], node.get("location") or "",
                     " ".join(node["characters"]),
                     *[d["text"] for d in (node.get("dialogues") or [])]])


_CLEAN_PIPE_RE = re.compile(r"\s*\|\s+\|\s*")
_CLEAN_HEAD_RE = re.compile(r"^(\[.*?)\s*\|\s*\]", re.M)
_CLEAN_TAIL_RE = re.compile(r"\s*\|\s*\n")
_CLEAN_BLANK_RE = re.compile(r"^[ |\t]+$", re.M)
_CLEAN_NL_RE = re.compile(r"\n{3,}")


def format_node(node: Dict[str, Any], S: PlusSettings) -> str:
    """formatNodeForContext — 템플릿 치환 후 정리 5종을 원본 순서대로."""
    dialogues = node.get("dialogues") or []
    dtext = "\n".join(f'{d["speaker"]}: "{d["text"]}"' for d in dialogues)
    tpl = S.event_format if node["type"] == "event" else S.facts_format
    out = (tpl.replace("[[time]]", node["time"])
              .replace("[[location]]", node.get("location") or "")
              .replace("[[characters]]", ", ".join(node["characters"]))
              .replace("[[content]]", node["content"])
              .replace("[[dialogues]]", dtext))
    out = _CLEAN_PIPE_RE.sub(" | ", out)
    out = _CLEAN_HEAD_RE.sub(r"\1]", out)
    out = _CLEAN_TAIL_RE.sub("\n", out)
    out = _CLEAN_BLANK_RE.sub("", out)
    out = _CLEAN_NL_RE.sub("\n\n", out)
    return out.strip()


# ── 요약 프롬프트 추출 (레포에 전문을 커밋하지 않는다 — 스펙 §0) ─────────

_PROMPT_ANCHOR = "const DEFAULT_SUMMARY_PROMPT = `"


def load_default_prompt(ts_path: str = DEFAULT_PLUGIN_SRC) -> str:
    """원본 .ts에서 DEFAULT_SUMMARY_PROMPT 백틱 리터럴을 잘라 온다.

    조용한 폴백 금지 — 파일이 없거나 앵커가 없으면 즉시 죽는다. 프롬프트가
    바뀌면 요약 캐시 키가 통째로 바뀌므로 실패를 숨기면 안 된다.
    """
    src = pathlib.Path(ts_path).read_text(encoding="utf-8")
    i = src.find(_PROMPT_ANCHOR)
    if i < 0:
        raise RuntimeError(f"DEFAULT_SUMMARY_PROMPT 앵커 없음: {ts_path}")
    i += len(_PROMPT_ANCHOR)
    j = src.find("`;", i)
    if j < 0:
        raise RuntimeError(f"프롬프트 리터럴 종료 없음: {ts_path}")
    return src[i:j].replace("\\`", "`").replace("\\\\", "\\")


# ── ChatML (1462-1479) ───────────────────────────────────────────────────

_CHATML_BLOCK_RE = re.compile(
    r"<\|im_start\|>\s*([^\r\n]+)\r?\n([\s\S]*?)<\|im_end\|>")


def parse_chatml(prompt: str) -> Optional[List[Dict[str, str]]]:
    """블록 사이에 비공백 텍스트가 있으면 None — 순수 ChatML만 파싱한다."""
    blocks: List[Dict[str, str]] = []
    cursor = 0
    for m in _CHATML_BLOCK_RE.finditer(prompt):
        if prompt[cursor:m.start()].strip() != "":
            return None
        role = m[1].strip().split()[0].lower()
        if role not in ("system", "user", "assistant"):
            return None
        blocks.append({"role": role, "content": m[2]})
        cursor = m.end()
    if not blocks or prompt[cursor:].strip() != "":
        return None
    return blocks


# ── 요약 (1481-1603) ─────────────────────────────────────────────────────

CACHE_DIR = ROOT / "dreaming_data" / "fixed_haystack" / "hypa-cache"


def summary_cache_key(batch: List[Dict[str, Any]], S: PlusSettings) -> str:
    """hypa 캐시와 같은 디렉터리를 쓰되 `system` 판별자로 절대 충돌하지 않는다."""
    payload = json.dumps(
        {"system": "hypaplus",
         "chats": [{"role": c["role"], "content": c["content"]} for c in batch],
         "prompt": S.summary_prompt, "model": hypa._lucid_model(),
         "max_tokens": hypa.SUMMARY_MAX_TOKENS, "temp": hypa.SUMMARY_TEMPERATURE},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_chunk(batch: List[Dict[str, Any]], S: PlusSettings,
                    send: Optional[Callable[[List[Dict[str, str]]], str]],
                    cache_dir: Optional[pathlib.Path] = None) -> Optional[str]:
    """청크 하나를 요약한다. 실패(예외·빈 응답)면 None — 원본은 fail-open (1526).

    원본은 메시지를 **role 접두어 없이** content만 이어붙인다 (1485).
    hypa(`f"{role}: {content}"`)와 다르다 — 스펙 §4 #1.
    """
    cdir = CACHE_DIR if cache_dir is None else cache_dir
    key = summary_cache_key(batch, S)
    cached = (cdir / f"{key}.txt") if cdir is not None else None
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8")
    if send is None:
        raise hypa.HypaError(f"요약 캐시 미스인데 send가 없다: {key}")

    text = "\n\n".join(c["content"] for c in batch)
    messages = parse_chatml(S.summary_prompt.replace("{{slot}}", text)) or [
        {"role": "user", "content": S.summary_prompt.replace("{{slot}}", text)}]

    hypa._TRUNCATED = False
    try:
        resp = send(messages)
    except Exception:                     # 원본 1526-1529: 로그 찍고 None
        return None
    out = hypa._THOUGHTS_RE.sub("", resp or "").strip()
    if not out:
        return None
    if cached is not None and not hypa._TRUNCATED:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(out, encoding="utf-8")
    return out


def last_summarized_index(state: Dict[str, Any]) -> int:
    """getLastSummarizedMsgIndex (888-902).

    **실패 청크도 센다** — 요약에 실패한 30개 메시지는 영구히 요약 대상에서
    빠지고 원문 창에서도 빠진다 (스펙 §4 #2, 기억 유실 경로).
    """
    mx = -1
    for chunk in state["chunks"]:
        if chunk["id"] == "0":            # 수동 청크는 제외 (890)
            continue
        for idx in chunk["msg_indices"]:
            if idx > mx:
                mx = idx
    return mx


def _next_chunk_id(state: Dict[str, Any]) -> str:
    mx = 0
    for c in state["chunks"]:
        try:
            num = int(c["id"])
        except (TypeError, ValueError):
            continue
        if num > mx:
            mx = num
    return str(mx + 1)


def run_summarization(msgs: List[Dict[str, Any]], state: Dict[str, Any],
                      S: PlusSettings,
                      send: Optional[Callable[[List[Dict[str, str]]], str]],
                      cache_dir: Optional[pathlib.Path] = None,
                      ) -> Optional[Dict[str, Any]]:
    """runSummarization — 청크 하나를 소비. 소비 안 했으면 None.

    반환은 trace 이벤트용 dict.
    """
    start = last_summarized_index(state) + 1
    window = msgs[start:start + S.chunk_size]
    if len(window) < S.chunk_size:        # 부분 청크는 대기 (1552-1556)
        return None
    batch = (window if S.include_user_messages
             else [m for m in window if m["role"] != "user"])
    if not batch:
        return None

    chunk = {"id": _next_chunk_id(state),
             "msg_indices": list(range(start, start + len(window))),
             "nodes": [], "summarized": "pending"}
    state["chunks"].append(chunk)

    raw = summarize_chunk(batch, S, send, cache_dir)
    if raw is None:
        chunk["summarized"] = "failed"    # 인덱스는 이미 소비됐다 (1596-1599)
    else:
        chunk["nodes"] = parse_nodes(raw, state)
        chunk["summarized"] = "done"
    return {"ev": "chunk", "chunk_id": chunk["id"], "start": start,
            "end": start + len(window) - 1, "n_msgs": len(batch),
            "status": chunk["summarized"], "n_nodes": len(chunk["nodes"])}


def track(msgs: List[Dict[str, Any]], state: Dict[str, Any], S: PlusSettings,
          send: Optional[Callable[[List[Dict[str, str]]], str]],
          cache_dir: Optional[pathlib.Path] = None,
          trace: Optional[list] = None) -> None:
    """trackUserMessage/trackAssistantMessage의 공통 루프 (2010-2021).

    pending이 임계 미만이 될 때까지, 또는 줄어들지 않을 때까지 반복.
    """
    while True:
        pending = len(msgs) - (last_summarized_index(state) + 1)
        if pending < S.summarize_threshold:
            return
        ev = run_summarization(msgs, state, S, send, cache_dir)
        if ev is None:                    # 소비 못 하면 중단 (무한루프 방지)
            return
        if trace is not None:
            trace.append(ev)


# ── 검색 (1429-1451, 1673-1691, 1763-1938) ──────────────────────────────

def build_query(msgs: List[Dict[str, Any]], S: PlusSettings) -> str:
    """buildEmbeddingQuery — 마지막 메시지 + 그 앞 N-1개를 "\\n"으로 결합."""
    if not msgs:
        return ""
    start = max(0, len(msgs) - S.embedding_context_messages)
    subset = msgs[start:]
    return "\n".join(m["content"] for m in subset)


def latest_node_time(state: Dict[str, Any]) -> Optional[str]:
    """getLatestNodeTime — 파싱 성공한 노드들 중 최댓값. 전부 실패하면 None."""
    latest: Optional[Dict[str, Any]] = None
    latest_d: Optional[datetime] = None
    for node in get_nodes(state):
        d = parse_time(node["time"])
        if d is None:
            continue
        if latest_d is None or d > latest_d:
            latest, latest_d = node, d
    return latest["time"] if latest is not None else None


def time_score(node: Dict[str, Any], last_time: Optional[str],
               S: PlusSettings) -> float:
    """computeTimeScore — 1/√(1+Δd/T). 파싱 실패는 0.5 고정(시간축 사망)."""
    if S.time_scoring_mode == "none":
        return 0.0
    if not last_time:
        return 0.5
    nd = parse_time(node["time"])
    rd = parse_time(last_time)
    if nd is None or rd is None:
        return 0.5
    if S.time_scoring_mode == "day":
        nd = nd.replace(hour=0, minute=0, second=0, microsecond=0)
        rd = rd.replace(hour=0, minute=0, second=0, microsecond=0)
    diff_days = abs((nd - rd).total_seconds()) / 86400.0
    return 1.0 / math.sqrt(1.0 + diff_days / S.time_decay_days)


def _node_vectors(nodes: List[Dict[str, Any]], state: Dict[str, Any],
                  embed: Callable) -> List[Sequence[float]]:
    """노드 벡터 — 원본은 노드에 영속 저장, 포트는 state 내 dict (스펙 §3)."""
    cache = state.setdefault("vec", {})
    texts = [node_text(n) for n in nodes]
    missing = [t for t in texts if t not in cache]
    if missing:
        for t, v in zip(missing, embed(missing)):
            cache[t] = v
    return [cache[t] for t in texts]


def retrieve(state: Dict[str, Any], query: str, last_time: Optional[str],
             budget: int, S: PlusSettings, embed: Optional[Callable] = None,
             trace: Optional[list] = None) -> List[Dict[str, Any]]:
    """retrieveRelevantNodes — 즐겨찾기 선입 후 combined 내림차순 그리디."""
    all_nodes = get_nodes(state)
    if not all_nodes:
        _plus_tr(trace, ev="retrieve", n_nodes=0, skipped="no_nodes")
        return []

    favorites = [n for n in all_nodes if n.get("favorite")]
    non_fav = [n for n in all_nodes if not n.get("favorite")]

    # 임베딩 미설정이 배포 기본값 — 그 경우 유사도는 전부 0.5로 고정된다.
    sim_scores: Dict[str, float] = {}
    if embed is not None and non_fav:
        qv = embed([query])[0]
        for node, nv in zip(non_fav, _node_vectors(non_fav, state, embed)):
            sim_scores[node["id"]] = hypa._cosine(qv, nv)

    scored: List[Tuple[Dict[str, Any], float, float, float]] = []
    for node in non_fav:
        sim = sim_scores.get(node["id"], 0.5)
        ts = time_score(node, last_time, S)
        tf = 1.0 if S.time_scoring_mode == "none" else (1.0 - ts) ** S.time_weight
        combined = 1.0 - tf * (1.0 - sim) ** S.similarity_weight
        scored.append((node, sim, ts, combined))

    ranked = sorted(scored, key=lambda x: x[3], reverse=True)

    selected: List[Dict[str, Any]] = []
    used = 0
    for node in sorted(favorites, key=_time_key):
        cost = est_tokens(format_node(node, S))
        if used + cost > budget:
            continue
        selected.append(node)
        used += cost

    n_floor = n_budget = 0
    for node, _sim, _ts, combined in ranked:
        if combined < 0.05:               # 바닥 컷 (1917)
            n_floor += 1
            continue
        cost = est_tokens(format_node(node, S))
        if used + cost > budget:          # 자르지 않고 건너뛴다 (1922)
            n_budget += 1
            continue
        selected.append(node)
        used += cost

    selected.sort(key=_time_key)

    picked = {n["id"] for n in selected}
    n_parsed = sum(1 for n in all_nodes if parse_time(n["time"]) is not None)
    _plus_tr(trace, ev="retrieve", n_nodes=len(all_nodes), query_len=len(query),
             ref_time=last_time, ref_time_parsed=bool(last_time and
                                                      parse_time(last_time)),
             budget=budget, used_tokens=used, n_selected=len(selected),
             n_skipped_budget=n_budget, n_skipped_floor=n_floor,
             embedded=embed is not None)
    _plus_tr(trace, ev="scores",
             rows=[[n["id"], round(sim, 6), round(ts, 6), round(cb, 6),
                    n["id"] in picked] for n, sim, ts, cb in ranked])
    _plus_tr(trace, ev="time_diag", n_nodes=len(all_nodes),
             n_time_parsed=n_parsed, n_time_failed=len(all_nodes) - n_parsed,
             n_facts_fallback=state.get("facts_fallback", 0))
    return selected


def _plus_tr(trace: Optional[list], **ev: Any) -> None:
    if trace is not None:
        trace.append(ev)


def build_memory(msgs: List[Dict[str, Any]], state: Dict[str, Any],
                 S: PlusSettings, embed: Optional[Callable] = None,
                 trace: Optional[list] = None) -> str:
    """buildMemoryContext — 노드 0개면 빈 문자열(주입 자체가 없다)."""
    query = build_query(msgs, S)
    nodes = retrieve(state, query, latest_node_time(state),
                     S.max_memory_tokens, S, embed, trace)
    if not nodes:
        return ""
    return "\n".join(format_node(n, S) for n in nodes)


# ── 주입 (2136-2189) ─────────────────────────────────────────────────────

def inject(msgs: List[Dict[str, Any]], memory: str) -> List[Dict[str, Any]]:
    """Strategy 2 — 프리셋에 [HypaPlus.memory] 플레이스홀더가 없을 때의 경로.

    첫 system 메시지 끝에 append. system이 없으면 index 0에 unshift.
    memory가 비면 무변경 (2185-2189).
    """
    out = [dict(m) for m in msgs]
    if not memory:
        return out
    for i, m in enumerate(out):
        if m.get("content") and "[HypaPlus.memory]" in m["content"]:
            out[i]["content"] = m["content"].replace("[HypaPlus.memory]", memory)
            return out
    for i, m in enumerate(out):
        if m["role"] == "system":
            out[i]["content"] = m["content"] + "\n\n" + memory
            return out
    out.insert(0, {"role": "system", "content": memory})
    return out


# ── 턴 1회 (인스펙터가 hypa_step과 동형으로 부른다) ──────────────────────

def hypaplus_step(history: List[Dict[str, Any]], S: PlusSettings,
                  state: Dict[str, Any],
                  send: Optional[Callable[[List[Dict[str, str]]], str]],
                  embed: Optional[Callable] = None,
                  cache_dir: Optional[pathlib.Path] = None,
                  trace: Optional[list] = None,
                  ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, Any],
                             Optional[str]]:
    """beforeRequest 한 번 분. 반환 (memory_text, kept, kept_start_msg, state, error).

    error는 항상 None — HypaPlus엔 하드 실패 경로가 없다 (fail-open, 2196-2199).
    memory_text는 노드 0개면 빈 문자열 (hypa의 None과 다르되 둘 다 falsy).
    """
    track(history, state, S, send, cache_dir, trace)
    memory = build_memory(history, state, S, embed, trace)
    kept_start = last_summarized_index(state) + 1
    return memory, history[kept_start:], kept_start, state, None


def hypaplus_post(history: List[Dict[str, Any]], S: PlusSettings,
                  state: Dict[str, Any],
                  send: Optional[Callable[[List[Dict[str, str]]], str]],
                  cache_dir: Optional[pathlib.Path] = None,
                  trace: Optional[list] = None) -> None:
    """afterRequest 한 번 분 (2024-2041). 이 건초더미에선 증명상 no-op이지만
    충실성을 위해 유지한다 — 다른 히스토리 규약에선 발동할 수 있다."""
    track(history, state, S, send, cache_dir, trace)
