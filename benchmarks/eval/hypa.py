"""HypaV3 재현 (뮈토스 하이파 V5 export 고정, exp 경로 단일).

규범 스펙: docs/superpowers/plans/2026-08-09-refs/hypav3-algorithm.md
Task 4: 설정 로드 + 토크나이저 + chats 변환 계약.
Task 5: 요약 배치 계획 + summarize.
Task 6: 선택 4단계 + 임베딩.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import tiktoken

from benchmarks.eval import config, transport

_ENC = tiktoken.get_encoding("o200k_base")

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "dreaming_data" / "eval" / "hypa-cache"


class HypaError(RuntimeError):
    """요약 실패 — 원본은 요청 전체를 에러로 끝낸다 (hypav3.ts:1729-1740)."""


@dataclass
class HypaSettings:
    max_chats_per_summary: int = 8
    query_chat_count: int = 3
    memory_tokens_ratio: float = 0.39
    extra_summarization_ratio: float = 0.01
    recent_memory_ratio: float = 0.6
    similar_memory_ratio: float = 0.4
    do_not_summarize_user_message: bool = False
    summarization_prompt: str = ""
    # export에 없는 필드 (HV3:1803 기본값) — hypav3.ts:105-116
    summary_chunk_separator: str = "\\n\\n"


# export 필드명 -> HypaSettings 필드명 (§1 표, 화이트리스트 병합 대상만)
_FIELD_MAP = {
    "maxChatsPerSummary": ("max_chats_per_summary", int),
    "queryChatCount": ("query_chat_count", int),
    "memoryTokensRatio": ("memory_tokens_ratio", float),
    "extraSummarizationRatio": ("extra_summarization_ratio", float),
    "recentMemoryRatio": ("recent_memory_ratio", float),
    "similarMemoryRatio": ("similar_memory_ratio", float),
    "doNotSummarizeUserMessage": ("do_not_summarize_user_message", bool),
    "summarizationPrompt": ("summarization_prompt", str),
}


def load_hypa_settings(export_path: str) -> HypaSettings:
    """뮈토스 하이파 V5 export → HypaSettings.

    "키 존재 + typeof 일치" 화이트리스트 병합 (hypav3.ts:1814-1824).
    summary_chunk_separator는 export에 없으므로 항상 코드 기본값을 쓴다.
    """
    with open(export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_settings = raw.get("data", {}).get("settings", {})

    kwargs: Dict[str, Any] = {}
    for export_key, (field_name, expected_type) in _FIELD_MAP.items():
        if export_key in raw_settings and isinstance(raw_settings[export_key], expected_type):
            kwargs[field_name] = raw_settings[export_key]
    return HypaSettings(**kwargs)


def tok_chat(chat: Dict[str, Any]) -> int:
    """tokenizer.tokenize_chat 재현 — 비-gpt 경로 (index.svelte.ts:287-293).

    name 항은 벤치 chats에 name 필드가 없어 생략 (스펙 이탈 아님).
    """
    return len(_ENC.encode(chat["content"])) + 3


def to_risu_chats(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """벤치 히스토리 -> RisuAI chats 배열 (memo 계약).

    memo = f"m{i}" — 히스토리 내 메시지 인덱스(greeting 포함 0부터).
    인덱스 기반이므로 edit_at(content 변형)·리롤(같은 자리 교체)에도 memo가
    불변 — start_idx 매칭(hypav3.ts:214-229)이 깨지지 않는다.
    """
    return [
        {"role": msg["role"], "content": msg["content"], "memo": f"m{i}"}
        for i, msg in enumerate(history)
    ]


# --- 요약 배치 계획 (hypav3.ts:253-378, §2 의사코드) ---

_SKIP_MEMOS = ("NewChatExample", "NewChat")
_SKIP_NAMES = ("example_user", "example_assistant")


def _is_skipped(chat: Dict[str, Any], S: HypaSettings) -> bool:
    return (chat.get("name") in _SKIP_NAMES
            or chat.get("memo") in _SKIP_MEMOS
            or chat["content"].strip() == ""
            or (S.do_not_summarize_user_message and chat["role"] == "user"))


def plan_batches(
    chats: List[Dict[str, Any]],
    start_idx: int,
    current_tokens: int,
    max_ctx: int,
    S: HypaSettings,
) -> Tuple[List[List[Dict[str, Any]]], float, int, Optional[str]]:
    """요약할 배치 목록을 계획한다 — 순수 함수.

    호출자가 current_tokens에 대해 지켜야 할 선행 조건 2개 (§2, 이 순서 그대로):
      1. db.maxResponse를 **이미 뺀** 값일 것 (hypav3.ts:197).
      2. should_reserve(= 요약이 1개 이상이거나 current_tokens > max_ctx)면
         memory_tokens = floor(max_ctx * memoryTokensRatio)를 **더한** 값일 것
         (hypav3.ts:238-250). 이 선점이 스스로 요약 발동을 유발하는 것이 §7 #11
         자기증식 병리이고 벤치가 보이려는 현상 자체다 — 예약 전 값을 넘기면
         함수는 멀쩡히 동작하면서 병리만 조용히 사라진다.

    반환은 (배치들, 남은 current_tokens, 새 start_idx, 에러).
    **에러가 있으면 배치는 항상 빈 리스트다** — 원본은 toSummarizeArray를 통째로
    버리고 요청 자체를 실패시키며, 요약 호출(rate-limiter 블록, hypav3.ts:382)에
    도달조차 하지 않는다 (hypav3.ts:265-273). 에러 시 남은 토큰·start_idx는
    진단용 값일 뿐이다.
    """
    batches: List[List[Dict[str, Any]]] = []
    if current_tokens <= max_ctx:          # summarizationMode (hypav3.ts:253)
        return batches, current_tokens, start_idx, None

    target_tokens = max_ctx * (1 - S.extra_summarization_ratio)

    while True:
        if current_tokens <= target_tokens:
            break
        if len(chats) - start_idx <= S.query_chat_count:
            # 마지막 queryChatCount개는 불가침 — 더는 줄일 수단이 없다.
            if current_tokens <= max_ctx:
                break
            # 원본은 계획된 배치를 버리고 요청을 실패시킨다 — 요약은 한 번도
            # 호출되지 않으므로 배치를 돌려주면 없던 비용이 생긴다.
            return ([], current_tokens, start_idx,
                    f"Cannot summarize further: {current_tokens} > {max_ctx}, "
                    f"but minimum {S.query_chat_count} messages required.")

        batch: List[Dict[str, Any]] = []
        batch_tokens = 0
        i = start_idx
        # maxChatsPerSummary는 '담긴 개수' 상한이지 스캔 범위가 아니다 (§7 #3).
        while len(batch) < S.max_chats_per_summary and i < len(chats) - S.query_chat_count:
            chat = chats[i]
            batch_tokens += tok_chat(chat)     # 스킵돼도 누적 — 원본 버그 보존
            if not _is_skipped(chat, S):       # (hypav3.ts:313, §7 #2)
                batch.append(chat)
            i += 1

        # 과요약 방지 (hypav3.ts:352-362)
        if current_tokens <= max_ctx and current_tokens - batch_tokens < target_tokens:
            break

        if batch:                              # 요약 0개 배치는 안 넣는다
            batches.append(batch)
        current_tokens -= batch_tokens
        start_idx = i

    return batches, current_tokens, start_idx, None


# --- summarize (hypav3.ts:1673-1773, §3) ---

_INLAY_RE = re.compile(r"{{(inlay|inlayed|inlayeddata)::(.+?)}}")
_THOUGHTS_RE = re.compile(r"<Thoughts>[\s\S]*?</Thoughts>")
_CHATML_STARTER = "<|im_start|>"
_DEFAULT_PROMPT = ("[Summarize the ongoing role story, It must also remove "
                   "redundancy and unnecessary text and content from the output.]")

# 이탈 기록 2건 (원본 대비):
#  - max_tokens=8192 — 원본은 db.maxResponse(뮈토스 30000). 요약 길이 상한만
#    다르고 실측 요약은 그보다 훨씬 짧다.
#  - temperature=0.0 — 원본은 db.temperature 센티널 -1000이라 해석 미확정.
#    리런 결정론을 위해 0.0으로 고정한다.
SUMMARY_MAX_TOKENS = 8192
SUMMARY_TEMPERATURE = 0.0

# _summarize_call이 누적하는 요약 비용 (run2가 cost_hypa로 수거).
SUMMARY_COST = 0.0
SUMMARY_CALLS = 0
# max_tokens 8192 이탈의 부작용 감시 — finish_reason == "length" 횟수.
# 원본(30000)에선 사실상 불가능했던 절단이라 별도로 센다.
SUMMARY_TRUNCATED = 0
_TRUNCATED = False


def mark_truncated() -> None:
    """send가 길이 절단(finish_reason == "length")을 알린다.

    절단된 요약은 영구 디스크 캐시에 쓰지 않는다 — 한 번 쓰면 이후 모든
    리런이 잘린 요약을 재사용한다.
    """
    global SUMMARY_TRUNCATED, _TRUNCATED
    SUMMARY_TRUNCATED += 1
    _TRUNCATED = True


def _director_model() -> str:
    """요약 모델 — HYPA_SUMMARY_MODEL env, 기본 DIRECTOR_MODEL 상속."""
    return transport.SUMMARY_MODEL


def _sanitize(content: str) -> str:
    """sanitizeSummaryContent — 인레이 이미지 토큰을 [Image]로 (hypav3.ts:1677)."""
    return _INLAY_RE.sub("[Image]", content)


def _parse_chatml(data: str) -> Optional[List[Dict[str, str]]]:
    """chatML.ts:8-11의 null 가드만 재현.

    고정 export(뮈토스 하이파 V5)의 summarizationPrompt는 starter로 시작하지
    않고 {{slot}}도 없어 항상 None이다 — 파싱 분기는 도달 불가.
    """
    if data.strip().startswith(_CHATML_STARTER):
        raise HypaError("chatML 경로는 고정 export에서 도달 불가 — 스펙 가정 붕괴")
    return None


def summary_cache_key(batch: List[Dict[str, Any]], S: HypaSettings) -> str:
    """캐시는 디스크에 영구 잔류한다 — 요약 결과를 바꾸는 입력은 전부 키에 넣는다.

    model 포함: DIRECTOR_MODEL은 env(DREAMING_EVAL_DIRECTOR)로 바뀌고 요약기
    비교 런도 있을 수 있다. 빠지면 다른 모델이 만든 요약을 조용히 재사용한다.
    """
    payload = json.dumps(
        {"chats": [{"role": c["role"], "content": c["content"]} for c in batch],
         "prompt": S.summarization_prompt, "model": _director_model(),
         "max_tokens": SUMMARY_MAX_TOKENS, "temp": SUMMARY_TEMPERATURE},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize(
    send: Callable[[List[Dict[str, str]]], str],
    batch: List[Dict[str, Any]],
    S: HypaSettings,
    cache_dir: Optional[pathlib.Path] = CACHE_DIR,
) -> str:
    """배치 하나를 요약한다. send는 OpenAI 메시지 배열 → 응답 문자열.

    기본은 CACHE_DIR 디스크 캐시 (리런 결정론·재결제 방지). None이면 캐시 없음.
    """
    key = summary_cache_key(batch, S)
    cached = cache_dir / f"{key}.txt" if cache_dir else None
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8")

    str_messages = "\n".join(
        f"{c['role']}: {_sanitize(c['content'])}" for c in batch)
    prompt = (S.summarization_prompt if S.summarization_prompt.strip()
              else _DEFAULT_PROMPT)
    messages = _parse_chatml(prompt.replace("{{slot}}", str_messages)) or [
        {"role": "user", "content": str_messages},
        {"role": "system", "content": prompt},      # system이 뒤 — 폴백 순서
    ]

    global _TRUNCATED
    _TRUNCATED = False
    resp = send(messages)
    if not resp.strip():
        raise HypaError("Empty summary returned")
    out = _THOUGHTS_RE.sub("", resp).strip()
    if not out:
        raise HypaError("Empty summary after removing thoughts content")

    if cached is not None and not _TRUNCATED:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(out, encoding="utf-8")
    return out


def _summarize_call(messages: List[Dict[str, str]]) -> str:
    """프로덕션 send — 일시 오류(5xx·타임아웃)는 재시도.

    run2._call_upstream과 같은 정책·같은 엔드포인트다. 발동 후엔 요약이 거의
    매 턴 일어나므로 나레이터보다 노출이 크다 — 502 한 방에 런이 죽는다.
    """
    for attempt in range(3):
        try:
            return _summarize_call_once(messages)
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if (isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500):
                raise                          # 4xx는 우리 잘못 — 즉시 전파
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("unreachable")


def _summarize_call_once(messages: List[Dict[str, str]]) -> str:
    """OpenRouter 직접 호출 (db.subModel = Gemini 3 Flash 계열)."""
    global SUMMARY_COST, SUMMARY_CALLS

    r = httpx.post(
        f"{config.UPSTREAM}/chat/completions", timeout=120,
        headers={"Authorization": f"Bearer {transport.key()}"},
        json={"model": _director_model(), "max_tokens": SUMMARY_MAX_TOKENS,
              "temperature": SUMMARY_TEMPERATURE, "messages": messages,
              "usage": {"include": True}})    # 없으면 usage에 cost가 안 실린다
    r.raise_for_status()
    data = r.json()
    SUMMARY_COST += (data.get("usage") or {}).get("cost") or 0.0
    SUMMARY_CALLS += 1
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        mark_truncated()                       # 절단본은 캐시에 남기지 않는다
    return choice["message"]["content"] or ""


# --- 선택 4단계 (hypav3.ts:502-874, §2 (a)~(d)) ---

SUMMARY_SEP = "\n\n"          # hypav3.ts:103
_SEP_REGEX_RE = re.compile(r"^/(.+)/([gimuy]*)$")


def _ident(obj: Any) -> Any:
    """dict 삽입 키 — JS Map 의 키 의미를 맞춘다.

    요약/청크는 dict(unhashable)이고 원본은 참조 동일성(===)으로 구분하므로
    id로 키잉한다. 문자열 등 hashable은 값으로 키잉해 동치 병합을 허용한다.
    """
    try:
        hash(obj)
    except TypeError:
        return ("__id__", id(obj))
    return obj


def _index_of(seq: Sequence[Any], obj: Any) -> int:
    """참조 동일성 기준 인덱스 — 내용이 같은 요약이 섞여도 안 헷갈린다."""
    for i, x in enumerate(seq):
        if x is obj:
            return i
    return -1


def _unused(summaries: Sequence[Any], selected: Sequence[Any]) -> List[Any]:
    return [s for s in summaries if _index_of(selected, s) == -1]


def _summary_tokens(summary: Dict[str, Any]) -> int:
    """선택 단계의 토큰 계산 — 요약 뒤에 구분자가 붙는다 (hypav3.ts:512 등)."""
    return tok_chat({"role": "system", "content": summary["text"] + SUMMARY_SEP})


def split_by_separator(text: str, sep: str) -> List[str]:
    """hypav3.ts:105-116 — `/regex/flags` 형식이면 본문만, 아니면 sep 자체가 정규식."""
    m = _SEP_REGEX_RE.match(sep)
    try:
        return re.split(m.group(1) if m else sep, text)
    except re.error:
        return text.split(SUMMARY_SEP)


def simple_cc(scored_lists: Sequence[Sequence[Tuple[Any, float]]],
              weights: Sequence[float]) -> List[Any]:
    """가중 점수 합산 (hypav3.ts:1832-1852). 동점은 첫 등장 순서 유지."""
    scores: Dict[Any, float] = {}
    items: Dict[Any, Any] = {}
    for lst, w in zip(scored_lists, weights):
        for item, sc in lst:
            k = _ident(item)
            items.setdefault(k, item)
            scores[k] = scores.get(k, 0.0) + sc * w
    order = sorted(scores.items(), key=lambda kv: -kv[1])   # stable → 삽입순 유지
    return [items[k] for k, _ in order]


def child_to_parent_rrf(ranked_children: Iterable[Any],
                        key: Callable[[Any], Any],
                        k: int = 60) -> List[Any]:
    """자식(청크) 순위를 부모(요약) 점수로 접는다 (hypav3.ts:1874-1893)."""
    scores: Dict[Any, float] = {}
    parents: Dict[Any, Any] = {}
    for rank, child in enumerate(ranked_children, start=1):
        parent = key(child)
        pk = _ident(parent)
        parents.setdefault(pk, parent)
        scores[pk] = scores.get(pk, 0.0) + 1.0 / (k + rank)
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    return [parents[pk] for pk, _ in order]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """exp 경로 유사도 (hypamemoryv2.ts:357-369). 정규화 벡터면 내적과 동일."""
    dot = math.fsum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(math.fsum(float(x) * float(x) for x in a))
    nb = math.sqrt(math.fsum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


_EMBED_MODEL: Any = None


def embed(texts: List[str]) -> Any:
    """MiniLM 임베딩 (transformers.ts:82의 기본 모델과 동일 계열).

    lazy 로드 — sentence-transformers는 torch를 끌고 와 설치가 무겁다.
    미설치면 조용한 폴백 대신 명확한 에러를 낸다.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:      # pragma: no cover - 설치 환경 의존
            raise HypaError(
                "sentence-transformers 미설치 — hypa similar 선택은 임베딩이 "
                "필수다. `pip install sentence-transformers` (torch 동반)."
            ) from e
        _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # 스펙 §6 대체안 A: transformers.js 경로는 tokenizer model_max_length=512로
        # 자른다. sentence-transformers 기본은 sentence_bert_config.json의 256이라
        # 한국어 문단(344자 ≈ wordpiece 494토큰)이 절반쯤 잘린다 → 512로 맞춘다.
        _EMBED_MODEL.max_seq_length = 512
    return _EMBED_MODEL.encode(list(texts), normalize_embeddings=True)


_DEFAULT_EMBED = embed


def select_summaries(
    summaries: List[Dict[str, Any]],
    available_tokens: int,
    recent_chats: List[Dict[str, Any]],
    S: HypaSettings,
    embed: Optional[Callable[[List[str]], Any]] = None,
) -> List[Dict[str, Any]]:
    """important → recent → similar → (random 스킵) 그리디 채우기.

    반환은 검색 순위가 아니라 `summaries` 인덱스 기준 시간순 (§7 #8).
    """
    available = available_tokens
    selected: List[Dict[str, Any]] = []

    # (a) important — available을 직접 깎는 유일한 단계 (§7 #6), 초과 시 break
    for s in summaries:
        if s.get("isImportant"):
            t = _summary_tokens(s)
            if t > available:
                break
            selected.append(s)
            available -= t

    # (b) recent — 최신부터 역순, 초과 시 break (§7 #5)
    reserved_recent = math.floor(available * S.recent_memory_ratio)
    used_recent = 0
    if S.recent_memory_ratio > 0:
        for s in reversed(_unused(summaries, selected)):
            t = _summary_tokens(s)
            if t + used_recent > reserved_recent:
                break
            selected.append(s)
            used_recent += t

    # (c) similar
    random_ratio = 1 - S.recent_memory_ratio - S.similar_memory_ratio
    reserved_similar = math.floor(available * S.similar_memory_ratio)
    used_similar = 0
    if S.similar_memory_ratio > 0:
        if random_ratio <= 0:          # recent 잔여 흡수 (§7 #7)
            reserved_similar += reserved_recent - used_recent
        unused = _unused(summaries, selected)

        chunks: List[Dict[str, Any]] = []
        for si, s in enumerate(unused):
            for ci, chunk in enumerate(
                    split_by_separator(s["text"], S.summary_chunk_separator)):
                if chunk.strip():
                    chunks.append({"id": f"{si}-{ci}", "content": chunk.strip(),
                                   "meta": s})

        # 쿼리: 최근 queryChatCount개 챗을 문단 단위 서브쿼리로 (hypav3.ts:676-695)
        valid = [c for c in recent_chats[-S.query_chat_count:] if c["content"].strip()]
        n = len(valid)
        queries: List[Dict[str, Any]] = []
        for idx, c in enumerate(valid):
            subs = [x for x in c["content"].split(SUMMARY_SEP) if x.strip()]
            if not subs:
                continue
            w = (idx + 1) / (n * (n + 1) / 2) / len(subs)
            queries += [{"content": x, "weight": w} for x in subs]

        if queries and chunks:
            ef = embed or _DEFAULT_EMBED
            chunk_vecs = ef([ch["content"] for ch in chunks])
            query_vecs = ef([q["content"] for q in queries])
            scored_lists = [
                sorted(((ch, _cosine(qv, cv)) for ch, cv in zip(chunks, chunk_vecs)),
                       key=lambda pair: -pair[1])
                for qv in query_vecs
            ]
            ranked = simple_cc(scored_lists, [q["weight"] for q in queries])
            for s in child_to_parent_rrf(ranked, key=lambda ch: ch["meta"]):
                t = _summary_tokens(s)
                if t + used_similar > reserved_similar:
                    break              # ★ break (§7 #5)
                selected.append(s)
                used_similar += t

    # (d) random — 고정 export는 ratio 0.0이라 도달 불가 (hypav3.ts:806-869)
    if random_ratio > 0:
        raise HypaError("random 선택은 고정 export(random_ratio=0)에서 도달 불가 "
                        "— 스펙 가정 붕괴")

    # 시간순 재정렬 (§7 #8)
    return sorted(selected, key=lambda s: _index_of(summaries, s))


# --- 오케스트레이터 (hypav3.ts:188-935, §2) ---

MEMORY_TAG = "Past Events Summary"          # hypav3.ts:102


def wrap_xml(tag: str, content: str) -> str:
    """hypav3.ts:1673-1675."""
    return f"<{tag}>\n{content}\n</{tag}>"


def hypa_step(
    history: List[Dict[str, Any]],
    preset_tokens: int,
    S: HypaSettings,
    data: Dict[str, Any],
    send: Optional[Callable[[List[Dict[str, str]]], str]],
    max_ctx: int,
    max_response: int,
) -> Tuple[Optional[str], List[Dict[str, Any]], int, Dict[str, Any], Optional[str]]:
    """한 턴 분의 HypaV3 — 요약 생성(쓰기)과 선택(읽기)이 같은 패스에서 돈다.

    반환 (memory_text, kept_history, kept_start_msg, data, error).
    memory_text는 요약이 0개면 None (§7 #9 — 그 턴엔 memory 카드가 비어 있다).
    kept_start_msg는 `history` 기준 slice 시작 인덱스 — 그 앞은 나레이터가
    아예 못 본다 (in_window 판정용).
    """
    chats = to_risu_chats(history)
    summaries: List[Dict[str, Any]] = data.setdefault("summaries", [])

    # 초기 토큰 (index.svelte.ts:614-618) → maxResponse 되돌려 빼기 (§7 #1)
    current = max_response + 50 + preset_tokens + sum(tok_chat(c) for c in chats)
    current -= max_response

    # startIdx — 마지막 요약의 마지막 memo 다음부터 (hypav3.ts:212-229).
    # preserveOrphanedMemory=true(고정 export)라 orphan 정리는 없다.
    start_idx = 0
    if summaries:
        last_memo = summaries[-1]["chatMemos"][-1]
        idx = next((i for i, c in enumerate(chats) if c["memo"] == last_memo), -1)
        if idx != -1:
            start_idx = idx + 1
            current -= sum(tok_chat(c) for c in chats[:start_idx])

    # 메모리 예산 예약 — 요약이 하나라도 있으면 매 턴 선점된다 (§7 #11).
    empty_mem_tokens = tok_chat({"role": "system",
                                 "content": wrap_xml(MEMORY_TAG, "")})
    memory_tokens = math.floor(max_ctx * S.memory_tokens_ratio)
    should_reserve = bool(summaries) or current > max_ctx
    available = memory_tokens - empty_mem_tokens if should_reserve else 0
    if should_reserve:
        current += memory_tokens

    batches, current, start_idx, error = plan_batches(
        chats, start_idx, current, max_ctx, S)
    if error:
        return None, history[start_idx:], start_idx, data, error

    for batch in batches:
        summaries.append({
            "text": summarize(send, batch, S),
            # 원본은 Set — 중복 제거 + 삽입순 보존 (hypav3.ts:460)
            "chatMemos": list(dict.fromkeys(c["memo"] for c in batch)),
            "isImportant": False})

    if not summaries:                       # 요약 0개면 미삽입 (§7 #9)
        return None, history[start_idx:], start_idx, data, None

    selected = select_summaries(summaries, available, chats, S)
    memory = wrap_xml(MEMORY_TAG, SUMMARY_SEP.join(s["text"] for s in selected))

    # 의도적 생략: §2 227-229 최종 토큰 정산
    #   (current -= memory_tokens; current += real_mem_tokens;
    #    current > max_ctx면 throw)은 여기 없다. 이 함수는 available
    #    (= memory_tokens 상한, 위 should_reserve 블록) 자체를 select_summaries
    #    의 예산으로 넘기고 그 안에서 그리디하게 채우므로 real_mem_tokens는
    #    available을 절대 넘지 못한다 — throw는 도달 불가능한 방어 코드다.
    #    또한 이 함수는 current_tokens를 호출부로 반환하지 않는다(반환 튜플에
    #    자리가 없다) — 다음 턴은 hypa_step을 처음부터 다시 불러 current를
    #    재계산하므로 정산값을 들고 있을 이유가 없다. 버그 아님.
    # §2 231-236 대체: select_summaries는 4단계 결과를 태그 없이 병합·시간순
    # 정렬해 반환하므로 단계별 개수는 여기서 복원 불가 — 대신 이번 턴 실제로
    # 선택된 요약 개수와 그 chatMemos를 남긴다. "요약 갱신 없는 턴에도 similar
    # 선택이 흔들려 캐시가 깨지는가"는 이 값의 턴간 변화로 사후 확인한다.
    data["metrics"] = {
        "selectedCount": len(selected),
        "selectedChatMemos": [memo for s in selected
                               for memo in s.get("chatMemos", [])],
    }
    return memory, history[start_idx:], start_idx, data, None
