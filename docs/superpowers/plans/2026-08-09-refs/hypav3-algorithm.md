# summary
HypaV3는 "요약 생성(쓰기)"과 "요약 선택(읽기)"이 매 턴 같은 함수에서 동기적으로 돌아가는 단일 패스 알고리즘이다. 발동은 순수 토큰 임계 — `currentTokens > maxContextTokens`(= db.maxContext)일 때만 요약 루프가 돌고, 목표는 `maxContext * (1 - extraSummarizationRatio)`까지 내려가는 것이다. 뮈토스 하이파 V5 export는 `useExperimentalImpl: true`이므로 실행 경로는 `hypaMemoryV3MainExp`(hypav3.ts:176) 하나뿐이고, 배칭은 `maxChatsPerSummary: 8`, 메모리 예산은 `memoryTokensRatio: 0.39 × maxContext`(뮈토스 6.2 기준 200000 → 78000토큰), 선택은 important → recent(0.6) → similar(0.4) → random(0.0, 스킵) 순의 그리디 토큰 채우기다. 요약 프롬프트는 `<|im_start|>`로 시작하지 않고 `{{slot}}`도 없어서 parseChatML이 null을 반환 → **[user=원문 chat 라인들, system=19,405자 커스텀 프롬프트]** 2-메시지 폴백으로 나가고, 모델은 `db.subModel`(뮈토스 6.2에서 Gemini 3 Flash Preview)이다. 유사도는 db.hypaModel(기본 MiniLM, transformers.js 로컬 WASM, mean-pool + L2정규화)로 요약 청크를 임베딩한 뒤 최근 3챗을 문단 단위 서브쿼리로 쪼개 가중 CC → child-to-parent RRF로 랭킹한다. 주입은 `{role:'system', content:'<Past Events Summary>…</Past Events Summary>', memo:'supaMemory'}` 단일 메시지를 chats 배열 맨 앞에 붙이고, 뮈토스 6.2 프리셋에는 `memory` 카드(index 35, "Past Summary")가 있으므로 이 메시지는 chat 블록에서 **떼어져 시스템 프롬프트 중간(전체 49카드 중 35번)** 으로 재배치된다 — 즉 프리픽스의 한가운데가 매 턴 바뀔 수 있는 구조이며, 요약이 하나라도 새로 추가되거나 similar/random 선택이 흔들리면 그 뒤 전부(카드 36~48 + 전체 chat 히스토리)의 캐시가 깨진다. 원본 챗은 삭제가 아니라 `chats.slice(startIdx)`로 잘려나가고, greeting(first message)은 memo 없이 chats에 들어가 요약 대상에 포함된다.

# spec
# HypaV3 재현 스펙 (Python 벤치 하네스용)

> 근거 파일 축약: `HV3` = `external/risuai/src/ts/process/memory/hypav3.ts`, `IDX` = `external/risuai/src/ts/process/index.svelte.ts`, `HM2` = `external/risuai/src/ts/process/memory/hypamemoryv2.ts`, `TF` = `external/risuai/src/ts/process/transformers.ts`.
> 대상 경로는 **exp 구현 하나만** (`useExperimentalImpl: true`, HV3:129). legacy는 재현 불필요.

---

## 0. 입출력 계약

```
def hypa_v3(chats, current_tokens, max_context_tokens, hypa_data, settings, db, tokenizer)
    -> (new_chats, new_current_tokens, new_hypa_data, error|None)
```

- `chats`: `[{role, content, memo, name}]` — RisuAI가 만든 **평탄화된 대화 배열**.
  구성 순서 (IDX:831-884, 901-1053):
  1. `exampleMessage()` 결과 (`name='example_user'|'example_assistant'`, `memo=None`; `<start>` 줄은 `memo='NewChatExample'`)
  2. `{role:'system', content:'[Start a new chat]', memo:'NewChat'}` (단, `promptSettings.trimStartNewChat`가 true면 생략)
  3. **greeting** `{role:'assistant', content:first_message, memo:None}` (그룹챗 아니고 allBefore 리셋 아닐 때)
  4. 실제 메시지들 `{role:'user'|'assistant', content, memo=chatId}`
- `current_tokens`: `db.maxResponse + 50 + Σ(템플릿 카드 토큰) + Σ(lorebook) + Σ(chats 토큰)` (IDX:614-618, 682, 1051)
- `max_context_tokens` = `db.maxContext` (IDX:345). **여기서 maxResponse를 빼지 않는다.**
- `hypa_data`: `{"summaries": [{"text": str, "chatMemos": [str|None], "isImportant": bool, "categoryId": str|None, "tags": [str]}], "metrics": {...}}`

`tokenizer.tokenize_chat(chat)` = `len(encode(content)) + chat_additional_tokens` (+ name이 있고 useName='name'이면 `len(encode(name)) + 1`).
`chat_additional_tokens` = 5 if aiModel.startswith('gpt') else 3 (IDX:287-293).
`useName` = 'noName' if gpt else 'name' (IDX:342).

---

## 1. 설정 매핑 (export → 코드 필드)

`hypaV3_export_뮈토스 하이파 V5.json`의 `.data.settings` → `HypaV3Settings` (HV3:29-51).
import 시 `createHypaV3Preset(name, settings)`가 **키 존재 + typeof 일치**일 때만 채택 (HV3:1814-1824).

| export 필드 | 값 | 코드에서 쓰이는 곳 |
|---|---|---|
| `summarizationModel` | `"subModel"` | HV3:1709 → `db.subModel` |
| `summarizationPrompt` | 19,405자 | HV3:1691-1706 |
| `reSummarizationPrompt` | `""` | HV3:1690 (자동 파이프라인 미사용) |
| `memoryTokensRatio` | `0.39` | HV3:238-240 → `floor(maxContext * 0.39)` |
| `extraSummarizationRatio` | `0.01` | HV3:254 → `targetTokens = maxContext * 0.99` |
| `maxChatsPerSummary` | `8` | HV3:294 |
| `recentMemoryRatio` | `0.6` | HV3:540-542 |
| `similarMemoryRatio` | `0.4` | HV3:588-590 |
| `enableSimilarityCorrection` | `false` | **exp 경로 미사용** (HV3:1371만) |
| `preserveOrphanedMemory` | `true` | HV3:207 → cleanOrphaned 스킵 |
| `processRegexScript` | `true` | **모달 전용**, 파이프라인 미사용 |
| `doNotSummarizeUserMessage` | `false` | HV3:339 |
| `useExperimentalImpl` | `true` | HV3:129 |
| `summarizationRequestsPerMinute` | `20` | HV3:385-387 |
| `summarizationMaxConcurrent` | `2` | HV3:389-391 |
| `embeddingRequestsPerMinute` | `100` | HV3:630-633 |
| `embeddingMaxConcurrent` | `3` | HV3:630-633 |
| `alwaysToggleOn` | `false` | stores.svelte.ts:189 |
| `queryChatCount` | `3` | HV3:263, 295, 677 |
| *(누락)* `summaryChunkSeparator` | → 기본 `"\\n\\n"` | HV3:1803, 105-116 |

파생값: `random_memory_ratio = 1 - 0.6 - 0.4 = 0.0` (float64 정확히 0) → **random 블록 스킵**, similar가 recent 잔여 흡수 (HV3:502, 594-607, 813).

---

## 2. 메인 알고리즘 의사코드

```python
MEMORY_TAG = "Past Events Summary"          # HV3:102
SUMMARY_SEP = "\n\n"                        # HV3:103

def wrap_xml(tag, content):                 # HV3:1673-1675
    return f"<{tag}>\n{content}\n</{tag}>"

def hypa_v3(chats, current_tokens, max_ctx, data, S, db, tok):
    # --- 검증 (HV3:188-194) ---
    if S.recentMemoryRatio + S.similarMemoryRatio > 1:
        return error("sum of Recent and Similar ratio > 1")

    # --- 초기 토큰 보정 (HV3:197) ---
    current_tokens -= db.maxResponse

    # --- orphan 정리 (HV3:207-209, 1646-1661) ---
    if not S.preserveOrphanedMemory:
        live = {c.memo for c in chats}
        data.summaries = [s for s in data.summaries if set(s.chatMemos) <= live]

    # --- startIdx 결정 (HV3:212-229) ---
    start_idx = 0
    if data.summaries:
        last_memo = data.summaries[-1].chatMemos[-1]           # Set 삽입순 == list 마지막
        idx = next((i for i,c in enumerate(chats) if c.memo == last_memo), -1)
        if idx != -1:
            start_idx = idx + 1
            for c in chats[:idx+1]:
                current_tokens -= tok(c)

    # --- 메모리 예산 예약 (HV3:234-250) ---
    empty_mem_tokens = tok({"role":"system", "content": wrap_xml(MEMORY_TAG, "")})
    memory_tokens = floor(max_ctx * S.memoryTokensRatio)
    should_reserve = (len(data.summaries) > 0) or (current_tokens > max_ctx)
    available = (memory_tokens - empty_mem_tokens) if should_reserve else 0
    if should_reserve:
        current_tokens += memory_tokens

    # --- 요약 배치 계획 (HV3:253-378) ---
    summarization_mode = current_tokens > max_ctx
    target_tokens = max_ctx * (1 - S.extraSummarizationRatio)
    to_summarize_array = []

    while summarization_mode:                    # 값은 루프 중 안 바뀜 → if + while True와 동치
        if current_tokens <= target_tokens: break
        if len(chats) - start_idx <= S.queryChatCount:
            if current_tokens <= max_ctx: break
            return error(f"Cannot summarize further: {current_tokens} > {max_ctx}, "
                         f"but minimum {S.queryChatCount} messages required.")   # 요청 자체가 실패

        batch, batch_tokens, i = [], 0, start_idx
        while len(batch) < S.maxChatsPerSummary and i < len(chats) - S.queryChatCount:
            c = chats[i]
            batch_tokens += tok(c)               # ★ 스킵돼도 누적 (HV3:313 — 원본 버그 그대로 재현)
            skip = (c.name in ("example_user","example_assistant")
                    or c.memo in ("NewChatExample","NewChat")
                    or c.content.strip() == ""
                    or (S.doNotSummarizeUserMessage and c.role == "user"))
            if not skip: batch.append(c)
            i += 1

        # 과요약 방지 (HV3:352-362)
        if current_tokens <= max_ctx and current_tokens - batch_tokens < target_tokens:
            break

        if batch: to_summarize_array.append(list(batch))
        current_tokens -= batch_tokens
        start_idx = i

    # --- 요약 실행 (HV3:381-466) ---
    # 원본은 rate-limited 병렬. 결과는 배치 순서대로 append.
    for k, batch in enumerate(to_summarize_array):
        text = summarize(batch, S, db)           # 실패하면 즉시 error 반환 + 지금까지의 data 저장
        data.summaries.append({
            "text": text,
            "chatMemos": [c.memo for c in batch],   # 원본은 Set (중복 제거 + 삽입순 보존)
            "isImportant": False, "categoryId": None, "tags": [],
        })

    # --- 요약 0개면 memory 메시지 없이 조기 반환 (HV3:480-499) ---
    if not data.summaries:
        return chats[start_idx:], current_tokens, data, None

    # --- 선택 4단계 ---
    selected, sel_important, sel_recent, sel_similar, sel_random = [], [], [], [], []
    random_ratio = 1 - S.recentMemoryRatio - S.similarMemoryRatio

    # (a) important (HV3:507-537)
    for s in data.summaries:
        if s.isImportant:
            t = tok({"role":"system", "content": s.text + SUMMARY_SEP})
            if t > available: break              # ★ break (더 작은 것 안 찾음)
            sel_important.append(s); available -= t
    selected += sel_important

    # (b) recent (HV3:540-585) — 최신부터 역순, 초과 시 break
    reserved_recent = floor(available * S.recentMemoryRatio)
    used_recent = 0
    if S.recentMemoryRatio > 0:
        unused = [s for s in data.summaries if s not in selected]
        for s in reversed(unused):
            t = tok({"role":"system", "content": s.text + SUMMARY_SEP})
            if t + used_recent > reserved_recent: break
            sel_recent.append(s); used_recent += t
    selected += sel_recent

    # (c) similar (HV3:588-804)
    reserved_similar = floor(available * S.similarMemoryRatio)
    used_similar = 0
    if S.similarMemoryRatio > 0:
        if random_ratio <= 0:                                    # HV3:596-607
            reserved_similar += (reserved_recent - used_recent)
        unused = [s for s in data.summaries if s not in selected]

        # 청크화 (HV3:615-626)
        ebd_texts = []
        for si, s in enumerate(unused):
            for ci, chunk in enumerate(split_by_separator(s.text, S.summaryChunkSeparator)):
                if chunk.strip():
                    ebd_texts.append({"id": f"{si}-{ci}", "content": chunk.strip(), "meta": s})

        # 쿼리 구성 (HV3:676-695)
        recent_chats = [c for c in chats[-S.queryChatCount:] if c.content.strip()]
        n = len(recent_chats)
        queries = []
        for idx, c in enumerate(recent_chats):
            subs = [x for x in c.content.split("\n\n") if x.strip()]
            w = (idx + 1) / (n * (n + 1) / 2) / len(subs)
            queries += [{"content": x, "weight": w} for x in subs]

        if queries:
            # 임베딩 + 코사인 (HM2:62-103, 357-369)
            scored_lists = [ rank_by_cosine(q["content"], ebd_texts) for q in queries ]
            ranked_chunks   = simple_cc(scored_lists, [q["weight"] for q in queries])
            ranked_summaries = child_to_parent_rrf(ranked_chunks, key=lambda ch: ch["meta"], k=60)
            for s in ranked_summaries:                            # HV3:751-772
                t = tok({"role":"system", "content": s.text + SUMMARY_SEP})
                if t + used_similar > reserved_similar: break     # ★ break
                sel_similar.append(s); used_similar += t
            selected += sel_similar

    # (d) random (HV3:806-869) — 뮈토스 설정에선 random_ratio == 0.0 → 스킵
    reserved_random = floor(available * random_ratio)
    used_random = 0
    if random_ratio > 0:
        reserved_random += (reserved_recent - used_recent) + (reserved_similar - used_similar)
        unused = shuffle([s for s in data.summaries if s not in selected])
        for s in unused:
            t = tok({"role":"system", "content": s.text + SUMMARY_SEP})
            if t + used_random > reserved_random: continue        # ★ continue (더 작은 것 계속 시도)
            sel_random.append(s); used_random += t
        selected += sel_random

    # --- 시간순 재정렬 + 조립 (HV3:871-935) ---
    selected.sort(key=lambda s: data.summaries.index(s))
    memory = wrap_xml(MEMORY_TAG, SUMMARY_SEP.join(s.text for s in selected))
    real_mem_tokens = tok({"role":"system", "content": memory})

    if should_reserve: current_tokens -= memory_tokens
    current_tokens += real_mem_tokens
    if current_tokens > max_ctx: raise RuntimeError("Unexpected error: ...")   # HV3:906-910

    data.metrics = {
        "lastImportantSummaries": [data.summaries.index(s) for s in sel_important],
        "lastRecentSummaries":    [data.summaries.index(s) for s in sel_recent],
        "lastSimilarSummaries":   [data.summaries.index(s) for s in sel_similar],
        "lastRandomSummaries":    [data.summaries.index(s) for s in sel_random],
    }

    new_chats = [{"role":"system", "content": memory, "memo":"supaMemory"}] + chats[start_idx:]
    return new_chats, current_tokens, data, None
```

### 보조 함수

```python
def split_by_separator(text, sep):                 # HV3:105-116
    m = re.match(r'^/(.+)/([gimuy]*)$', sep)
    try:
        if m: return re.split(m.group(1), text)
        return re.split(sep_to_regex(sep), text)   # "\\n\\n" → 정규식 \n\n → 빈 줄 기준 분할
    except re.error:
        return text.split("\n\n")

def simple_cc(scored_lists, weights):              # HV3:1832-1852
    scores = defaultdict(float)
    for lst, w in zip(scored_lists, weights):
        for item, sc in lst: scores[item] += sc * w
    return [it for it, _ in sorted(scores.items(), key=lambda kv: -kv[1])]

def child_to_parent_rrf(ranked_children, key, k=60):   # HV3:1874-1893
    scores = defaultdict(float)
    for rank, child in enumerate(ranked_children, start=1):
        scores[key(child)] += 1.0 / (k + rank)
    return [p for p, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
```

> 주의: `simple_cc` / `child_to_parent_rrf`는 JS `Map` 삽입순 + `Array.sort` (V8 TimSort, **stable**)에 의존한다. Python `sorted`도 stable이므로 동점 시 첫 등장 순서가 유지되도록 dict 삽입 순서를 그대로 쓰면 동작이 일치한다.

---

## 3. `summarize()` 재현 (HV3:1681-1773)

```python
def summarize(batch, S, db):
    str_messages = "\n".join(f"{c.role}: {strip_inlay(c.content)}" for c in batch)   # HV3:1685-1687
    prompt = S.summarizationPrompt if S.summarizationPrompt.strip() else \
             "[Summarize the ongoing role story, It must also remove redundancy and unnecessary text and content from the output.]"

    filled = prompt.replace("{{slot}}", str_messages)          # 뮈토스 프롬프트엔 {{slot}} 없음 → 무변화
    parsed = parse_chatml(filled)                              # '<|im_start|>' 로 시작 안 하면 None
    formated = parsed if parsed is not None else [
        {"role": "user",   "content": str_messages},           # ← 뮈토스는 항상 이 경로
        {"role": "system", "content": prompt},                 # ← system이 뒤!
    ]

    resp = llm_call(model=db.subModel, messages=formated,
                    max_tokens=db.maxResponse,                 # 명시 안 함 → db.maxResponse (뮈토스 30000)
                    temperature=db.temperature/100,
                    stream=False)
    if not resp.strip(): raise Error("Empty summary returned")
    out = re.sub(r"<Thoughts>[\s\S]*?</Thoughts>", "", resp).strip()   # HV3:1735-1736
    if not out: raise Error("Empty summary after removing thoughts content")
    return out
```

**뮈토스 하이파 V5 확정 사실** (`hypaV3_export_뮈토스 하이파 V5.json` 실측):
`summarizationPrompt`는 `<hypa_memory_extraction>`으로 시작, `<|im_start|>` 0회, `{{slot}}` 0회
→ `parse_chatml` = None → **[user=chat 원문, system=19,405자 프롬프트] 2메시지**로 고정.

모델: `settings.summarizationModel == "subModel"` → `requestChatData(..., "memory")` → `db.subModel`
(request.ts:440-447; `db.seperateModelsForAxModels`가 true이고 `db.seperateModels["memory"]`가 비어있지 않으면 그쪽 우선).
뮈토스 6.2 DeepSeek 프리셋: `subModel = "pluginmodel:::🧁 [VertexAI] Gemini 3 Flash Preview"`, `seperateModelsForAxModels = False`.

---

## 4. 와이어 주입 재현 (IDX:1169-1186, 1383-1444)

HypaV3가 반환한 `new_chats[0]`(memo='supaMemory')은 **프리셋에 `memory` 카드가 있느냐**로 위치가 갈린다.

```python
memories = []
chat_block = []
for v in new_chats:
    if v.memo not in ("supaMemory", "hypaMemory"):
        v.removable = True; chat_block.append(v)
    elif supa_memory_card_used:                    # 프리셋에 type='memory' 카드가 존재
        memories.append(v)                         # chat 블록에서 제거 (빈 system → filter로 소멸)
    else:
        v.content = f"<Previous Conversation>{v.content}</Previous Conversation>"
        chat_block.append(v)
chat_block = [v for v in chat_block if v.content.strip()]

# 이후 promptTemplate 카드 순서대로 조립:
#   type='chat'   → chat_block[rangeStart:rangeEnd] 를 push
#   type='memory' → safe_clone(memories); innerFormat 있으면 content = innerFormat.replace('{{slot}}', content, 1)
```

**뮈토스 6.2 (DeepSeek) 프리셋 실측** — promptTemplate 49카드:
- `[35] type='memory'` name=`Past Summary`, role=`system`, id=`mythos_v62_past_summary`, innerFormat 있음(`{{slot}}` 포함)
- `[37] type='chat'` rangeStart=0, rangeEnd=-2
- `[39] type='chat'` rangeStart=-2, rangeEnd='end'
- `type='cache'` 카드 **0개**, `automaticCachePoint` 미설정

→ `supa_memory_card_used = True`. 요약 블록은 **chat 히스토리보다 앞, 시스템 프롬프트 35/49 지점**에 앉는다.

---

## 5. 캐시 함의 (판정)

| 관측 | 근거 |
|---|---|
| memory 메시지가 프롬프트 중간(35/49)에 위치 | 뮈토스 promptTemplate[35] + IDX:1429-1443 |
| memory content는 매 턴 재계산 (recent는 요약 추가 시, similar는 최근 3챗 쿼리 변화 시) | HV3:676-695 → 741-772 → 877-880 |
| 원본 챗은 `chats.slice(start_idx)`로 앞부분이 잘려나감 | HV3:934 |
| cache 카드 없음, automaticCachePoint는 있어도 마지막 user 3개에만 | IDX:1413-1426, 815-817 |

**판정: 프리픽스 캐시는 요약이 갱신되는 턴마다 카드 35 이후 전부(카드 36~48 + 전체 히스토리)가 무효화된다.**
요약이 갱신되지 않는 턴이라도 similar 선택은 최근 3챗 임베딩 쿼리 결과라 흔들릴 수 있어, 캐시 유지가 **보장되지 않는다.**
(SAGA가 마지막 user에 prepend하는 방식과 정확히 반대되는 안티패턴.)

---

## 6. 임베딩 — 결정론 대체안

원본 (기본값 `db.hypaModel='MiniLM'`):
`Xenova/all-MiniLM-L6-v2`, transformers.js, dtype **q8**, device wasm/webgpu, `pooling:'mean', normalize:true` (TF:61-93).
유사도는 exp 경로에서 **코사인** (HM2:357-369). 정규화 벡터라 legacy의 내적과 값이 같다.

### 대체안 A (권장, 충실도 최상)
Python `sentence-transformers/all-MiniLM-L6-v2` fp32, mean pooling + L2 정규화, 코사인.
- **차이**: 원본은 q8 양자화 ONNX → 코사인 값이 소수 3~4째 자리에서 갈릴 수 있다. RRF는 절대값이 아니라 **순위**를 쓰므로 상위권 순위는 거의 동일하지만, 동점·근접 구간에서 순서가 뒤바뀔 수 있다.
- **차이**: `pipeline`의 기본 truncation(512 토큰)을 그대로 켤 것. 청크가 512 토큰을 넘으면 뒤가 잘린다.
- **차이**: 브라우저 localforage 임베딩 캐시(HM2:134-172, 371-384)는 재현 불필요(결과 동일).

### 대체안 B (모델 없이 완전 결정론)
TF-IDF(char 3-gram + word) 코사인 또는 BM25.
- **차이 명시 필수**: 의미 검색 → 어휘 검색으로 바뀐다. 한국어 RP에서 동의어/대명사 지시("그 여자", "당채련")를 못 잇는다. similar 슬롯의 히트율이 체계적으로 **낮게** 나오므로, HypaV3 대비 베이스라인 비교에서 **HypaV3에 불리하게** 작동한다 → 리포트에 "retrieval backend 차이" 태그를 반드시 달 것.

### 대체안 C (API 기반 재현)
`db.hypaModel`을 `openai3small`로 놓고 `text-embedding-3-small`을 호출 (HM2:456-475).
- 원본 코드에 실재하는 경로이므로 "설정이 다른 실제 HypaV3"로서 정당하다. 다만 커뮤니티 기본값(MiniLM)과는 다른 조건임을 명시.

**공통 결정론 조치**
- random 슬롯은 뮈토스 설정(`random_ratio == 0.0`)에서 실행되지 않으므로 시드 문제 없음. 다른 설정을 테스트할 땐 `Math.random()` 셔플을 고정 시드 셔플로 대체하고 그 사실을 기록.
- 요약 LLM 호출은 temperature 고정 + 요약 결과를 디스크 캐시하여 리런 간 동일 요약을 재사용.

---

## 7. 재현 체크리스트 (실수하기 쉬운 지점)

1. `current_tokens -= db.maxResponse` 를 빼먹지 말 것 (HV3:197). 안 빼면 요약이 과도하게 조기 발동한다.
2. 스킵된 챗의 토큰도 `batch_tokens`에 더한다 (HV3:313). "버그지만 원본 동작".
3. `maxChatsPerSummary`는 **담긴 개수** 상한이지 스캔 범위가 아니다 — 스킵이 많으면 `currentIndex`가 8칸보다 더 나간다 (HV3:293-296). legacy(1059-1062)와 다르니 혼동 금지.
4. 마지막 `queryChatCount`(3)개는 절대 요약되지 않고 항상 원문으로 남는다.
5. `important`/`recent`/`similar`는 예산 초과 시 **break**, `random`만 **continue**.
6. `available`은 important 단계에서만 감소한다. recent/similar/random 예산은 **important 차감 후의 available**에 비율을 곱한 값이다.
7. `random_ratio <= 0`일 때만 similar가 recent 잔여를 흡수한다 (HV3:596). `> 0`이면 잔여는 random으로 간다.
8. 최종 선택은 반드시 `data.summaries` 인덱스 기준 **시간순 재정렬** (HV3:871-874). 검색 순위 순으로 붙이면 안 된다.
9. 요약 0개면 memory 메시지를 **삽입하지 않는다** (exp 경로, HV3:480-499). legacy와 다르다.
10. greeting은 `memo=None`으로 chats에 들어가 요약 대상에 포함된다 (IDX:868-884). 요약 후 `chatMemos`에 `None`이 들어가고, 다음 턴 `start_idx` 매칭이 `memo is None`인 **첫** 챗을 찾으므로 example 메시지가 있으면 오매칭한다 — 원본 버그. 벤치 카드에 exampleMessage가 없으면 발동하지 않는다.
11. `memoryTokensRatio`는 `maxContext`에 곱해진다 — 뮈토스 6.2 기준 **78,000토큰**이 상시 선점된다. 실제 채워지는 양(realMemoryTokens)과는 별개로, 예약 단계에서 이 값이 요약 발동을 스스로 유발한다는 점이 핵심 병리다.

# claims
- [진입점/구현 분기] hypaMemoryV3는 settings.useExperimentalImpl로 두 구현 중 하나를 고른다. 뮈토스 export는 useExperimentalImpl=true이므로 실제 실행 경로는 hypaMemoryV3MainExp 하나뿐이며, hypaMemoryV3Main(legacy, 955~1618줄)은 이 설정에서 절대 실행되지 않는다.
  근거: external/risuai/src/ts/process/memory/hypav3.ts:129-149 `if (settings.useExperimentalImpl) { ... return await hypaMemoryV3MainExp(...) } return await hypaMemoryV3Main(...)` / export JSON `.data.settings.useExperimentalImpl = True`
- [1. 발동 조건 — maxContext] maxContextTokens는 DBState.db.maxContext를 그대로 전달한다. 별도 마진 차감 없음. 뮈토스 6.2 DeepSeek 프리셋에서 maxContext=200000, maxResponse=30000.
  근거: external/risuai/src/ts/process/index.svelte.ts:345 `let maxContextTokens = DBState.db.maxContext` / :1104 `await hypaMemoryV3(chats, currentTokens, maxContextTokens, currentChat, nowChatroom, tokenizer)`
- [1. 발동 조건 — currentTokens 구성] currentTokens는 db.maxResponse + 50에서 시작해 promptTemplate 카드 토큰 + lorebook + 전체 chats 토큰을 누적한 값이다. memory 카드는 사전 패스에서 토큰을 전혀 더하지 않고 supaMemoryCardUsed 플래그만 세운다.
  근거: external/risuai/src/ts/process/index.svelte.ts:614-618 `let currentTokens = DBState.db.maxResponse` … `currentTokens += 50`; :811-813 `case 'memory':{ supaMemoryCardUsed = true; break }`
- [1. 발동 조건 — 실제 임계식] HypaV3는 먼저 currentTokens에서 db.maxResponse를 되돌려 빼고, 메모리 예산(memoryTokens)을 더한 뒤 `currentTokens > maxContextTokens`이면 요약 모드에 들어간다. 즉 임계는 (프롬프트+히스토리+50+memoryTokens) > maxContext.
  근거: hypav3.ts:197 `currentTokens -= db.maxResponse;` / :247-250 `if (shouldReserveMemoryTokens) { currentTokens += memoryTokens; }` / :253 `const summarizationMode = currentTokens > maxContextTokens;`
- [1. 발동 조건 — 목표치] 요약 루프는 currentTokens가 targetTokens = maxContextTokens * (1 - extraSummarizationRatio) 이하로 내려갈 때까지 돈다. export의 extraSummarizationRatio=0.01 → 200000*0.99 = 198000.
  근거: hypav3.ts:254-255 `const targetTokens = maxContextTokens * (1 - settings.extraSummarizationRatio);` / :258-261 `while (summarizationMode) { if (currentTokens <= targetTokens) break; }` / export `.data.settings.extraSummarizationRatio = 0.01`
- [1. 발동 조건 — 메모리 예약 게이트] 메모리 토큰 예약 여부는 `data.summaries.length > 0 || currentTokens > maxContextTokens`. 즉 요약이 하나라도 쌓이면 그 이후 매 턴 78000토큰(0.39×200000)이 선점되고, 그 결과 currentTokens가 maxContext를 넘겨 요약 모드가 계속 유지되는 자기유지 루프가 된다.
  근거: hypav3.ts:238-250 `const memoryTokens = Math.floor(maxContextTokens * settings.memoryTokensRatio); const shouldReserveMemoryTokens = data.summaries.length > 0 || currentTokens > maxContextTokens;`
- [1. 발동 조건 — 하한 가드] 남은 미요약 챗 수가 queryChatCount 이하이면 더 요약하지 않는다. 여전히 maxContext 초과면 에러 반환(요청 중단). export queryChatCount=3.
  근거: hypav3.ts:263-274 `if (chats.length - startIdx <= settings.queryChatCount) { if (currentTokens <= maxContextTokens) break; else return { ... error: ... minimum ${settings.queryChatCount} messages required } }`
- [2. 요약 배칭 — 개수 제어 필드] 한 배치에 담는 챗 수는 settings.maxChatsPerSummary가 상한이고, 마지막 queryChatCount개는 절대 배치에 들어가지 않는다. export: maxChatsPerSummary=8, queryChatCount=3.
  근거: hypav3.ts:293-296 `while (toSummarize.length < settings.maxChatsPerSummary && currentIndex < chats.length - settings.queryChatCount)` / export `.maxChatsPerSummary = 8`, `.queryChatCount = 3`
- [2. 요약 배칭 — 스킵 규칙] example_user/example_assistant name, memo 'NewChatExample', memo 'NewChat', 공백 content, (doNotSummarizeUserMessage일 때) user role은 배치에서 제외된다. export의 doNotSummarizeUserMessage=false이므로 user 메시지도 요약된다.
  근거: hypav3.ts:317-342 `if (chat.name === "example_user" || chat.name === "example_assistant" || chat.memo === "NewChatExample") { shouldSummarize = false } ... if (settings.doNotSummarizeUserMessage && chat.role === "user")` / export `.doNotSummarizeUserMessage = False`
- [2. 요약 배칭 — 토큰 회계 함정] 스킵된 챗의 토큰도 toSummarizeTokens에 더해진 뒤 currentTokens에서 차감된다(스킵 판정보다 먼저 누적). 즉 요약 안 된 챗도 '요약된 것처럼' 토큰이 빠지고, startIdx는 스킵 챗까지 넘어간다 → 그 챗들은 영원히 요약되지 않고 컨텍스트에서도 사라진다.
  근거: hypav3.ts:313 `toSummarizeTokens += chatTokens;` (스킵 판정 315-342줄보다 먼저) / :376-377 `currentTokens -= toSummarizeTokens; startIdx = currentIndex;`
- [2. 요약 배칭 — exp vs legacy 차이] exp 경로는 배치 계획을 전부 세운 뒤(toSummarizeArray) TaskRateLimiter로 병렬 요약하고, legacy는 루프 안에서 순차 요약한다. export: summarizationRequestsPerMinute=20, summarizationMaxConcurrent=2 (subModel일 때만 적용; 로컬 모델이면 1000rpm/동시성1로 강제).
  근거: hypav3.ts:384-393 `tasksPerMinute: settings.summarizationModel === "subModel" ? settings.summarizationRequestsPerMinute : 1000, maxConcurrentTasks: ... : 1` / :404-417 `const summarizationTasks = toSummarizeArray.map((item) => () => summarize(item)); ... rateLimiter.executeBatch<string>(summarizationTasks)` / legacy 대비 :1151 `const summarizeResult = await summarize(toSummarize);`
- [3. 요약 프롬프트 — 기본값 vs 커스텀] settings.summarizationPrompt가 trim 후 비어 있을 때만 코드 기본값 `[Summarize the ongoing role story, ...]`을 쓴다. 뮈토스 export는 19,405자 커스텀 프롬프트가 있으므로 코드 기본값은 사용되지 않는다.
  근거: hypav3.ts:1691-1693 `: settings.summarizationPrompt.trim() === "" ? "[Summarize the ongoing role story, It must also remove redundancy and unnecessary text and content from the output.]" : settings.summarizationPrompt;` / export `.summarizationPrompt` 길이 19405
- [3. 요약 프롬프트 — 해석 경로(중요)] 프롬프트는 {{slot}}를 chat 원문으로 치환한 뒤 parseChatML에 넣는다. parseChatML은 문자열이 '<|im_start|>'로 시작하지 않으면 null을 반환한다. 뮈토스 프롬프트는 <hypa_memory_extraction>로 시작하고 {{slot}}/im_start/im_sep/im_end가 0개이므로 → null → 폴백 배열 [ {role:'user', content:원문}, {role:'system', content:프롬프트} ]가 그대로 나간다. system이 user 뒤에 오는 순서다.
  근거: hypav3.ts:1695-1706 `const formated: OpenAIChat[] = parseChatML(summarizationPrompt.replaceAll("{{slot}}", strMessages)) ?? [ { role: "user", content: strMessages }, { role: "system", content: summarizationPrompt } ];` / external/risuai/src/ts/parser/chatML.ts:9-11 `if (!trimedData.startsWith(starter)) { return null }` / export 프롬프트: startswith('<|im_start|>')=False, '{{slot}}' 포함=False
- [3. 요약 프롬프트 — 원문 포맷] 요약 입력 원문은 `${chat.role}: ${content}` 라인을 '\n'으로 이어붙인 문자열이고, inlay 이미지 토큰은 '[Image]'로 치환된다.
  근거: hypav3.ts:1685-1687 `const strMessages = oaiMessages.map((chat) => `${chat.role}: ${sanitizeSummaryContent(chat.content)}`).join("\n");` / :1677-1679 `content.replace(inlayTokenRegex, "[Image]")`
- [3. 요약 프롬프트 — 재요약] reSummarizationPrompt는 isResummarize=true일 때만 쓰이고, 이 플래그는 hypaMemoryV3 자동 파이프라인에서는 절대 true가 되지 않는다(모달 수동 재요약 전용). export의 reSummarizationPrompt=''이라 기본값 'Re-summarize this summaries.'.
  근거: hypav3.ts:1681 `export async function summarize(oaiMessages: OpenAIChat[], isResummarize: boolean = false)` / :1689-1690 `isResummarize ? (settings.reSummarizationPrompt.trim() === "" ? "Re-summarize this summaries." : ...)` / :405 `() => summarize(item)` (2번째 인자 없음)
- [4. 요약 모델] summarizationModel='subModel'이면 requestChatData(..., 'memory')로 나가고, 이는 db.subModel을 쓴다(단 db.seperateModelsForAxModels가 true이고 db.seperateModels['memory']가 비어있지 않으면 그쪽 우선). 뮈토스 6.2 DeepSeek 프리셋: subModel='pluginmodel:::🧁 [VertexAI] Gemini 3 Flash Preview', seperateModelsForAxModels=False → Gemini 3 Flash로 요약.
  근거: hypav3.ts:1709-1720 `if (settings.summarizationModel === "subModel") { ... await requestChatData({ formated, bias:{}, useStreaming:false, noMultiGen:true }, "memory") }` / external/risuai/src/ts/process/request/request.ts:440-447 `targ.aiModel = arg.staticModel ? arg.staticModel : (model === 'model' ? db.aiModel : db.subModel)` … `if(db.seperateModelsForAxModels ...)`
- [4. 요약 모델 — 파라미터] 요약 호출은 maxTokens/temperature를 명시하지 않으므로 db.maxResponse(뮈토스 30000)와 db.temperature/100이 그대로 적용된다. 응답에서 <Thoughts>…</Thoughts>는 제거된다. summarizationModel이 'subModel'이 아니면 WebLLM 로컬(max_tokens 8192, temperature 0)로 가고 system 메시지를 맨 앞으로 끌어올린다.
  근거: request.ts:457-458 `targ.maxTokens = arg.maxTokens ?? db.maxResponse; targ.temperature = arg.temperature ?? (db.temperature / 100)` / hypav3.ts:1735-1736 `const thoughtsRegex = /<Thoughts>[\s\S]*?<\/Thoughts>/g;` / :1746-1758 `const firstSystemIndex = formated.findIndex(m => m.role === 'system'); ... chatCompletion(formated, settings.summarizationModel, { max_tokens: 8192, temperature: 0, ...})`
- [5. 메모리 토큰 예산] 0.39는 settings.memoryTokensRatio이며 maxContextTokens(=db.maxContext)에 곱해진다. memoryTokens = floor(200000 × 0.39) = 78000. 실제 선택 가능 예산 availableMemoryTokens = memoryTokens − emptyMemoryTokens(빈 <Past Events Summary></Past Events Summary> 시스템 메시지 토큰).
  근거: hypav3.ts:234-245 `const emptyMemoryTokens = await tokenizer.tokenizeChat({ role: "system", content: wrapWithXml(memoryPromptTag, "") }); const memoryTokens = Math.floor(maxContextTokens * settings.memoryTokensRatio); let availableMemoryTokens = shouldReserveMemoryTokens ? memoryTokens - emptyMemoryTokens : 0;` / export `.memoryTokensRatio = 0.39`
- [5. 메모리 토큰 예산 — 정산] 선택이 끝나면 예약분(memoryTokens)을 되돌려 빼고 실제 memory 메시지 토큰(realMemoryTokens)을 더한다. 그래도 maxContext를 넘으면 throw.
  근거: hypav3.ts:886-891 `if (shouldReserveMemoryTokens) { currentTokens -= memoryTokens; } currentTokens += realMemoryTokens;` / :906-910 `if (currentTokens > maxContextTokens) { throw new Error(...) }`
- [6. 선택 알고리즘 — 4단계 순서] important(isImportant 플래그) → recent → similar → random 순서로 그리디하게 토큰을 채운다. important는 availableMemoryTokens를 직접 깎고, 나머지 3개는 깎인 뒤의 availableMemoryTokens에 각 비율을 곱해 예산을 나눈다.
  근거: hypav3.ts:507-537 important 블록 `availableMemoryTokens -= summaryTokens;` / :540-542 `const reservedRecentMemoryTokens = Math.floor(availableMemoryTokens * settings.recentMemoryRatio);` / :588-590 `let reservedSimilarMemoryTokens = Math.floor(availableMemoryTokens * settings.similarMemoryRatio);` / :807-809 `let reservedRandomMemoryTokens = Math.floor(availableMemoryTokens * randomMemoryRatio);`
- [6. 선택 알고리즘 — 비율 필드] recentMemoryRatio=0.6, similarMemoryRatio=0.4, randomMemoryRatio는 필드가 아니라 1 − recent − similar로 파생된다. 0.6+0.4에서는 IEEE754상 정확히 0.0이 되어 random 블록은 스킵되고, 대신 recent가 못 쓴 잔여 토큰이 similar 예산에 합산된다.
  근거: hypav3.ts:502-503 `const randomMemoryRatio = 1 - settings.recentMemoryRatio - settings.similarMemoryRatio;` / :594-607 `if (settings.similarMemoryRatio > 0) { if (randomMemoryRatio <= 0) { const unusedRecentTokens = reservedRecentMemoryTokens - consumedRecentMemoryTokens; reservedSimilarMemoryTokens += unusedRecentTokens; } }` / :813 `if (randomMemoryRatio > 0)` / export `.recentMemoryRatio = 0.6`, `.similarMemoryRatio = 0.4`
- [6. 선택 알고리즘 — recent] recent는 아직 선택되지 않은 요약들을 뒤(최신)에서부터 하나씩 담고, 예산 초과 시 즉시 break(더 작은 걸 찾지 않음). random만 break이 아니라 continue를 쓴다.
  근거: hypav3.ts:553-569 `for (let i = unusedSummaries.length - 1; i >= 0; i--) { ... if (summaryTokens + consumedRecentMemoryTokens > reservedRecentMemoryTokens) { break; } ... }` / :843-849 random은 `continue;`
- [6. 선택 알고리즘 — 유사도 청크화] 각 요약을 summaryChunkSeparator로 쪼개 청크 단위로 임베딩한다. export에는 summaryChunkSeparator 필드가 없으므로 import 시 createHypaV3Preset 기본값 '\\n\\n'(리터럴 백슬래시-n 2개)가 들어가고, splitBySeparator가 new RegExp("\\n\\n")로 컴파일해 실제 빈 줄 기준 분할이 된다.
  근거: hypav3.ts:615-626 `const splitted = splitBySeparator(summary.text, settings.summaryChunkSeparator).filter((e) => e.trim().length > 0); return splitted.map((chunk, chunkIndex) => ({ id: `${summaryIndex}-${chunkIndex}`, content: chunk.trim(), metadata: summary }))` / :105-116 splitBySeparator / :1803 `summaryChunkSeparator: "\\n\\n"` / :1819-1823 import 병합 루프 `if (key in settings && typeof value === typeof settings[key])`
- [6. 선택 알고리즘 — 임베딩 프로바이더] 임베딩 모델은 하이파 프리셋이 아니라 전역 db.hypaModel이 결정한다(export에 없음). 기본값은 'MiniLM' = Xenova/all-MiniLM-L6-v2로 브라우저 로컬(transformers.js, q8, wasm) 실행이며 mean pooling + L2 정규화. 'ada'/'openai3small'/'openai3large'는 OpenAI API, 'custom'은 임의 URL, 'voyageContext3'은 Voyage contextualized embeddings API.
  근거: external/risuai/src/ts/process/memory/hypamemoryv2.ts:43 `model: db.hypaModel || "MiniLM"` / external/risuai/src/ts/storage/database.svelte.ts:394 `data.hypaModel ??= 'MiniLM'` / external/risuai/src/ts/process/transformers.ts:82 `let result = await extractor(texts, { pooling: 'mean', normalize: true });` / hypamemoryv2.ts:456-483 OpenAI/custom/voyage 분기
- [6. 선택 알고리즘 — 유사도 함수] exp 경로(HypaProcessorV2)는 코사인 유사도를 쓴다. legacy 경로(hypamemory.ts similarity)는 정규화 없는 순수 내적이다. 로컬 MiniLM은 이미 L2 정규화된 벡터라 두 값이 일치한다.
  근거: hypamemoryv2.ts:357-369 `return dot / (Math.sqrt(magA) * Math.sqrt(magB));` / external/risuai/src/ts/process/memory/hypamemory.ts:232-238 `export function similarity(a,b){ let dot = 0; for(...) dot += a[i]*b[i]; return dot }`
- [6. 선택 알고리즘 — 쿼리 구성] 쿼리는 마지막 queryChatCount(=3)개 챗 중 비어있지 않은 것들을 '\n\n' 기준 서브쿼리로 쪼개 만든다. 각 서브쿼리 가중치 = (index+1) / (n(n+1)/2) / (해당 챗의 서브쿼리 수) — 최신 챗일수록 큰 가중치, 서브쿼리가 많을수록 개별 가중치는 희석.
  근거: hypav3.ts:676-695 `const recentChats = chats.slice(-settings.queryChatCount).filter(...); const queries = recentChats.map((chat, index) => { const subQueries = chat.content.split("\n\n").filter(...); const weight = (index + 1) / ((recentChats.length * (recentChats.length + 1)) / 2) / subQueries.length; ... }).flat();`
- [6. 선택 알고리즘 — 랭킹 융합] 쿼리별 점수 리스트를 simpleCC(가중 점수 합산)로 청크 랭킹을 만들고, 그 순위를 childToParentRRF(k=60, rrf=1/(60+rank))로 부모 요약 단위 점수로 접는다. 즉 한 요약의 여러 청크가 상위에 들면 그 요약 점수가 누적된다.
  근거: hypav3.ts:741-749 `const rankedChunks = simpleCC<EmbeddingResult<Summary>>(batchScoredResults, (listIndex) => queries[listIndex].weight); const rankedSummaries = childToParentRRF<EmbeddingResult<Summary>, Summary>(rankedChunks, (chunk) => chunk.metadata);` / :1832-1852 simpleCC / :1874-1893 childToParentRRF `const rrfTerm = 1 / (k + rank);`
- [6. 선택 알고리즘 — similar 채우기] 랭킹 상위부터 예산 안에 들어가면 담고, 초과하면 즉시 break(뒤의 작은 요약을 더 찾지 않음).
  근거: hypav3.ts:751-772 `while (rankedSummaries.length > 0) { const summary = rankedSummaries.shift(); ... if (summaryTokens + consumedSimilarMemoryTokens > reservedSimilarMemoryTokens) { ... break; } selectedSimilarSummaries.push(summary); }`
- [6. 선택 알고리즘 — enableSimilarityCorrection] 이 옵션은 legacy 경로에만 존재한다(최근 챗을 한 번 더 요약해 추가 쿼리로 씀). exp 경로에는 참조가 아예 없으므로 useExperimentalImpl=true인 뮈토스 설정에서는 사실상 무의미하다. export도 false.
  근거: hypav3.ts:1371 `if (settings.enableSimilarityCorrection && recentChats.length > 1)` (유일한 사용처, legacy 함수 내부) / export `.enableSimilarityCorrection = False`
- [7. 주입 — 메시지 형태] 선택된 요약들은 data.summaries 원래 순서(시간순)로 재정렬된 뒤 '\n\n'로 join되어 <Past Events Summary>\n{내용}\n</Past Events Summary>로 감싸지고, {role:'system', content: memory, memo:'supaMemory'} 단일 메시지로 chats 배열 맨 앞(index 0)에 삽입된다.
  근거: hypav3.ts:871-880 `selectedSummaries.sort((a, b) => data.summaries.indexOf(a) - data.summaries.indexOf(b)); const memory = wrapWithXml(memoryPromptTag, selectedSummaries.map((e) => e.text).join(summarySeparator));` / :928-935 `const newChats: OpenAIChat[] = [ { role: "system", content: memory, memo: "supaMemory" }, ...chats.slice(startIdx) ];` / :101-103 `const memoryPromptTag = "Past Events Summary"; const summarySeparator = "\n\n";` / :1673-1675 wrapWithXml
- [7. 주입 — 원본 챗 제거] 요약된 원본 챗은 삭제되는 게 아니라 chats.slice(startIdx)로 앞부분이 잘려나간다. startIdx는 마지막 요약의 마지막 chatMemo와 일치하는 챗의 다음 인덱스로 매 턴 재계산된다.
  근거: hypav3.ts:214-229 `const lastChatIndex = chats.findIndex((chat) => chat.memo === [...lastSummary.chatMemos].at(-1)); if (lastChatIndex !== -1) { startIdx = lastChatIndex + 1; ... }` / :934 `...chats.slice(startIdx)`
- [7. 주입 — 실제 배치 위치(뮈토스)] 뮈토스 6.2 프리셋에는 type='memory' 카드(index 35, name 'Past Summary', role system, innerFormat 있음)가 있다. 이 경우 supaMemoryCardUsed=true가 되어 memo==='supaMemory' 메시지는 chat 블록에서 빠져 memories로 이동하고, chats 자리에는 빈 system이 들어갔다가 필터로 제거된다. 그리고 카드 35 위치에서 innerFormat의 {{slot}}에 치환되어 다시 push된다.
  근거: external/risuai/src/ts/process/index.svelte.ts:1169-1186 `if(v.memo !== 'supaMemory' && v.memo !== 'hypaMemory'){ v.removable = true } else if(supaMemoryCardUsed){ memories.push(v); return { role:'system', content:'' } } else { v.content = `<Previous Conversation>${v.content}</Previous Conversation>` }` + `.filter((v) => v.content.trim() !== '' ...)` / :1429-1443 `case 'memory':{ let pmt = safeStructuredClone(memories); applyPromptBlockRole(pmt, card.role2); if(card.innerFormat && pmt.length > 0){ pmt[i].content = risuChatParser(card.innerFormat,...).replace('{{slot}}', pmt[i].content) } pushPrompts(pmt) }` / 뮈토스 프리셋 promptTemplate[35] type='memory', id='mythos_v62_past_summary'
- [7. 주입 — memory 카드가 없을 때] memory 카드가 없는 프리셋에서는 요약 메시지가 chat 블록 index 0에 남고 <Previous Conversation>…</Previous Conversation>로 한 번 더 감싸진다.
  근거: external/risuai/src/ts/process/index.svelte.ts:1180-1182 `else{ v.content = `<Previous Conversation>${v.content}</Previous Conversation>` }`
- [8. 캐시 함의 — 위치] 뮈토스 6.2에서 memory 카드는 promptTemplate 49개 카드 중 index 35이고, chat 카드(37: rangeStart 0/rangeEnd -2, 39: rangeStart -2/rangeEnd end)보다 앞이다. 따라서 요약 블록은 시스템 프롬프트 한가운데에 들어가며, 그 뒤의 카드 36~48과 전체 대화 히스토리가 모두 그 뒤에 온다.
  근거: 뮈토스 프리셋 promptTemplate: [35] type='memory' name='Past Summary'; [37] type='chat' rangeStart=0 rangeEnd=-2; [39] type='chat' rangeStart=-2 rangeEnd='end' / index.svelte.ts:1383-1428 chat 카드 처리, :1429-1444 memory 카드 처리 (템플릿 순서대로 formated에 push)
- [8. 캐시 함의 — 판정] 요약 선택은 매 턴 다시 계산되며 (a) 새 요약이 추가되면 recent 선택이 바뀌고 (b) similar 선택은 최근 3챗(=매 턴 바뀜)에 대한 임베딩 검색 결과라 내용이 흔들린다. 그 결과 index 35의 system 메시지 content가 턴마다 달라질 수 있고, 프리픽스 캐시는 카드 35 이후 전체가 무효화된다. 원본 챗 제거(slice) 역시 chat 블록 앞부분을 잘라내므로 별도의 캐시 브레이크 요인이다.
  근거: hypav3.ts:676-695 (쿼리 = 매 턴 바뀌는 최근 3챗) → :741-772 (선택 결과 변동) → :877-880, :928-935 (memory content 변동) / index.svelte.ts:1429-1443 (그 content가 템플릿 중간에 삽입)
- [8. 캐시 함의 — 캐시 포인트] RisuAI의 자동 캐시 포인트는 chat 카드 처리 직후 마지막 user 메시지 3개에만 찍히며, 그것도 db.automaticCachePoint가 true이고 cache 카드가 없을 때만이다. 뮈토스 6.2 프리셋에는 cache 카드가 0개이고 automaticCachePoint 값도 프리셋에 없다(undefined). 즉 memory 블록 앞뒤에 명시적 캐시 경계가 없다.
  근거: index.svelte.ts:1413-1426 `if(DBState.db.automaticCachePoint && !hasCachePoint){ let pointer = formated.length - 1; let depthRemaining = 3; while(...) if(formated[pointer].role === 'user'){ formated[pointer].cachePoint = true; depthRemaining-- } }` / :815-817 `case 'cache':{ hasCachePoint = true }` / 뮈토스 프리셋 카드 타입 집계: cache 0개, automaticCachePoint=None
- [9. first message(greeting) 처리] greeting은 group chat이 아니고 disabled='allBefore'로 리셋되지 않은 경우 role='assistant'로 chats에 push되며 memo 필드가 없다(undefined). 요약 스킵 조건(example name / NewChat / 빈 content / user role)에 걸리지 않으므로 요약 대상에 포함된다.
  근거: external/risuai/src/ts/process/index.svelte.ts:868-884 `if(nowChatroom.type !== 'group' && !msReseted){ const firstMsg = currentChat.fmIndex === -1 ? nowChatroom.firstMessage : nowChatroom.alternateGreetings[currentChat.fmIndex]; const chat:OpenAIChat = { role: 'assistant', content: ... }; chats.push(chat) }` (memo 미지정) / hypav3.ts:317-342 스킵 조건에 memo undefined 케이스 없음
- [9. greeting — memo undefined 부작용] greeting이 배치의 마지막 챗이면 chatMemos에 undefined가 들어가고, JSON 직렬화에서 null이 되었다가 로드 시 undefined로 복원된다. 다음 턴 startIdx 계산은 `chat.memo === undefined`로 첫 매치를 찾는데, exampleMessage들도 memo가 undefined라 example이 있으면 잘못된 인덱스에 매칭될 수 있다.
  근거: hypav3.ts:460 `chatMemos: new Set(toSummarizeArray[i].map((chat) => chat.memo))` / :1628-1632 `chatMemos: new Set(summary.chatMemos.map((memo) => (memo === null ? undefined : memo)))` / :216-218 `chats.findIndex((chat) => chat.memo === [...lastSummary.chatMemos].at(-1))` / external/risuai/src/ts/process/exampleMessages.ts:58-65 `return { role: r.role, content: ..., name: r.name, memo: r.memo }` (example_user/assistant는 memo undefined)
- [export 필드 → 코드 매핑] export의 19개 필드는 createHypaV3Preset(name, settings)의 화이트리스트 병합으로 들어간다. 키가 HypaV3Settings에 있고 typeof가 기본값과 같아야만 채택되며, 누락된 summaryChunkSeparator는 기본값 '\\n\\n'가 유지된다.
  근거: external/risuai/src/lib/Setting/Pages/OtherBotSettings.svelte:1156-1159 `const newPreset = createHypaV3Preset(objImport.data.name || "Imported Preset", objImport.data.settings || {})` / hypav3.ts:1814-1824 `for (const [key, value] of Object.entries(existingSettings)) { if (key in settings && typeof value === typeof settings[key]) { settings[key] = value } }` / hypav3.ts:1790-1812 기본값 목록
- [alwaysToggleOn] alwaysToggleOn=true면 캐릭터 선택 시 supaMemory 토글이 강제로 켜진다. 이 토글(nowChatroom.supaMemory)이 꺼져 있으면 HypaV3 자체가 호출되지 않는다. export는 false.
  근거: external/risuai/src/ts/stores.svelte.ts:189-191 `if (DBState.db.hypaV3 && DBState.db.hypaV3Presets?.[DBState.db.hypaV3PresetId]?.settings?.alwaysToggleOn) { DBState.db.characters[selIdState.selId].supaMemory = true; }` / index.svelte.ts:1068 `if(nowChatroom.supaMemory && (... || DBState.db.hypaV3)){` / export `.alwaysToggleOn = False`
- [orphan 정리] preserveOrphanedMemory=true면 cleanOrphanedSummary가 호출되지 않는다. false면 현재 chats의 memo 집합에 chatMemos가 부분집합이 아닌 요약을 전부 버린다(리롤/삭제로 챗이 사라지면 해당 요약 소멸). export는 true라 보존된다.
  근거: hypav3.ts:207-209 `if (!settings.preserveOrphanedMemory) { cleanOrphanedSummary(chats, data); }` / :1646-1661 `data.summaries = data.summaries.filter((summary) => isSubset(summary.chatMemos, currentChatMemos));` / export `.preserveOrphanedMemory = True`
- [요약 0개일 때 exp vs legacy 차이] exp 경로는 요약이 하나도 없으면 memory 메시지를 아예 삽입하지 않고 chats.slice(startIdx)만 반환한다. legacy 경로는 빈 <Past Events Summary></Past Events Summary> system 메시지를 삽입한다. 이 차이는 첫 턴들의 프리픽스 형태를 바꾼다.
  근거: hypav3.ts:480-499 `if (data.summaries.length === 0) { const newChats: OpenAIChat[] = chats.slice(startIdx); ... return { currentTokens, chats: newChats, memory: ... } }` / :1188-1216 legacy `const memory = wrapWithXml(memoryPromptTag, ""); const newChats = [{ role:"system", content: memory, memo:"supaMemory" }, ...chats.slice(startIdx)]`
- [토크나이저 오버헤드] tokenizeChat은 content 토큰 + chatAdditionalTokens를 더한다. chatAdditionalTokens는 aiModel이 'gpt'로 시작하면 5, 아니면 3. name은 useName==='name'일 때만 추가(gpt 계열은 'noName').
  근거: external/risuai/src/ts/tokenizer.ts:421-439 `let encoded = (await encode(data.content)).length + this.chatAdditionalTokens; if(data.name && this.useName ==='name'){ encoded += (await encode(data.name)).length + 1 }` / index.svelte.ts:287-293 `if(DBState.db.aiModel.startsWith('gpt')){ caculatedChatTokens += 5 } else { caculatedChatTokens += 3 }` / :342 `new ChatTokenizer(chatAdditonalTokens, DBState.db.aiModel.startsWith('gpt') ? 'noName' : 'name')`

# open_questions
db.hypaModel(임베딩 모델)은 하이파 프리셋 export에 들어있지 않은 전역 설정이다. 뮈토스 커뮤니티가 실제로 어떤 값을 쓰는지(MiniLM 로컬 vs openai3small vs bgeM3Ko vs voyageContext3) 확인되지 않았다. 벤치에서는 MiniLM 기본값 가정을 명시하고 sensitivity로 다뤄야 한다.
db.automaticCachePoint 값이 뮈토스 프리셋에 없다(undefined). 사용자 전역 설정이므로 실제 커뮤니티 유저가 켜는지 여부는 미확인. 켜져 있어도 마지막 user 3개에만 찍히므로 memory 블록 앞 프리픽스는 보호되지 않는다는 결론은 동일하다.
db.temperature가 뮈토스 프리셋에서 -1000(센티널)로 나온다. requestChatData가 db.temperature/100 = -10을 그대로 쓰는지, 프로바이더 단에서 sentinel로 걸러 파라미터를 생략하는지 확인 필요(요약 결정론성에 영향).
doNotSummarizeUserMessage=false이므로 user 메시지도 요약되지만, 스킵된 챗의 토큰이 이미 차감되는 회계 버그(hypav3.ts:313)가 실측 런에서 얼마나 자주 발동하는지는 미측정. 재현 시 이 버그를 그대로 살릴지 여부를 결정해야 한다.
exp 경로에서 startIdx 매칭이 memo undefined일 때 example 메시지에 잘못 걸리는 케이스가 실제 뮈토스 카드(exampleMessage 유무)에 따라 발생하는지 미확인.
summarize() 호출이 maxTokens=db.maxResponse(30000)를 쓰므로 요약 출력이 사실상 무제한이다. 뮈토스 프롬프트가 스스로 길이를 제한하는지, 실측 요약 길이가 몇 토큰인지는 라이브 런으로만 확인 가능.