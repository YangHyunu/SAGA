# summary
RisuAI는 maxContext를 시스템프롬프트/로어북/프리셋(front-load) + 응답예약(maxResponse) + 채팅히스토리가 공유하는 단일 토큰 풀로 다루며, 메모리모듈 OFF 시 히스토리를 개별 메시지 단위(페어 정렬 없음)로 가장 오래된 것부터 하나씩 FIFO 제거한다(index.svelte.ts:1143-1154). greeting(첫 메시지)은 이 큐에서 구조적으로 보호되지 않고 examples/NewChat마커 바로 뒤에 위치해 히스토리보다 먼저 잘려나갈 수 있다. 메모리모듈 ON(hypaV2/V3/supaMemory)이면 잘려나갈 구간을 LLM 요약으로 만들어 "system, memo:'supaMemory'" 메시지로 치환하는데, 이때도 greeting 자체는 특별 보호 대상이 아니고 오직 'NewChat' 마커와 example 메시지만 요약 대상에서 제외된다(hypav3.ts:1110-1113, hypav2.ts:433-437). 로어북은 별도의 loreBookToken 예산으로 먼저 선별된 뒤 고정 비용으로 currentTokens에 편입되며, 로어북/프리셋/persona 자체는 최종 안전장치(removable 플래그, index.svelte.ts:1507-1523)를 빼면 트림 대상이 아니다. 우리 벤치의 token_trim은 페어 경계로만 자르고, 예산이 preset/lore 크기와 완전히 분리된 고정 상수(TRIM_TOKENS=12000)이며, greeting의 토큰 비용을 트림 판정 계산에서 아예 누락시킨다는 점에서 RisuAI 실동작과 다르다.

# spec
## 1. maxContext 계산

- `maxContextTokens = DBState.db.maxContext` — 유저 프리셋 설정값 그대로, 기본 4000 (`storage/database.svelte.ts:1995`). 별도 계산·클램핑 없음(`index.svelte.ts:345`).
- 응답 예약은 "예산에서 뺀다"기보다 "이미 쓴 걸로 치고 시작"하는 방식: `currentTokens = DBState.db.maxResponse` (기본 300, `storage/database.svelte.ts:1996`) `+ 50`(안전마진)로 카운터를 초기화(`index.svelte.ts:614,618`). 이후 프리셋/로어북/히스토리 토큰이 여기에 계속 더해지고, 이 누적값이 `maxContextTokens`를 넘는지로 판정한다. 결과적으로 히스토리+프리셋+로어북이 실제로 쓸 수 있는 몫은 `maxContext - maxResponse - 50`.
- 최종 조립본(`formated`)에 대해 실토크나이저로 다시 계산하는 2차 체크도 있음(`index.svelte.ts:1500-1529`), 여기서 `outputTokens`(응답 예약)도 `inputTokens + outputTokens > maxContextTokens`면 축소됨(1527-1528행).

## 2. 자르기 단위

- **메시지 단위, 페어 아님.** 메모리모듈 OFF 경로(`index.svelte.ts:1143-1154`)는 `chats[0]` 하나씩 `splice`로 제거 — user/assistant 쌍을 함께 보장하지 않는다.
- **가장 오래된 것부터.** `chats` 배열 자체가 시간순(examples → NewChat마커 → greeting → msg1...msgN)으로 조립되고 인덱스 0부터 제거하므로 FIFO.
- 메모리모듈 ON 경로(hypaV2/V3/supaMemory)도 오래된 쪽(`startIdx`/`chats[0]`)부터 소비하지만, 여러 메시지를 배치로 묶어 LLM에 요약 요청하는 점이 다름 — 이때도 배치 경계가 user/assistant 페어에 정렬된다는 보장은 없다(예: hypaV2는 `chats.length-4`라는 메시지 개수 마진만 둠, `hypav2.ts:417`).

## 3. first message(greeting) 보존 여부

**보존되지 않는다.** 구조적 특권이 없다.
- push될 때 `memo` 필드조차 없음(`index.svelte.ts:871-876`) — hypaV3/V2가 보호하는 `memo==='NewChat'`(마커 전용) 매칭 대상이 아님.
- OFF 경로: `chats[0]`부터 잘리는 큐에서 examples/NewChat마커 바로 다음(즉 실제 대화 히스토리보다 먼저) 위치 → 히스토리가 자라면 greeting이 먼저 증발.
- ON 경로: hypaV3는 `chat.memo==='NewChat'`과 example만 요약 텍스트에서 skip(`hypav3.ts:1110-1113`), hypaV2는 인덱스 0(보통 NewChat 마커)만 skip(`hypav2.ts:432-437`) — greeting은 둘 다 skip 대상이 아니라 결국 요약에 흡수됨.

## 4. 메모리모듈 OFF vs ON 경로 차이

게이트: `nowChatroom.supaMemory && (supaModelType!=='none' || hanuraiEnable || hypav2 || hypaV3)` (`index.svelte.ts:1068`).

- **OFF**: 순수 슬라이딩 윈도우. `while(currentTokens>maxContextTokens){ chats[0] 제거 }`, 못 줄이면(`chats.length<=1`) 에러(`index.svelte.ts:1143-1154`).
- **ON**: 잘려나갈 구간을 LLM으로 요약해서 대체.
  - hypaV3: 오래된 인덱스 구간을 배치로 나눠 `summarize()` 호출 → `data.summaries`에 축적 → 최종적으로 `[{role:'system', content: '<Past Events Summary>...', memo:'supaMemory'}, ...chats.slice(startIdx)]` 형태로 재조립(`hypav3.ts:1192-1199`, `1593-1600`). **이 지점이 "잘린 챗 → 요약 대체" 지점.**
  - hypaV2: 유사 구조, `[Start a new chat]`(인덱스 0)만 건너뛰고 청크 단위로 요약(`hypav2.ts:400-479`).
  - 구버전 supaMemory: NewChat 마커 이전(examples)만 무조건 삭제 후, 남은 큐를 청크로 나눠 반복 요약(`supaMemory.ts:26-38, 286-382`), `chats.unshift({role:'system', content:supaMemory, memo:'supaMemory'})`로 대체(`supaMemory.ts:378-382`).

## 5. 로어북/프리셋 토큰 예산 처리

- 로어북은 자체 예산(`char.loreSettings?.tokenBudget ?? DBState.db.loreBookToken`, `lorebook.svelte.ts:85`)으로 활성 항목을 먼저 골라낸 뒤, 그 결과가 `unformated.lorebook`에 담겨 `currentTokens`에 **고정 비용**으로 편입(`index.svelte.ts:723-725` 또는 `823-828`).
- 프리셋(main/jailbreak/globalNote/authorNote/description/postEverything/personaPrompt)도 동일하게 먼저 토큰화되어 `currentTokens`에 누적(`index.svelte.ts:676-829`).
- **이들은 히스토리 트림 루프(1143-1154)의 대상이 아니다** — 오직 `unformated.chats`(실제 대화 메시지)만 `removable=true`가 붙어 트림/2차 안전장치 대상이 됨(`index.svelte.ts:1169-1172`, `1507-1523`). 로어북/프리셋이 너무 크면 히스토리를 1개까지 다 비워도(chats.length<=1) 여전히 초과할 수 있고, 그 경우 에러로 죽는다(`toomuchtoken`).
- 즉 예산 배분 순서: `maxContext` → (maxResponse 예약분 차감) → (프리셋+로어북 고정 소비) → 남는 몫만 히스토리.

## 6. 우리 token_trim vs RisuAI 실동작

일치: 방향성(오래된 것부터 FIFO), greeting이 결국 보호받지 못하고 사라진다는 정성적 결론.

**불일치 (수정 필요 목록)**:

1. **자르기 입도**: `token_trim`(run2.py:141-164)은 `starts`(user 인덱스) 경계에서만 자름 — 완전한 페어 단위. RisuAI는 메시지 단위(`index.svelte.ts:1150-1151`)라 페어 중간에서 끊길 수 있다. → 페어 단위 트림을 메시지 단위로 바꾸거나, 최소한 이 차이를 알고 "우리 쪽이 약간 더 보수적으로(더 많이) 자른다"는 방향성 편향을 감안해야 함.
2. **예산이 preset/lore 크기와 완전히 분리된 고정 상수**: `TRIM_TOKENS=12000`(run2.py:59)은 `build_wire()`가 만드는 system_prompt/lore/persona/authornote 비용과 무관하다(run2.py:382, 389). RisuAI는 이 모든 게 `maxContext` 하나를 공유하고 preset/lore가 먼저 소비한 나머지만 히스토리 몫이 된다(위 5번). → 카드별 preset+lore 실측 토큰을 빼고 남은 값을 `trim_tokens`로 넘기도록 고쳐야 RisuAI의 "공유 풀" 동작을 재현한다.
3. **greeting 토큰 비용이 트림 판정에서 완전히 빠짐**: `starts`가 user 인덱스만 담기 때문에, greeting(`history[0]`, role=assistant)은 어떤 `seg` 합산에도 절대 포함되지 않는다(run2.py:149-164). RisuAI는 greeting을 `tokenizer.tokenizeChat`으로 실제 카운트해서 `currentTokens`에 반영한다(`index.svelte.ts:883`). → `token_trim`이 트림 불필요 판정(`keep>=total_pairs`)을 내릴 때도 greeting 토큰을 합산에 포함시켜야 한다. 큰 greeting을 가진 카드에서는 지금 구조상 예산 초과를 놓칠 수 있다.
4. (참고, 우선순위 낮음) 우리 시뮬레이션에는 RisuAI의 examples/NewChat마커/2차 removable 안전장치에 대응하는 요소가 없다 — 벤치 스코프상 의도된 단순화로 보이지만, "RisuAI 기본 동작 충실 재현"이 목표라면 이 차이도 명시적으로 문서화해 둘 것.


# claims
- [maxContext 출처] maxContextTokens는 DBState.db.maxContext를 그대로 사용 (유저 프리셋 설정값, 기본 4000). 응답 예약은 별도 변수에서 뺀 게 아니라 currentTokens 쪽에 maxResponse+50을 먼저 더해 시작해서 사실상 (예산 - maxResponse)만큼만 나머지 프롬프트/히스토리가 쓸 수 있게 만드는 방식.
  근거: index.svelte.ts:345 `let maxContextTokens = DBState.db.maxContext`; index.svelte.ts:614,618 `let currentTokens = DBState.db.maxResponse; ... currentTokens += 50`; storage/database.svelte.ts:1995 `maxContext: 4000`
- [자르기 단위 = 메시지 단위, 페어 아님] 메모리모듈 OFF 경로는 chats[0] 하나씩 splice로 제거하는 순수 메시지 단위 FIFO다. user/assistant 페어를 함께 자른다는 보장이 전혀 없다 — 잘린 뒤 남은 첫 메시지가 assistant일 수도 있다.
  근거: index.svelte.ts:1143-1154 `while(currentTokens > maxContextTokens){ if(chats.length<=1){throwError...} currentTokens -= await tokenizer.tokenizeChat(chats[0]); chats.splice(0,1) }`
- [가장 오래된 것부터 (FIFO, 배열 앞쪽)] chats 배열은 [examples..., '[Start a new chat]' 마커, greeting, msg1, msg2, ...] 순서로 조립되고 인덱스 0부터 제거하므로 항상 오래된 쪽부터 잘린다.
  근거: index.svelte.ts:837-845(examples+NewChat마커 push), 866-884(greeting push), 900-1053(ms 루프로 실제 히스토리 메시지 개별 push), 1150-1151(chats[0] 기준 제거)
- [greeting(첫 메시지) 보존 여부] greeting은 구조적으로 보호되지 않는다. push될 때 memo 필드조차 없고(그냥 role/content만), OFF 경로에서는 examples/NewChat마커 다음으로 큐 맨 앞쪽에 있어 히스토리보다 먼저 잘려나갈 수 있다. ON 경로(hypaV2/V3)도 memo==='NewChat' 마커와 example만 요약 대상에서 skip할 뿐 greeting 자체는 skip 대상이 아니라 결국 요약에 흡수된다.
  근거: index.svelte.ts:871-876 (greeting 객체에 memo 없음); hypav3.ts:1110-1113 `if (chat.memo === 'NewChat') { ...continue }` (greeting은 매치 안 됨); hypav2.ts:432-437 `if (idx === 0) { ...idx++; continue }` (인덱스 0 마커만 skip, greeting은 skip 안 됨)
- [메모리모듈 게이트 조건 (ON/OFF 분기점)] nowChatroom.supaMemory가 true이면서 supaModelType!=='none' 이거나 hanuraiEnable/hypav2/hypaV3 중 하나라도 켜져 있으면 ON 경로(hanurai/hypaV2/hypaV3/supaMemory 중 하나 실행), 아니면 OFF(순수 슬라이딩 윈도우 while loop).
  근거: index.svelte.ts:1068 `if(nowChatroom.supaMemory && (DBState.db.supaModelType !== 'none' || DBState.db.hanuraiEnable || DBState.db.hypav2 || DBState.db.hypaV3))` ... else 분기가 1141-1154
- [ON일 때 잘린 챗 → 요약 치환 지점] hypaV3는 예산 초과분을 여러 배치로 나눠 LLM 요약(summarize)한 뒤, 최종적으로 '<Past Events Summary>...</Past Events Summary>' XML을 담은 system 메시지(memo:'supaMemory') 하나를 배열 맨 앞에 넣고 그 뒤에 chats.slice(startIdx)(요약 안 된 최근 구간)를 이어붙인다. 이 패턴이 코드베이스에 반복 등장한다(요약 0개/N개 케이스 모두 동일 구조).
  근거: hypav3.ts:1192-1199 및 hypav3.ts:1593-1600 `const newChats = [{role:'system', content: memory, memo:'supaMemory'}, ...chats.slice(startIdx)]`; summarize 배치 로직은 hypav3.ts:1058-1174
- [supaMemory(구버전) 트림도 메시지 단위 + LLM 요약] supaMemory 함수도 NewChat 마커 이전(examples)만 무조건 제거한 뒤, 남은 큐(마커+greeting+히스토리)를 chats[0]부터 순서대로 chunk에 담아 요약 LLM 호출로 압축한다. 개별 항목 role 필터 없이 stringlizedChat에 다 들어간다.
  근거: supaMemory.ts:26-38 (NewChat 이전만 제거), supaMemory.ts:286-348 (while(currentTokens>maxContextTokens) 루프, chats[spiceLen] 순차 소비 후 splice(0,spiceLen))
- [로어북 토큰 예산] 로어북은 loadLoreBookV3Prompt() 내부에서 char.loreSettings.tokenBudget ?? DBState.db.loreBookToken이라는 별도 예산으로 먼저 활성 항목을 선별한다. 선별된 로어북 텍스트는 unformated.lorebook에 담겨 currentTokens에 고정비용으로 편입되며, 이후 히스토리 트림 루프의 대상이 아니다(로어북 자체는 removable 플래그가 안 붙음).
  근거: lorebook.svelte.ts:85 `const loreToken = char.loreSettings?.tokenBudget ?? DBState.db.loreBookToken`; index.svelte.ts:723-725/1169-1172 (unformated.lorebook 토큰화는 되지만 removable=true는 chats(히스토리)에만 부여)
- [프리셋(main/jailbreak/globalNote 등) 토큰] 프리셋 각 섹션(main, jailbreak, globalNote, authorNote, description, postEverything, personaPrompt)은 promptTemplate 순회 중 tokenizeChatArray로 currentTokens에 누적되며(템플릿 없으면 823-828의 단순 for 루프), 이 값들은 examples/greeting/히스토리보다 먼저 계산되어 예산에서 '선차감'된다 — 즉 남는 예산만 히스토리 몫이 된다.
  근거: index.svelte.ts:676-829 (promptTemplate 순회 tokenizeChatArray 호출들, 특히 697/708/720/724/728/772/777/808행); 823-828 (템플릿 미사용 시 단순 합산)
- [최종 안전장치 (2차 트림)] 전체 formated 배열을 실토크나이저로 재계산했을 때도 초과하면, removable=true가 붙은 항목(오직 unformated.chats 유래 히스토리 메시지만 해당, memo가 supaMemory/hypaMemory인 요약메시지는 제외)만 앞에서부터 content를 지우는 2차 안전장치가 있다. 로어북/프리셋/depth프롬프트는 removable이 아니라서 이 단계에서도 안 잘린다.
  근거: index.svelte.ts:1169-1172 (`v.removable = true`는 supaMemory/hypaMemory 메모가 아닌 chats 항목에만); index.svelte.ts:1507-1523 (`while(inputTokens > maxContextTokens){ if(formated[pointer].removable){...} pointer++ }`)
- [our token_trim: 페어 경계로만 자름 (RisuAI는 메시지 단위)] token_trim은 user 메시지 인덱스(starts)를 경계로만 후보 컷포인트를 만들어 완전한 페어 단위로만 자른다. RisuAI 실동작(메시지 단위 FIFO, 페어 정렬 없음)보다 더 거친 입도다.
  근거: run2.py:149-164 `starts = [i for i,m in enumerate(history) if m['role']=='user']; ... seg = history[starts[k-1]:]`
- [our token_trim: 예산이 preset/lore 크기와 무관한 고정 상수] TRIM_TOKENS=12000(기본, --trim-tokens로 변경 가능)은 build_wire()가 조립하는 system_prompt/lore/persona/authornote 크기와 완전히 분리된 고정값이다. RisuAI는 이 모든 게 하나의 maxContextTokens 풀을 공유하고 preset/lore가 먼저 소비한 나머지만 히스토리에 배정되는데, 우리 시뮬레이션은 이 상호작용을 재현하지 않는다.
  근거: run2.py:59 `TRIM_TOKENS = 12000`; run2.py:382 `window, win_start = token_trim(history, trim_tokens)` — build_wire(preset,...)의 preset 비용과 무관하게 고정 budget만 사용
- [our token_trim: greeting 토큰 비용이 예산 계산에서 누락됨] greeting은 history[0](role=assistant)로 들어가지만 starts 리스트(user 인덱스)에 없으므로, keep>=total_pairs(트림 불필요) 케이스에서도 loop의 세그먼트 합산(`history[starts[k-1]:]`)에는 절대 포함되지 않는다. 즉 greeting이 아무리 커도 트림 여부/가능 판정에 전혀 반영되지 않고, 트림이 한번 발동하면 greeting은 그 즉시 통째로 사라진다(주석에도 명시).
  근거: run2.py:149-164 (seg는 항상 starts[...] 부터 시작 — index 0의 greeting은 어떤 seg 합산에도 안 들어감); run2.py:146-147 주석 '첫 user 이전 메시지는 트림이 시작되는 순간 통째로 떨어진다'

# open_questions
run2.py의 --trim-tokens 기본값(12000)과 실제 벤치 실행에서 쓰인 값(유저 프롬프트에 언급된 32K)이 다른데, 실측 리포트가 어느 값 기준인지 확인 필요
hypaMemoryV3의 useExperimentalImpl(hypaMemoryV3MainExp) 분기가 실제 프로덕션 기본 프리셋에서 켜져 있는지 여부 — 두 구현이 거의 동형이라 본 조사에서는 Main 경로만 상세 인용함
우리 벤치가 재현 목표로 삼는 RisuAI 설정이 supaMemory OFF(순수 슬라이딩 윈도우)인지 hypaV2/V3 ON인지에 따라 5번 스펙의 '고칠 것' 우선순위가 달라짐 — 어느 쪽이 목표인지 확인 필요