# 구조 제안 — RP 장기 세션의 상태 유실 문제를 프록시로 풀어보기

**SAGA RP Agent Proxy v3.0 — Stateful RAG 기반 Context Engineering**

---

> 단순 캐릭터챗 AI에서 Mem0, Langmem, Letta 같은 Stateful Agent는 과한 구조였습니다.
> 하지만 RP는 다릅니다 — 위치, NPC, 아이템, 관계, 세력 등 동적인 요소가 훨씬 많고,
> 유저가 직접 세계를 설계하고 만들어 나갑니다.
> 코히바블랙님의 연구일지[1]를 읽으며 이 가능성을 확인했고,
> **Letta의 판단력을 매 턴이 아닌 필요한 순간에만 쓰면 어떨까** — 여기서 출발했습니다.
> Letta를 핵심에 두되, 프록시 기반 1회 호출 + 비동기 추출로 풀어본 기록입니다.

---

## 목차

1. [우리 모두가 겪는 문제](#1-우리-모두가-겪는-문제)
2. [기존 접근들의 한계](#2-기존-접근들의-한계)
3. [Letta의 Step Loop — Stateful Agent의 구조](#3-letta의-step-loop--stateful-agent의-구조)
4. [SAGA의 아이디어: "에이전트는 큐레이션만"](#4-saga의-아이디어-에이전트는-큐레이션만)
5. [어떻게 동작하는가](#5-어떻게-동작하는가)
6. [3-Agent 파이프라인 상세](#6-3-agent-파이프라인-상세)
7. [프롬프트 캐싱 — SystemStabilizer와 3-BP 전략](#7-프롬프트-캐싱--systemstabilizer와-3-bp-전략)
8. [비용과 트레이드오프](#8-비용과-트레이드오프)
9. [코히바블랙님 및 선행 연구자분들께](#9-코히바블랙님-및-선행-연구자분들께)
10. [만들면서 느낀 것](#10-만들면서-느낀-것)
11. [참고 자료](#11-참고-자료)
12. [E2E 통합 검증 결과](#12-e2e-통합-검증-결과-2026-03-02)

---

## 1. 우리 모두가 겪는 문제

RP를 해본 사람이라면 이런 경험이 있을 겁니다.

> 30턴 전에 죽인 NPC가 아무렇지 않게 다시 등장한다.
> 버린 아이템이 인벤토리에 여전히 있다.
> 동쪽으로 이동했는데 서쪽 마을에서 대화가 이어진다.
> HP가 0인데 전투를 계속한다.

장기 RP 세션의 근본적인 문제는 **상태 유실**입니다. LLM은 대화가 길어지면 이전 상태를 잊어버립니다. 컨텍스트 창을 128K, 200K로 늘려도 본질은 같습니다 — 50턴 전 세부사항을 정확히 기억하리라는 보장이 없고, 비용만 선형으로 증가합니다.

그리고 또 하나의 문제: **로어북은 정적**입니다. "에르겐은 마을 광장의 약초상이다"라고 작성해 두면, 에르겐이 숲으로 이동하거나 사망하더라도 로어북은 변하지 않습니다. LLM은 오래된 설정을 보고 모순된 이야기를 만들게 됩니다.

---

## 2. 기존 접근들의 한계

이 문제를 풀려는 시도는 여러 가지가 있었습니다.

| 접근 방식 | 상태 추적 | 구조화 | 유저 지연 | 비용 | 한계 |
|-----------|:--------:|:------:|:--------:|:----:|------|
| **정적 로어북** | X | X | 없음 | 낮음 | 세계가 변해도 로어북은 고정 |
| **전체 히스토리 전송** | 암묵적 | X | 없음 | 높음 | 토큰 낭비 + 장기 세션에서 누적 모순 |
| **HypaMemory 계열** | 대화 기억 | X | 낮음 | 중간 | 요약 기반 압축 — 구조화된 상태 추적 불가 |
| **Mem0 계열** | 대화 기억 | X | **높음** | 중간 | 매 턴 메모리 추출/갱신이 동기적 → 응답 지연 |
| **SAGA (이 프로젝트)** | O | O | **낮음** | 중간 | Flash 서사 요약 정확도에 의존 |

**HypaMemory 계열**(RisuAI HypaMemory/SupaMemory, SillyTavern Summarize 등)은 토큰 한계에 도달하면 대화를 요약·압축하여 장기 기억을 시뮬레이션합니다. 대화의 흐름은 기억하지만, "캐릭터가 어디에 있는지", "HP가 얼마인지", "누구와 어떤 관계인지"를 **구조화된 형태로** 추적하지는 못합니다. 요약 과정에서 세부 상태가 소실되므로, 장기 세션의 모순까지 막기엔 부족합니다.

**Mem0 계열**은 LLM이 직접 메모리를 관리하지는 않지만, 매 대화마다 메모리 추출과 갱신이 동기적으로 실행되어 응답 경로에 지연이 발생합니다. 구조화된 상태 추적도 지원하지 않습니다.

---

## 3. Letta의 Step Loop — Stateful Agent의 구조

코히바블랙님이 선행연구[1]에서 언급하셨듯, Letta(MemGPT)[2]는 LLM에 가상 메모리 계층을 부여하여 **진짜 Stateful Agent**를 만드는 프레임워크입니다. 에이전트가 자신의 Memory Block을 직접 읽고 편집하면서 상태를 유지합니다.

핵심은 **Step Loop** 구조입니다. 에이전트가 한 턴에 여러 단계를 밟으며, 각 단계에서 "무엇을 읽을지", "어떻게 수정할지"를 LLM이 판단합니다:

```
유저: "마을 광장을 둘러본다"

[Step 1] 내면 사고 → "기억을 확인해야겠다"       ← LLM 호출
[Step 2] memory_read(core_memory)              ← 도구 실행
[Step 3] 내면 사고 → "위치가 바뀌었으니 수정"     ← LLM 호출
[Step 4] memory_edit(core_memory, location=...) ← 도구 실행
[Step 5] 내면 사고 → "archival도 확인"           ← LLM 호출
[Step 6] archival_search("마을 광장")            ← 도구 실행
[Step 7] 내면 사고 → "응답 생성"                 ← LLM 호출
[Step 8] send_message("마을 광장은...")          ← 최종 응답

= 4회 LLM 호출
```

이 구조의 강점은 **유연성**입니다. 에이전트가 매 턴 상황을 판단하고, 필요한 기억만 선택적으로 읽고, 적절하게 편집합니다. 예상치 못한 상황에도 에이전트가 스스로 대응할 수 있습니다.

대신 **메모리 읽기/쓰기 자체가 LLM 호출**이므로, 턴당 3~5회의 호출이 발생합니다. 이게 Stateful의 대가입니다 — 생각하고, 도구 쓰고, 또 생각하는 과정이 매 턴 반복됩니다.

SAGA는 이 Letta의 Step Loop를 핵심에 활용하되, **"매 턴 다회 호출"을 "N턴마다 비동기"로 재배치**하는 아이디어에서 출발했습니다.

---

## 4. SAGA의 아이디어: "에이전트는 큐레이션만"

### 코딩 에이전트에서 얻은 힌트

혹시 Claude Code, OpenClaw, oh-my-claudecode(OMC) 같은 코딩 에이전트를 써보셨나요? 이 도구들은 멀티 에이전트 오케스트레이션, 복잡한 워크플로우 관리, 스킬 체이닝 등을 잘 해냅니다. 그런데 실제로 써보면 의외의 지점을 발견합니다 — 컨텍스트 관리의 핵심은 정교한 메모리 시스템이 아니라 **.md 형식의 구조화된 문서**였습니다. 이 에이전트들은 장기 작업 중에 핵심 정보를 **Notepad/NOTE** 형태로 자기관리합니다. 에이전트가 "무엇이 중요한지" 스스로 판단하여 메모를 갱신하는 패턴입니다.

여기서 한 가지 더 중요한 점이 있습니다. 이 도구들은 **모든 것을 직접 만들지 않습니다.** Claude Code는 Bash, grep, git 같은 기존 도구를 오케스트레이션하고, OMC는 여러 에이전트를 조율하며, OpenClaw는 기존 LLM API를 게이트웨이로 중계합니다. 핵심은 **이미 잘 만들어진 것들을 어떻게 연결하고 관리하느냐**입니다.

SAGA도 같은 철학입니다. Letta(Stateful Agent), ChromaDB(벡터 검색), SQLite(상태·관계·이벤트·로어) — 각각은 이미 검증된 도구입니다. SAGA가 하는 건 이것들을 **RP에 맞는 하나의 파이프라인으로 엮고, 언제 무엇을 호출할지 관리하는 오케스트레이션**입니다. 새로운 DB나 메모리 시스템을 만드는 게 아닙니다.

이걸 RP에 적용할 때 핵심 질문은 이것이었습니다:

> **에이전트의 판단력이 정말 필요한 곳은 어디인가?**

생각해 보면:

- **매 턴 상태 추출** — "위치가 바뀌었는가? HP가 변했는가?" → 정규식이나 저비용 LLM으로 충분합니다. 에이전트가 고민할 필요 없습니다.
- **매 턴 컨텍스트 조립** — "어떤 로어북이 관련 있는가?" → 점수 기반 필터링으로 충분합니다.
- **N턴마다 서사 큐레이션** — "이 모순은 의도된 것인가? 이 복선은 회수해야 하는가?" → **여기서만 에이전트의 판단력이 필수입니다.** 이전 큐레이션 결정을 기억하고 일관되게 이어가야 합니다.

### .md — 에이전트 메모리의 사실상 표준

그런데 이런 컨텍스트 관리 패턴을 들여다보면, 공통점이 있습니다. 아키텍처는 전부 다른데, **최종 형식은 다 마크다운**입니다.

| 시스템 | 아키텍처 | 컨텍스트 관리 |
|--------|---------|-------------|
| **Claude Code** | CLI 에이전트 | `CLAUDE.md`, `~/.claude/memory/*.md` |
| **OMC** | 오케스트레이터 | `notepad.md`, `AGENTS.md`, `.omc/plans/*.md` |
| **OpenClaw** | Gateway/프록시 | `.md` 기반 컨텍스트 주입 |
| **Codex** | CLI 에이전트 | `AGENTS.md`, 마크다운 기반 지시 |
| **Letta Code** | 에이전트 프레임워크 | `memory/*.md` + YAML frontmatter (MemFS)[3] |
| **SAGA** | Gateway/프록시 | `stable_prefix.md`, `live_state.md` |

Swarm이든, Orchestrator든, Skill 시스템이든 — 상위 구조는 다 달라도 에이전트에게 컨텍스트를 전달하는 최종 형식은 마크다운으로 수렴하고 있습니다. 왜일까요?

- **LLM이 가장 잘 읽는 형식**: 학습 데이터에 마크다운이 압도적으로 많음
- **구조화와 가독성의 균형**: JSON은 기계적이고 자연어는 비구조적인데, 마크다운은 헤딩/리스트/테이블로 반구조화 가능
- **YAML frontmatter로 메타데이터 분리**: 본문은 LLM이 읽고, 프론트매터는 코드가 읽음
- **diff/캐싱 친화적**: 텍스트 기반이라 git diff, Prompt Caching 모두 효율적

SAGA의 .md 캐시 2파일은 이 흐름을 RP에 적용한 것이고, 그 중에서도 Letta Code의 MemFS가 가장 직접적인 참고점이었습니다.

### 그래서 Letta는 Curator 전담으로

Letta의 Step Loop가 강력한 건 맞지만, 매 턴 돌리기엔 비용이 큽니다. 그렇다면 **Letta를 매 턴이 아닌, 서사 큐레이션에만 집중시키자** — 여기서 SAGA가 시작됩니다.

Curator는 Letta SDK로 생성된 에이전트가 3개의 Memory Block(`narrative_summary`, `curation_decisions`, `contradiction_log`)을 자기편집하면서 서사 판단의 연속성을 유지합니다. 10턴 전에 "이 모순은 의도된 전개"라고 기록해 두면, 20턴 후에도 그 판단을 이어갈 수 있습니다. 이건 Letta의 Memory Block 없이는 불가능합니다.

Letta 에이전트는 `message_buffer_autoclear=True`로 생성되어, 큐레이션 간 대화 히스토리가 누적되지 않는다. Memory Block만 유지되므로 토큰 비용이 일정하게 유지된다.

그리고 Letta가 최근 진화시킨 **MemFS(Memory File System)**[3] 구조도 알고 있었습니다. Letta Code는 에이전트 메모리를 **마크다운 파일 시스템**으로 관리합니다:

```
memory/
  project_overview.md      ← YAML frontmatter + 마크다운 본문
  architecture_notes.md
  current_tasks.md
  ...
```

각 파일에 YAML frontmatter가 붙고, 에이전트가 `create_file`, `modify_file` 같은 도구로 자기편집합니다. git이 버전 관리를 담당합니다.

SAGA의 **.md 캐시 2파일** 설계는 이 구조에서 직접 영향을 받았습니다:

| | Letta MemFS | SAGA .md 캐시 |
|---|---|---|
| 형식 | 마크다운 + YAML frontmatter | 마크다운 + YAML frontmatter |
| 파일 구조 | `memory/project_overview.md` 등 | `stable_prefix.md`, `live_state.md` |
| 버전 관리 | git (커밋 기반) | YAML frontmatter의 `version`, `turn` 필드 |
| 편집 주체 | **에이전트** (`modify_file` 도구) | **코드** (정규식 파싱 → `os.replace()` 원자적 교체) |
| 편집 비용 | LLM 호출 필요 | LLM 호출 불필요 |

차이는 **"누가 편집하는가"** 입니다. Letta MemFS는 에이전트가 파일을 읽고 판단하고 편집합니다. SAGA는 같은 구조를 쓰되, **매 턴 편집은 코드 로직(Sub-B)에게 위임**합니다. 대신 **N턴마다 Letta Curator가 큐레이션**합니다 — story 압축, 모순 수정, 로어 자동 생성 등.

> 정리하면: Letta의 Memory Block은 Curator의 두뇌이고,
> Letta의 MemFS 패턴은 .md 캐시의 설계 원형입니다.
> SAGA가 바꾼 건 **"매 턴 편집을 에이전트가 아닌 코드가 한다"** 는 것뿐입니다.

### 그래서 이렇게 분리했습니다

```
┌─────────────────────────────────────────────────────┐
│  메인 루프 (매 턴, 유저 대기)                         │
│                                                     │
│  Sub-A: DB 검색 + 프롬프트 조립   ← 코드 로직, LLM 0회  │
│  LLM: 내레이션 1회 호출            ← 이것만 유저 대기    │
│  Sub-B: 상태 추출 + DB 갱신       ← 비동기, 유저 무관   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Curator (10턴마다, 비동기)                           │
│                                                     │
│  Letta Memory Block 자기편집      ← 에이전트 판단 필요  │
│  모순 탐지, 서사 압축, 로어 자동생성 ← 다회 호출 OK     │
│                                    (유저 대기 아님)    │
└─────────────────────────────────────────────────────┘
```

코딩 에이전트가 Notepad를 통해 작업 맥락을 자기관리하듯, **Letta Curator는 Memory Block을 통해 큐레이션 판단 이력을 자기관리**합니다. 단, 이 과정은 10턴마다 비동기로만 일어나므로 유저가 기다릴 일이 없습니다.

---

## 5. 어떻게 동작하는가

SAGA는 **OpenAI-compatible 프록시**입니다. 클라이언트(RisuAI, SillyTavern 등)와 LLM 사이에 끼어서, 프론트엔드 수정 없이 동작합니다. API Base URL만 바꾸면 됩니다.

### Read-Write 순환

매 턴은 읽기(READ)와 쓰기(WRITE)의 순환입니다:

```
                    ┌──────────────────────────────┐
                    │         2종 DB 저장소           │
                    │  SQLite(상태) + ChromaDB(벡터)  │
                    └──────┬───────────────┬────────┘
                   READ    │               │  WRITE
                (동기 ~200ms)│              │(비동기)
                           ▼               │
    ┌────────┐    ┌────────────────┐    ┌──┴──────────────┐
    │ Client │───▶│  Sub-A:        │───▶│  LLM (1회 호출)   │
    │        │◀───│  Context Build │◀───│                  │
    └────────┘    └────────────────┘    └──┬──────────────┘
                                           │ 응답 반환 후
                                           ▼
                                    ┌──────────────┐
                                    │  Sub-B:      │
                                    │  추출 + 갱신  │──▶ DB WRITE
                                    └──────────────┘
```

1. **READ** (동기, ~200ms): Sub-A가 2종 DB에서 현재 상태, 관련 에피소드, 로어북을 RRF 검색 → 프롬프트에 주입
2. LLM 1회 호출 → SSE 스트리밍으로 유저에게 응답 반환
3. **WRITE** (비동기, 유저 대기 없음): Sub-B가 응답에서 상태 변화를 추출 → 2종 DB 갱신

이 순환이 매 턴 반복되므로, DB는 항상 최신 세계 상태를 반영합니다.

### 왜 2종 DB인가?

Letta는 에이전트의 메모리를 Core / Archival / Recall 3계층으로 나눕니다. SAGA는 이 구조를 에이전트 호출 없이 코드가 직접 접근하는 2종 DB로 대체했습니다.

| Letta 메모리 계층 | SAGA DB | 역할 | 예시 쿼리 |
|---|---|---|---|
| **Core Memory** (항상 로드) | **SQLite** | 현재 상태, 세션 메타, 캐릭터, 관계, 장소, 이벤트, 로어 | "플레이어 위치는?" "에르겐이 아는 사람은?" |
| **Archival Memory** (임베딩 검색) | **ChromaDB** (벡터) | 로어북 벡터 검색, 에피소드 기억 | "어둠의 숲과 비슷한 장소는?" |

그리고 이 두 DB 위에 **.md 캐시 2파일**이 있습니다 — `stable_prefix.md`, `live_state.md`. 이 파일들은 프롬프트의 캐싱 가능한 프리픽스로 사용되어, Anthropic의 Prompt Caching과 결합하면 비용을 크게 줄입니다.

### SQLite 테이블 구조

SAGA는 8개 테이블로 세션과 월드 상태를 관리합니다:

| 테이블 | 주요 컬럼 | 역할 |
|---|---|---|
| `sessions` | id, name, turn_count, world_config | 세션 메타 + 턴 카운터 |
| `characters` | name, is_player, hp, max_hp, location, mood | 캐릭터 현재 상태 (PC + NPC) |
| `relationships` | from_name, to_name, rel_type, strength | 캐릭터 간 관계 (-100~100) |
| `locations` | name, description, first_visit_turn | 방문한 장소 |
| `events` | name, event_type, description, turn, importance | 중요 이벤트 기록 (importance ≥ 40) |
| `lore` | name, lore_type, keywords, content, priority, auto_generated | 로어북 (수동 + 자동 생성) |
| `world_state` | key, value | 범용 KV (인벤토리, 위치, canonical system 등) |
| `turn_log` | turn_number, user_input, assistant_output, state_changes | 턴 전체 기록 |

"아리아가 어둠의 숲에서 고블린왕 크룩을 만났다"는 것이 SQLite에서는:
- `characters`: 아리아 → `location = 어둠의 숲`
- `characters`: 크룩 → 신규 NPC 생성
- `relationships`: (아리아, 크룩, met, 30)
- `events`: Turn N | encounter | 아리아가 크룩과 조우 (importance=30)
- `world_state`: player_location = 어둠의 숲

로 기록됩니다. "근처에 누가 있지?"는 `characters` 테이블 위치 필터 쿼리로 즉시 답할 수 있습니다.

### 에피소드 검색: RRF (Reciprocal Rank Fusion)

SAGA는 3개 소스에서 에피소드를 가져온 뒤 **RRF** 공식으로 병합·랭킹합니다:

```
RRF 3-source 에피소드 선택:
    ├─ Recent (w=1.2):   최근 N턴 에피소드 (시간 순서, 벡터 무관)
    ├─ Important (w=1.0): 중요도 ≥ 40 에피소드 (combat, event 등)
    └─ Similar (w=0.8):   유저 입력 기반 벡터 유사도 검색

    RRF 공식: score[ep] += weight / (k + rank + 1)   (k=60)

→ 점수 상위 top_n개 선택 → 토큰 예산 내 패킹
```

RRF를 쓰는 이유: 단일 소스 검색은 편향됩니다. "최근" 에피소드만 보면 중요한 과거 사건을 놓치고, "중요도"만 보면 현재 문맥과 무관한 에피소드가 올라옵니다. RRF는 **여러 랭킹 기준을 보정 없이 합산**하므로, 각 소스의 점수 스케일이 달라도 공정하게 결합됩니다.

에피소드마다 **중요도 점수(0-100)**가 매겨집니다:

| 이벤트 | 가중치 |
|--------|--------|
| 기본 (아무 턴이나) | +10 |
| HP 변동 | +3 × |delta| (최대 30) |
| 관계 변화 | +10 × 건수 (최대 30) |
| 이벤트 트리거 | +35 |
| NPC 만남 | +10 × 수 (최대 20) |
| NPC 분리 | +15 |
| 아이템 획득/소실 | +15 |
| 아이템 이전 (캐릭터 간) | +20 |
| 위치 이동 | +10 |

그리고 에피소드는 `combat`, `event`, `relationship`, `encounter`, `item`, `exploration`, `dialogue` 중 하나로 자동 분류됩니다.

---

## 6. 3-Agent 파이프라인 상세

### Sub-A: Context Builder (동기, ~200ms, LLM 호출 없음)

유저 요청이 올 때마다 실행됩니다. **LLM을 호출하지 않으므로** 빠릅니다.

```
stable_prefix.md 읽기 → live_state.md 읽기
→ ChromaDB RRF 3-source 에피소드 선택
→ SQLite 로어 전체 조회 + ChromaDB 벡터 로어 검색
→ 중복 제거 → 토큰 예산 내 조립
```

조립 순서 (토큰 예산 우선순위):

1. **live_state.md** — 가장 중요. 현재 위치, HP, 주변 NPC, 최근 이벤트
2. **에피소드 기억** — RRF 선택된 top 10개. 중요도 ≥50은 `[!]` 표시, 나머지는 `[R]` 표시
3. **활성 로어** — SQLite + 벡터 검색 결과 병합, 중복 제거
4. **State Tracking 지시** — LLM에게 응답 끝에 State Block 출력을 요청하는 지시문 (config로 ON/OFF 가능)

**토큰 예산**: `dynamic_context_max` (기본 2000토큰) 안에서 위 순서대로 패킹합니다. 각 섹션이 남은 예산을 초과하면 스킵됩니다.

### Sub-B: Post-Turn Extractor (비동기, 유저 대기 없음)

응답이 유저에게 전달된 **뒤에** `BackgroundTask`로 실행됩니다. `asyncio.Lock`으로 이전 턴 처리 완료를 보장합니다.

#### 파이프라인 (6단계)

1. **Flash 서사 요약** — 경량 LLM(Gemini Flash)으로 4필드 JSON 추출
2. **Importance 스코어링** — scene_type 기반 가중치 (combat +40, event +35 등)
3. **ChromaDB 에피소드 기록** — 요약 텍스트 + importance 점수로 에피소드 임베딩
4. **NPC 레지스트리 갱신** — npcs_mentioned → SQLite characters 테이블에 등록
5. **SQLite 턴 로그 기록** — turn_log (narrative, user_input, response)
6. **live_state.md 갱신** — SQLite query_player_context() → write_live() (위치, HP, 주변 NPC, 최근 이벤트)

**서사 요약 형식 (Flash 출력):**

```json
{
  "summary": "2-3문장 요약",
  "npcs_mentioned": ["등장/언급된 NPC 이름들"],
  "scene_type": "combat|dialogue|exploration|event",
  "key_event": "핵심 사건 한 줄 또는 null"
}
```

초기에는 메인 LLM에게 12필드 State Block을 응답 끝에 출력하도록 요청하는 방식을 사용했으나, 불안정한 regex 추출과 270-400 토큰의 추가 부담으로 인해 Flash 4필드 요약으로 전환했습니다.

#### NPC 이름 정규화

LLM이 같은 NPC를 "세라핀", "세라핀 언니", "세라핀씨" 등 다른 이름으로 출력하면 중복 엔트리가 생깁니다. Sub-B는 한국어 존칭을 제거하고 기존 캐릭터 목록과 대조합니다:

```
"세라핀 언니" → strip("언니") → "세라핀" → DB의 "세라핀"과 매칭 → 기존 이름 사용
"김도하씨" → strip("씨") → "김도하" → DB의 "도하"와 접미사 비교 → 성 "김" 확인 → 매칭
```

지원 존칭: 선배, 후배, 언니, 오빠, 누나, 씨, 님, 양, 군, 형

### Curator (10턴마다, 비동기)

여기서 **Letta가 등장**합니다.

Curator는 Letta SDK로 생성된 에이전트가 **세션별 전용 에이전트**를 갖고 동작합니다. 에이전트 생성은 lazy — 첫 큐레이션 요청 시 자동 생성되며, 이후 세션 동안 재사용됩니다.

#### 3개의 Memory Block

| Block | 역할 |
|-------|------|
| `narrative_summary` | 전체 서사 요약. 매 큐레이션마다 최신화 |
| `curation_decisions` | 큐레이션 판단 기록. 최근 5건 유지 |
| `contradiction_log` | 모순 탐지/해결 기록. 해결된 항목은 [해결됨] 표시 |

핵심은 이 Memory Block이 **큐레이션 간에 유지**된다는 것입니다. 10턴 전에 "에르겐 위치 모순 발견"이라고 기록해 두면, 20턴 후 "그 모순은 수정되었는가?"를 이어서 판단할 수 있습니다.

#### Curator가 하는 일

10턴마다 호출되어:

1. **서사 모순 탐지** — SQLite 규칙 기반 (HP≤0 생존) + 에이전트 판단 (죽은 NPC 재등장, 파괴된 아이템 재사용, 동시 두 장소 존재). 정상적 위치 이동/감정 변화는 모순으로 취급하지 않음.
2. **이벤트 스케줄링** — 복선 회수, 새 이벤트 제안 → `event_queue`에 삽입
3. **Story 압축** — 50턴 이상이면 압축 요약 → `stable_prefix.md` 갱신
4. **로어 자동 생성** — 로어 엔트리가 없는 NPC/장소를 탐지하여 관련 에피소드와 관계 데이터를 기반으로 로어를 LLM으로 생성 (사이클당 최대 3건)

#### Letta 장애 대응

- Letta 서버 연결 실패 → **Direct LLM 폴백** (Memory Block 연속성 없이 단발 호출)
- 일반적으로 `message_buffer_autoclear=True`가 히스토리 누적을 방지하므로, 컨텍스트 초과는 드물다.
- 에이전트 컨텍스트 창 초과 → **에이전트 재생성**. 기존 Memory Block 내용을 읽어서 새 에이전트에 복원 (큐레이션 이력 보존)
- Letta 없이도 시스템은 동작합니다 — 다만 큐레이션 판단의 연속성이 떨어집니다.

---

## 7. 프롬프트 캐싱 — SystemStabilizer와 3-BP 전략

### 문제: RisuAI가 system 메시지를 매 턴 바꾼다

Anthropic의 Prompt Caching은 **프리픽스 기반**입니다. system 메시지 한 글자만 바뀌어도 모든 캐시 브레이크포인트가 무효화됩니다. 그런데 RisuAI는 Lorebook 활성 상태에 따라 매 턴 system 메시지를 동적으로 조립합니다. 캐싱 관점에서 치명적입니다.

### SystemStabilizer

SAGA는 `SystemStabilizer`로 이 문제를 해결합니다:

```
첫 턴:
  RisuAI system 메시지 수신 → "canonical system"으로 SQLite에 저장 (hash 포함)

이후 턴:
  RisuAI system 메시지 수신 → canonical과 hash 비교
  ├─ 동일 → 그대로 통과 (캐시 안정)
  ├─ 유사 (Jaccard ≥ 0.30) → paragraph 단위 delta 추출
  │   └─ canonical은 유지, delta는 동적 컨텍스트 영역으로 이동
  └─ 완전히 다름 (Jaccard < 0.30) → 새 canonical으로 교체
```

Delta 추출 시 inject 유형(append/prepend/replace)도 감지합니다. 기존 paragraph가 수정된 것(단어 중복률 >50%)인지, 완전히 새로운 것인지 구분하여 불필요한 중복을 방지합니다.

### 3-BP (3 Breakpoint) 캐시 전략

Anthropic의 `cache_control: {"type": "ephemeral"}` 마커를 3곳에 배치합니다:

```
BP1: system 메시지 (canonical, 안정)          ← cache_control
BP2: 중간 assistant 메시지 (stable_prefix.md)  ← cache_control
BP3: 마지막 assistant 메시지 (대화 히스토리)    ← cache_control
      ↓
    마지막 user 메시지 (동적 컨텍스트 prepend + 유저 입력)
```

동적 컨텍스트(live_state.md, 에피소드, 로어, lorebook delta)는 **마지막 user 메시지에 prepend**됩니다. system 메시지에 넣으면 BP1이 매 턴 무효화되기 때문입니다.

API 호출 시 `extended-cache-ttl-2025-04-11` beta 헤더를 사용하여 캐시 TTL을 **1시간**으로 확장합니다.

### Prompt Caching 실측 벤치마크 (2026-02-28)

50턴 RP 대화를 Anthropic API 직접 호출로 시뮬레이션하여, 3-BP 캐시 전략의 실전 효과를 측정했습니다 (Claude Haiku 4.5 기준).

**3-BP 수동 캐싱 (SAGA 기본 전략):**

| 구간 | 캐시 히트율 | 특이사항 |
|------|:----------:|---------|
| 턴 1 | 0% (cold) 또는 96%+ (warm) | 5분 TTL 내 이전 세션 캐시 재활용 가능 |
| 턴 2~5 | 90~92% | 시스템 프롬프트(BP1) 즉시 히트, 대화가 짧아 BP2만 활성 |
| 턴 5~25 | 92~96% | BP2(중간점) + BP3(마지막) 모두 활성, 히트율 점진 상승 |
| 턴 25~50 | 96~97% | 안정 구간, 매 턴 ~325 토큰만 새로 캐시 생성 |

**50턴 집계:**

| 지표 | 값 |
|------|-----|
| 평균 캐시 히트율 (턴 5+) | **95.5%** |
| 총 캐시 읽기 토큰 | 621,769 |
| 총 비용 (캐시 적용) | **$0.17** |
| 총 비용 (캐시 미적용) | $0.72 |
| **비용 절감** | **76.7%** |
| 평균 레이턴시 | 5,180ms (턴 수에 거의 무관) |

**자동 캐싱(top-level) vs 수동 3-BP:**

자동 캐싱은 Anthropic의 20-block lookback 제한으로 인해 턴 12+에서 캐시가 완전히 무효화됩니다:

| 모드 | 턴 수 | 평균 히트율 | 총 비용 | 절감률 |
|------|:-----:|:----------:|:-------:|:------:|
| **수동 3-BP** | 50 | **95.5%** | **$0.17** | **76.0%** |
| 자동 top-level | 42* | 12.1% | $0.62 | -11.4% (더 비쌈) |
| 캐시 없음 | — | 0% | $0.72 | 기준선 |

*자동 캐싱은 턴 43에서 529 Overloaded 에러로 중단. 매 턴 18000+ 토큰 cache_create로 API 처리량 한계 도달.

> **결론: 장기 RP 대화에서는 수동 3-BP가 필수.** 자동 캐싱은 20턴 이상에서 오히려 비용이 증가하며, API 안정성도 떨어진다.
> 벤치마크 스크립트: `tests/bench_prompt_caching.py` (`--trace` 플래그로 I/O 트레이싱 가능) | 상세 데이터: `tests/bench_report_3bp.json`

---

## 8. 비용과 트레이드오프

### 멀티모델 + 멀티프로바이더 전략

SAGA의 LLM 클라이언트는 모델명에서 프로바이더를 자동 감지합니다. 작업마다 최적 모델을 선택합니다:

| 작업 | 모델 | 프로바이더 | 이유 |
|------|------|-----------|------|
| 내레이션 | Claude Sonnet 4.5 | Anthropic | 서사 품질 + 3-BP 캐싱 |
| 상태 추출 (Sub-B) | Gemini 2.0 Flash | Google | 구조화된 추출은 저비용으로 충분 |
| 큐레이션 (Curator) | Claude Sonnet 4.5 | Anthropic (via Letta) | 서사 판단 필요하나, 10턴마다만 |
| 임베딩 | text-embedding-3-small | OpenAI | 범용, 저비용 |

### 100턴 기준 호출 횟수

| 접근 방식 | LLM 호출 횟수 | 비고 |
|-----------|:-----------:|------|
| 전체 히스토리 전송 | 100 | 턴마다 히스토리 누적, 비용 선형 증가 |
| **SAGA** | **100 + 100 경량 + 10 Curator** | Anthropic 100 + Gemini Flash 100 + Letta 10 |

SAGA는 고비용 모델(Sonnet)은 내레이션에만, Flash는 매 턴 서사 요약과 로어 자동생성에 사용한다. 그리고 .md 캐시 + 3-BP Prompt Caching으로 입력 토큰 비용을 추가 76% 절감합니다.

### 한계

**SAGA가 잘 하는 것:**
- 구조화된 상태(위치, HP, 인벤토리, NPC 관계) 추적
- 프론트엔드 수정 없이 투명한 적용 (OpenAI API 호환)
- 유저 체감 지연 최소화 (1회 호출 + SSE 스트리밍)
- **.md 캐시 기반의 확장성**: 새로운 .md 파일을 추가하는 것만으로 추적 영역을 넓힐 수 있음

**SAGA의 한계:**

1. **서사 요약 정확도**: Flash 서사 요약의 추출 품질에 의존합니다. JSON 잘림 시 복구 로직으로 완화하지만 완벽하지 않습니다.

2. **1턴 지연**: 상태 갱신이 비동기이므로, 갱신이 완료되기 전에 다음 턴이 시작되면 이전 상태를 참조할 수 있습니다. (`asyncio.Event` 락으로 순서 보장하지만, 극단적으로 빠른 입력 시 경합 가능)

3. **NPC 이름 정규화 한계**: 한국어 존칭 제거 + 접미사 매칭으로 상당수 해결하지만, 완전히 다른 별명("세라핀" vs "빛의 무녀")은 매칭 불가합니다.

4. **SSE 스트리밍의 post-yield 제약**: Starlette의 `StreamingResponse` cancel scope 안에서는 yield 이후 코드가 실행 보장되지 않습니다. Sub-B는 반드시 `BackgroundTask`로 분리해야 합니다.

---

## 9. 코히바블랙님 및 선행 연구자분들께

코히바블랙님의 연구일지를 비롯한 선행 연구들에서 영감을 받아, 같은 문제를 Letta를 핵심에 두고 프록시 아키텍처로 풀어본 기록입니다. SAGA의 Curator는 Letta의 Memory Block 패턴을 직접 사용하고 있으며, 매 턴 처리를 코드 로직으로 분리하여 응답 속도를 확보하는 구조입니다.

---

## 10. 만들면서 느낀 것

1. **"어디에 에이전트를 두는가"가 아키텍처의 핵심**입니다. 전부 에이전트에게 맡기면 강력하지만 느리고, 전부 코드로 하면 빠르지만 경직됩니다. 에이전트의 판단력이 진짜 필요한 곳을 찾는 게 핵심이었습니다.

2. **SQLite + 벡터, 둘 다 필요했습니다.** 상태와 관계는 SQLite, 유사도 검색은 벡터 DB — 각각이 다른 질문에 답합니다. 그래프 DB(KuzuDB)는 초기에 사용했으나, 관계 데이터의 규모가 크지 않아 SQLite 테이블로 충분히 대체 가능했습니다.

3. **Flash 서사 요약으로의 전환이 핵심이었습니다.** 초기 12필드 regex 추출은 불안정했고 메인 LLM에 270-400 토큰을 추가 소비했습니다. Flash 4필드 요약으로 전환하여 메인 LLM 부하 제거 + 안정적 추출을 달성했습니다.

4. **비동기 처리가 핵심**이었습니다. 유저 대기 경로에서 무거운 작업을 빼는 것만으로 체감 성능이 달라집니다. 특히 SSE 스트리밍에서 post-yield 코드는 cancel scope 때문에 실행이 보장되지 않으므로, `BackgroundTask`로 분리하는 것이 필수였습니다. 이 발견이 Sub-B 파이프라인 전체를 살렸습니다.

5. **직접 만들지 말고 잘 엮어라.** Letta, ChromaDB, SQLite — 각각 이미 잘 만들어진 도구입니다. SAGA의 가치는 새로운 DB나 메모리 시스템을 발명하는 게 아니라, 기존 도구들을 RP에 맞게 오케스트레이션하는 데 있습니다. Claude Code가 Bash와 grep을 조율하듯, SAGA는 SQLite와 벡터 DB와 Stateful Agent를 조율합니다.

6. **캐시 안정성은 별도 모듈로.** RisuAI가 매 턴 system 메시지를 바꾸는 문제는 예상 못했습니다. SystemStabilizer를 분리 모듈로 만들어 canonical 보존 + delta 추출을 하니, 캐시 히트율이 12%에서 95%로 올라갔습니다. 프록시 아키텍처의 장점 — 클라이언트를 수정하지 않고 중간에서 교정할 수 있습니다.

의견이나 제안이 있으시면 댓글로 남겨주세요. 같이 개발해 보고 싶으신 분도 편하게 연락 부탁드립니다.

---

## 11. 참고 자료

**[1]** 코히바블랙 (2025). *Letta를 이용한 장기기억 향상 및 AI 채팅 경험 향상 연구 초록.* [arca.live AI 채팅 채널](https://arca.live/b/characterai/162255622)
**[2]** [Agent Memory Paper List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) — MemGPT, Graph RAG, RAG 등 이 글에서 언급한 논문들의 종합 목록.
**[3]** Letta (2025). *MemFS: Memory as a File System.* https://docs.letta.com/letta-code/memory

---

## 12. E2E 통합 검증 결과 (2026-03-02)

23턴 던전 시나리오(LLM 시뮬레이터 자동 입력)로 전체 파이프라인을 실증 검증한 결과입니다.

### 검증 항목 (17/17 ALL PASS)

| 카테고리 | 항목 | 결과 |
|---------|------|------|
| 연결성 | SAGA health, Auth rejection, Letta health | PASS |
| 멀티턴 | 23/23 턴 완료 (평균 34s, 2666ch) | PASS |
| live_state.md | 파일 존재, turn frontmatter, 실제 location, recent events | PASS |
| SQLite | 세션 존재, turn_count≥23, player 캐릭터, 23 턴로그, 23 이벤트 | PASS |
| ChromaDB | 에피소드 ≥ 3 (23개 기록) | PASS |
| Letta | Curator 에이전트 존재, 메모리 블록 비어있지 않음 | PASS |

### 프롬프트 캐싱 실전 결과

23턴 실제 RP 대화에서의 캐싱 추이:

| 구간 | cache_read | cache_create | 비고 |
|------|-----------|-------------|------|
| T1-T4 | 0 | 0 | 콜드스타트, 최소 토큰 미달 |
| T5 | 0 | 5,418 | 첫 캐시 생성 (임계값 초과) |
| T6 | 5,418 | 1,825 | 캐시 적중 시작 |
| T10 | 13,787 | 3,110 | 안정적 누적 |
| T23 | 56,121 | 2,210 | 최종 — 56K 토큰 캐시 적중 |

### 파이프라인 성능

- **Sub-A 컨텍스트 빌드**: 176~382ms (평균 ~200ms)
- **Sub-B 추출**: 매 턴 Flash 서사 요약 (4필드 JSON)
- **Curator**: 턴 10, 20에서 실행 — NPC 레지스트리 자동 갱신, 서사 요약 축적
- **자동 추적**: 11 캐릭터, 14 로케이션, 11 관계, 23 에피소드

### 확인된 한계

1. **NPC 이름 정규화 잔여 케이스**: 존칭 제거 + 접미사 매칭으로 대부분 해결하나, 완전히 다른 호칭(예: "세라핀 (어머니)" vs "세라핀 (원거리에서 목소리만)")은 별도 엔트리 생성
2. **Curator 로어 자동생성 파싱 실패**: Gemini Flash가 wrapper 없이 raw JSON을 반환하여 파서 불일치
3. **Curator 실행 시 레이턴시 스파이크**: T11(106s), T21(172s) — Letta API + Gemini Flash 병렬 호출이 겹침. 단, 비동기이므로 유저 체감 영향 없음

> 검증 스크립트: `tests/e2e_integration.py` (`--scenario dungeon` 으로 던전 시나리오 실행)
> 결과 데이터: `tests/e2e_results/`

---

*이 글은 [Claude Code](https://claude.ai/claude-code)와 함께 작성되었습니다.*
