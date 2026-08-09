"""eval v2 설정 — env 오버라이드 + 벤치 상수. 최하층: eval 내부 import 금지."""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = os.environ.get("DREAMING_EVAL_PROXY", "http://127.0.0.1:8790")
UPSTREAM = os.environ.get("DREAMING_EVAL_UPSTREAM",
                          "https://openrouter.ai/api/v1")
MODEL = os.environ.get("DREAMING_EVAL_MODEL", "deepseek/deepseek-v4-pro")
JUDGE_MODEL = os.environ.get("DREAMING_EVAL_JUDGE",
                             "anthropic/claude-sonnet-4.5")
DIRECTOR_MODEL = os.environ.get("DREAMING_EVAL_DIRECTOR",
                                "google/gemini-3-flash-preview")
HYPA_EXPORT = os.environ.get(
    "DREAMING_EVAL_HYPA_EXPORT",
    str(pathlib.Path.home() / "Downloads" / "뮈토스6.2"
        / "🏺뮈토스 프롬프트 하이파" / "hypaV3_export_뮈토스 하이파 V5.json"))
TURNS = 80
PROBE_EVERY = 10              # 이 간격마다 발화 하나가 과거를 슬며시 되짚는다
# maxContext — RisuAI의 단일 토큰 풀 (index.svelte.ts:614-618). trim·hypa 공용.
# 원 프리셋 200K 대비 4.4× 축소라 hypa의 memoryTokens 선점도 78,000 → 17,550이다.
MAX_CONTEXT = 45000
UPDATE_EVENTS = (12, 28)      # 지식갱신 강제 턴
# NPC 등장은 당채련 하나로 고정 — 런 간 같은 사건 축이라 비교 가능하다.
# 로어 7엔트리는 항상 주입되므로(키워드 게이팅 없음) 활성화 문제가 아니라
# 장면 유도 문제다: 이름을 직접 불러 나레이터가 꺼내게 한다. 이 이름이
# 디렉터 카드 지식 선취 금지의 유일한 예외. 파일럿 50/80턴 NPC 0명 실측.
NPC_NAME = "당채련"
NPC_EVENT_TURN = 40           # 0-기준 (표시 T41)에 첫 유도
NPC_EVENT_RETRY = 44          # 이때까지 미등장이면 다시 유도
# 캡처에서 RisuAI가 실제로 보낸 값이 4000이다 (capture-mythos req-001).
# 실측 완성 평균은 771토큰이라 캡이 물리지 않는다 — 절단은 기억 실패로
# 오인되는 교란이라 finish_reason을 턴마다 기록해 0%임을 증명한다.
MAX_TOKENS = 4000
# 프로바이더가 거부(NSFW 등)를 반복하면 리롤 비용만 태운다 — 런 전체
# 누적 리롤이 이 값에 닿으면 결과를 저장하고 런을 중단한다.
MAX_RUN_REROLLS = 10

# 확정 토글: RP 모드·한국어·성인 지침 ON·중립 렌더링 프리필 ON, 나머지 기본
# select은 옵션 인덱스 문자열: response_language 1=🇰🇷 한국어, execution_mode 0=💬 RP.
# 나머지는 불리언. 미설정 전역변수는 "null"이라(preset2wire.UNSET) 실행 모드는
# 반드시 명시해야 tis:: 분기가 걸린다.
TOGGLES = {"mythos_response_language": "1",           # 🇰🇷 한국어
           "mythos_execution_mode": "0",              # 💬 RP
           "mythos_user_persona_usage": "0",          # 🙋 사용 — 안 켜면 슬롯이 통째로 빈다
           "mythos_bot_structure": "0",               # 💬 캐릭터 중심
           # select 토글은 미설정이면 사이드바 SelectInput이 바인딩하며 첫 옵션
           # 인덱스를 써 넣는다. 프리셋의 templateDefaultVariables는 여기 안
           # 먹는다 — getChatVar 전용이고 tis는 getGlobalChatVar를 본다
           # (chatVar.svelte.ts:15 vs 35, parser.svelte.ts:1284).
           "mythos_user_character_authorship": "0",   # 🛡️ 보호 — 캡처 req-005 확인
           "mythos_input_authority": "0",             # 🔨 사실 확정
           "mythos_prose_register": "0",              # 🤷 미지정
           "mythos_narrative_pov": "0",               # 🤷 자율
           "mythos_narrative_pacing": "0",            # 🤷 자율
           "mythos_response_length_band": "0",        # 🤷 미지정
           "mythos_size_scenario": "0",               # 🤷 미지정
           "mythos_genre_ero": "1",
           "mythos_mature_content_guidance": "1",
           "mythos_domain_neutral_rendering_prefill": "1"}
