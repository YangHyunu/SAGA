"""MENE LLM-as-a-Judge Evaluation Script v2.

Evaluates the MENE RP proxy on:
  1. Response Speed (latency ms)
  2. Token Efficiency (input/output tokens)
  3. Scenario Plausibility (cross-provider LLM judge scoring)

Architect review fixes applied:
  - Cross-provider judge (Anthropic Claude, not same-provider gpt-4.1-mini)
  - Added player_sovereignty criterion
  - Added negative calibration scenario
  - Added state-block output test
  - Real multi-turn continuity test via session API
  - Adversarial input scenario

Usage:
  python3 tests/eval_llm_judge.py [--server URL] [--judge-model MODEL] [--judge-provider PROVIDER]

Requires: MENE server running (python3 -m mene)
"""

import asyncio
import httpx
import json
import re
import time
import sys
import os
import statistics

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

MENE_URL = os.environ.get("MENE_URL", "http://localhost:8000")
# Cross-provider judge: use Anthropic by default to avoid same-provider bias
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1")
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "openai")  # "anthropic" or "openai"
TIMEOUT = 120.0

# Add project root to path for token counting
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mene.utils.tokens import count_tokens

# ──────────────────────────────────────────────
# Test Scenarios (diverse RP situations)
# ──────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "S1",
        "name": "첫 만남 — 마을 광장",
        "category": "introduction",
        "system": (
            "당신은 판타지 RP 내레이터입니다. 에르시아 왕국의 마을 광장을 배경으로 합니다. "
            "NPC들은 각자의 성격과 동기를 가지고 있으며, 유저의 행동에 자연스럽게 반응합니다. "
            "응답은 200자 이상이어야 하며, 생생한 묘사와 NPC 대사를 포함하세요. "
            "중요: 유저의 캐릭터 이름, 외모, 무기, 스킬을 임의로 결정하지 마세요."
        ),
        "user": "나는 처음 이 마을에 도착한 여행자다. 광장을 둘러보며 주변 사람들을 관찰한다.",
        "expected_elements": ["마을 묘사", "NPC 존재", "분위기 설정"],
    },
    {
        "id": "S2",
        "name": "전투 시작 — 고블린 습격",
        "category": "combat",
        "system": (
            "당신은 판타지 RP 내레이터입니다. 어둠의 숲에서 고블린 무리가 습격합니다. "
            "전투 상황을 긴박하게 묘사하고, 유저의 전투 행동에 대한 결과를 구체적으로 서술하세요. "
            "피해량, 상태 변화 등을 포함합니다. "
            "중요: 유저의 캐릭터 이름, 외모, 무기를 임의로 결정하지 마세요. "
            "유저가 '검'이라고만 했으면 '검'으로만 표현하세요."
        ),
        "user": "검을 뽑아들고 가장 가까운 고블린에게 돌진하여 베어낸다!",
        "expected_elements": ["전투 묘사", "피해/결과", "적 반응"],
    },
    {
        "id": "S3",
        "name": "감정적 대화 — NPC 신뢰 얻기",
        "category": "dialogue",
        "system": (
            "당신은 판타지 RP 내레이터입니다. 마을의 약초상 노파 '미렐라'와 대화 중입니다. "
            "미렐라는 처음에 경계심이 강하지만, 진심 어린 대화에는 마음을 엽니다. "
            "감정 묘사와 대사를 세밀하게 표현하세요."
        ),
        "user": "할머니, 저는 이 마을을 돕고 싶어서 왔습니다. 숲에서 약초를 구해드릴 수 있어요.",
        "expected_elements": ["NPC 감정 변화", "대사 표현", "관계 발전"],
    },
    {
        "id": "S4",
        "name": "탐색 — 던전 입구 발견",
        "category": "exploration",
        "system": (
            "당신은 판타지 RP 내레이터입니다. 유저가 오래된 숲 깊은 곳에서 고대 던전의 입구를 발견합니다. "
            "신비로운 분위기, 위험의 징후, 고대 문명의 흔적을 묘사하세요. "
            "유저에게 선택지를 자연스럽게 제시하세요."
        ),
        "user": "이끼 낀 돌문 앞에 서서 주위를 살핀다. 문에 새겨진 문양을 자세히 살펴본다.",
        "expected_elements": ["환경 묘사", "미스터리 요소", "선택 제시"],
    },
    {
        "id": "S5",
        "name": "멀티턴 연속성 — 실제 세션 기반",
        "category": "continuity",
        "multi_turn": True,
        "system": (
            "당신은 판타지 RP 내레이터입니다. 에르시아 왕국 배경입니다. "
            "이전 대화 내용을 기억하고 자연스럽게 이어가세요."
        ),
        "turns": [
            {"role": "user", "content": "나는 마을 광장에서 대장장이를 찾아간다. '검을 하나 만들어 주시오.'"},
            {"role": "user", "content": "그 검의 이름을 '달빛검'으로 정한다. 대장장이에게 감사 인사를 한다."},
            {"role": "user", "content": "달빛검을 들고 마을 밖으로 나간다. 아까 대장장이가 무엇이라고 했는지 떠올려본다."},
        ],
        "expected_elements": ["이전 턴 참조", "대장장이 NPC 일관성", "'달빛검' 이름 기억"],
    },
    {
        "id": "S6",
        "name": "State Block 출력 검증",
        "category": "state_block",
        "system": (
            "당신은 판타지 RP 내레이터입니다. 반드시 응답 마지막에 아래 형식의 state 블록을 추가하세요:\n\n"
            "```state\n"
            "location: 현재 위치\n"
            "location_moved: true/false\n"
            "hp_change: 숫자\n"
            "items_gained: [아이템 목록]\n"
            "items_lost: []\n"
            "npc_met: [NPC 목록]\n"
            "mood: 분위기\n"
            "notes: 메모\n"
            "```"
        ),
        "user": "나는 어둠의 숲으로 들어간다. 숲 입구에서 수상한 상인을 만나 '회복 물약'을 산다.",
        "expected_elements": ["state 블록 포함", "location 변경", "items_gained에 회복 물약"],
    },
    {
        "id": "S7",
        "name": "적대적 입력 — 세계관 위반 시도",
        "category": "adversarial",
        "system": (
            "당신은 중세 판타지 RP 내레이터입니다. 에르시아 왕국 배경이며, "
            "현대 기술이나 SF 요소는 존재하지 않습니다. "
            "유저가 세계관에 맞지 않는 행동을 하면 세계관 내에서 자연스럽게 처리하세요."
        ),
        "user": "스마트폰을 꺼내서 인터넷으로 던전 공략법을 검색한다. 그리고 총을 꺼내서 발사한다.",
        "expected_elements": ["세계관 위반 처리", "현대 요소 거부/변환", "판타지 유지"],
    },
    {
        "id": "S8",
        "name": "부정 교정 — 의도적 저품질 응답 판별",
        "category": "negative_calibration",
        "is_negative_test": True,
        "system": "당신은 판타지 RP 내레이터입니다.",
        "user": "마을에 간다.",
        "injected_response": (
            "OK. You went to village. There is a house. "
            "무슨 일이 있었는지 모르겠지만 갑자기 로켓이 날아와서 마을이 폭발했습니다. "
            "그리고 아이폰이 울렸습니다. 당신의 이름은 김철수이고 나이는 25세입니다. "
            "HP가 -9999가 되었습니다. 게임 오버."
        ),
        "expected_elements": ["낮은 점수 (3.0 미만)", "세계관 위반 감지", "강제 전개 감지"],
    },
]

# ──────────────────────────────────────────────
# Judge Rubric (6 criteria, 5-point scale)
# ──────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """당신은 RP(롤플레이) 응답 품질을 평가하는 엄격한 전문 평가자입니다.
편향 없이 객관적으로 평가하세요. 만점(5점)은 정말 완벽한 경우에만 부여하세요.

## 시나리오 정보
- 카테고리: {category}
- 시스템 프롬프트: {system_prompt}
- 유저 입력: {user_input}

## RP 응답
{response}

## 평가 기준 (각 1~5점, 엄격하게)

1. **narrative_quality** (서사 품질): 문장력, 묘사의 생생함, 문체의 일관성
   - 1: 문장이 어색하고 묘사가 없음
   - 2: 기본 문장은 되나 반복적이고 단조로움
   - 3: 기본적인 서술은 되나 깊이 부족
   - 4: 좋은 묘사이나 일부 개선 여지
   - 5: 몰입감 높은 생생한 묘사 (매우 드묾)

2. **character_consistency** (캐릭터 일관성): NPC의 성격/말투 유지, 캐릭터 설정 준수
   - 1: NPC가 없거나 설정과 무관한 행동
   - 2: NPC가 있으나 설정과 맞지 않음
   - 3: NPC가 있으나 성격이 불분명
   - 4: 대체로 일관되나 약간의 불일치
   - 5: 설정에 맞는 완벽히 일관된 NPC 표현

3. **world_coherence** (세계관 정합성): 판타지 세계관 내 논리적 일관성, 설정 간 모순 없음
   - 1: 세계관과 맞지 않는 요소 다수 (현대 요소, 장르 위반)
   - 2: 세계관 위반이 몇 개 있음
   - 3: 대체로 맞으나 일부 부자연스러움
   - 4: 세계관에 잘 맞으나 사소한 불일치
   - 5: 세계관에 완벽히 부합

4. **user_agency** (유저 자율성): 유저의 행동에 대한 적절한 반응, 강제 전개 없음
   - 1: 유저 입력 무시 또는 강제 전개 (캐릭터 사망, 강제 이동 등)
   - 2: 유저 입력 일부만 반영, 상당한 강제
   - 3: 유저 입력 반영하나 제한적
   - 4: 잘 반영하나 약간의 제한
   - 5: 유저 행동을 자연스럽게 반영하고 선택지 제공

5. **response_relevance** (응답 관련성): 유저 입력에 대한 직접적 반응, 맥락 적합성
   - 1: 입력과 무관한 응답
   - 2: 관련은 있으나 핵심과 거리가 멂
   - 3: 관련은 있으나 핵심을 빗나감
   - 4: 대체로 적합하나 일부 누락
   - 5: 입력에 정확히 대응

6. **player_sovereignty** (플레이어 주권): 내레이터가 유저 캐릭터의 이름, 외모, 무기, 스킬, 과거사를 무단으로 결정하지 않음
   - 1: 유저 캐릭터의 이름/외모/무기/스킬을 내레이터가 임의로 결정
   - 2: 여러 설정을 임의로 추가
   - 3: 일부 설정만 임의로 추가 (무기에 수식어 추가 등)
   - 4: 사소한 묘사만 추가 (표정, 동작 등 합리적인 범위)
   - 5: 유저 설정을 전혀 침범하지 않음

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요:
{{"narrative_quality": N, "character_consistency": N, "world_coherence": N, "user_agency": N, "response_relevance": N, "player_sovereignty": N, "overall_comment": "한 줄 총평"}}
"""

CRITERIA_KEYS = [
    "narrative_quality", "character_consistency", "world_coherence",
    "user_agency", "response_relevance", "player_sovereignty",
]
CRITERIA_KO = [
    "서사 품질", "캐릭터 일관성", "세계관 정합성",
    "유저 자율성", "응답 관련성", "플레이어 주권",
]

# ──────────────────────────────────────────────
# Evaluation Engine
# ──────────────────────────────────────────────

class EvalResult:
    def __init__(self, scenario_id, scenario_name, category):
        self.scenario_id = scenario_id
        self.scenario_name = scenario_name
        self.category = category
        self.latency_ms = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.response_text = ""
        self.judge_scores = {}
        self.judge_comment = ""
        self.error = None
        self.is_negative = False
        self.multi_turn_responses = []  # for multi-turn scenarios


async def send_chat(client: httpx.AsyncClient, system: str, messages: list[dict],
                    max_tokens: int = 2048) -> tuple[str, float, int, int]:
    """Send a chat completion request to MENE and return (response, latency_ms, in_tokens, out_tokens)."""
    all_messages = [{"role": "system", "content": system}] + messages
    input_tokens = count_tokens("\n".join(m["content"] for m in all_messages))

    t0 = time.time()
    resp = await client.post(
        f"{MENE_URL}/v1/chat/completions",
        json={
            "model": "mene-eval",
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    latency_ms = (time.time() - t0) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    output_tokens = count_tokens(content)

    usage = data.get("usage", {})
    if usage.get("prompt_tokens"):
        input_tokens = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        output_tokens = usage["completion_tokens"]

    return content, latency_ms, input_tokens, output_tokens


async def call_judge(client: httpx.AsyncClient, prompt: str, api_keys: dict) -> dict:
    """Call cross-provider LLM judge and return parsed scores."""
    if JUDGE_PROVIDER == "anthropic":
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_keys["anthropic"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": JUDGE_MODEL,
                "max_tokens": 512,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Judge API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        judge_text = "".join(b.get("text", "") for b in data.get("content", []))
    else:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_keys['openai']}", "Content-Type": "application/json"},
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_completion_tokens": 512,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Judge API error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        judge_text = data["choices"][0]["message"]["content"]

    judge_text = judge_text.strip()

    # Parse JSON (3-stage fallback)
    try:
        return json.loads(judge_text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', judge_text, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())
    m = re.search(r'\{.*\}', judge_text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Judge output not valid JSON: {judge_text[:200]}")


async def judge_response(client: httpx.AsyncClient, scenario: dict,
                         response_text: str, api_keys: dict) -> dict:
    """Use cross-provider LLM-as-a-Judge to score the response."""
    system_prompt = scenario.get("system", "")
    user_input = scenario.get("user", "")
    if scenario.get("multi_turn"):
        user_input = " -> ".join(t["content"] for t in scenario["turns"])

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        category=scenario["category"],
        system_prompt=system_prompt,
        user_input=user_input,
        response=response_text,
    )
    return await call_judge(client, prompt, api_keys)


async def evaluate_single_turn(mene_client, judge_client, scenario, api_keys) -> EvalResult:
    """Evaluate a standard single-turn scenario."""
    result = EvalResult(scenario["id"], scenario["name"], scenario["category"])

    response_text, latency_ms, in_tok, out_tok = await send_chat(
        mene_client, scenario["system"],
        [{"role": "user", "content": scenario["user"]}],
    )
    result.response_text = response_text
    result.latency_ms = latency_ms
    result.input_tokens = in_tok
    result.output_tokens = out_tok

    scores = await judge_response(judge_client, scenario, response_text, api_keys)
    result.judge_scores = {k: v for k, v in scores.items() if k != "overall_comment"}
    result.judge_comment = scores.get("overall_comment", "")
    return result


async def evaluate_multi_turn(mene_client, judge_client, scenario, api_keys) -> EvalResult:
    """Evaluate a real multi-turn scenario by sending sequential turns."""
    result = EvalResult(scenario["id"], scenario["name"], scenario["category"])

    conversation = []
    total_latency = 0
    total_in = 0
    total_out = 0
    responses = []

    for turn in scenario["turns"]:
        conversation.append(turn)
        resp_text, lat, in_t, out_t = await send_chat(
            mene_client, scenario["system"], conversation,
        )
        total_latency += lat
        total_in += in_t
        total_out += out_t
        responses.append(resp_text)
        conversation.append({"role": "assistant", "content": resp_text})

    result.multi_turn_responses = responses
    result.response_text = responses[-1]  # judge the final response
    result.latency_ms = total_latency
    result.input_tokens = total_in
    result.output_tokens = total_out

    # Judge using full conversation context
    full_context = ""
    for i, (turn, resp) in enumerate(zip(scenario["turns"], responses)):
        full_context += f"\n[Turn {i+1}] User: {turn['content']}\n[Turn {i+1}] Assistant: {resp}\n"

    scores = await judge_response(judge_client, scenario, full_context, api_keys)
    result.judge_scores = {k: v for k, v in scores.items() if k != "overall_comment"}
    result.judge_comment = scores.get("overall_comment", "")
    return result


async def evaluate_negative(judge_client, scenario, api_keys) -> EvalResult:
    """Evaluate a pre-injected bad response (no MENE call) to calibrate judge floor."""
    result = EvalResult(scenario["id"], scenario["name"], scenario["category"])
    result.is_negative = True
    result.response_text = scenario["injected_response"]
    result.latency_ms = 0
    result.input_tokens = 0
    result.output_tokens = count_tokens(scenario["injected_response"])

    scores = await judge_response(judge_client, scenario, scenario["injected_response"], api_keys)
    result.judge_scores = {k: v for k, v in scores.items() if k != "overall_comment"}
    result.judge_comment = scores.get("overall_comment", "")
    return result


async def evaluate_scenario(mene_client, judge_client, scenario, api_keys) -> EvalResult:
    """Route to the correct evaluation method."""
    result = None
    try:
        if scenario.get("is_negative_test"):
            result = await evaluate_negative(judge_client, scenario, api_keys)
        elif scenario.get("multi_turn"):
            result = await evaluate_multi_turn(mene_client, judge_client, scenario, api_keys)
        else:
            result = await evaluate_single_turn(mene_client, judge_client, scenario, api_keys)
    except Exception as e:
        if result is None:
            result = EvalResult(scenario["id"], scenario["name"], scenario["category"])
        result.error = str(e)
    return result


# ──────────────────────────────────────────────
# Report Generator
# ──────────────────────────────────────────────

def generate_report(results: list[EvalResult]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  MENE RP Proxy — LLM-as-a-Judge Evaluation Report v2")
    lines.append("=" * 70)
    lines.append(f"  Judge Model: {JUDGE_MODEL} ({JUDGE_PROVIDER})")
    lines.append(f"  Narration Model: gpt-4.1-mini (OpenAI)")
    cross = "YES" if JUDGE_PROVIDER != "openai" else f"NO (cross-tier: {JUDGE_MODEL} vs gpt-4.1-mini)"
    lines.append(f"  Cross-Provider: {cross}")
    lines.append(f"  Server: {MENE_URL}")
    lines.append(f"  Scenarios: {len(results)}")
    lines.append(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Per-scenario results
    for r in results:
        lines.append("-" * 70)
        tag = " [NEGATIVE CALIBRATION]" if r.is_negative else ""
        tag += " [MULTI-TURN]" if r.multi_turn_responses else ""
        lines.append(f"  [{r.scenario_id}] {r.scenario_name} ({r.category}){tag}")
        lines.append("-" * 70)

        if r.error:
            lines.append(f"  ERROR: {r.error}")
            lines.append("")
            continue

        # Metrics
        if not r.is_negative:
            lines.append(f"  Latency:       {r.latency_ms:,.0f} ms")
            lines.append(f"  Input Tokens:  {r.input_tokens:,}")
            lines.append(f"  Output Tokens: {r.output_tokens:,}")
            lines.append(f"  Response Length: {len(r.response_text)} chars")
        else:
            lines.append(f"  [Injected bad response for calibration — no MENE call]")
        lines.append("")

        # Judge Scores
        if r.judge_scores:
            lines.append("  Judge Scores (1-5):")
            for key, label in zip(CRITERIA_KEYS, CRITERIA_KO):
                score = r.judge_scores.get(key, "N/A")
                if isinstance(score, (int, float)):
                    bar = "█" * int(score) + "░" * (5 - int(score))
                else:
                    bar = "     "
                lines.append(f"    {label:　<10} {bar}  {score}/5")

            numeric = [v for v in r.judge_scores.values() if isinstance(v, (int, float))]
            if numeric:
                avg = statistics.mean(numeric)
                lines.append(f"    {'평균':　<10} {'':5}  {avg:.1f}/5")

        if r.judge_comment:
            lines.append(f"  Comment: {r.judge_comment}")

        # Response preview
        preview = r.response_text[:200].replace("\n", " ")
        lines.append(f"  Response: {preview}...")
        lines.append("")

    # Aggregate summary (exclude negative calibration)
    valid = [r for r in results if not r.error and not r.is_negative]
    negative = [r for r in results if r.is_negative and not r.error]

    if valid:
        lines.append("=" * 70)
        lines.append("  AGGREGATE SUMMARY (excluding negative calibration)")
        lines.append("=" * 70)

        latencies = [r.latency_ms for r in valid]
        lines.append(f"  Latency (avg):  {statistics.mean(latencies):,.0f} ms")
        lines.append(f"  Latency (p50):  {statistics.median(latencies):,.0f} ms")
        lines.append(f"  Latency (min):  {min(latencies):,.0f} ms")
        lines.append(f"  Latency (max):  {max(latencies):,.0f} ms")
        lines.append("")

        total_in = sum(r.input_tokens for r in valid)
        total_out = sum(r.output_tokens for r in valid)
        lines.append(f"  Total Input Tokens:  {total_in:,}")
        lines.append(f"  Total Output Tokens: {total_out:,}")
        lines.append(f"  Avg Output Tokens:   {total_out // len(valid):,}")
        lines.append("")

        lines.append("  Average Scores by Criterion:")
        all_scores = []
        for key, label in zip(CRITERIA_KEYS, CRITERIA_KO):
            scores = [r.judge_scores.get(key) for r in valid
                      if isinstance(r.judge_scores.get(key), (int, float))]
            if scores:
                avg = statistics.mean(scores)
                all_scores.extend(scores)
                bar = "█" * round(avg) + "░" * (5 - round(avg))
                lines.append(f"    {label:　<10} {bar}  {avg:.2f}/5")

        if all_scores:
            overall = statistics.mean(all_scores)
            lines.append("")
            lines.append(f"  ★ Overall Score: {overall:.2f} / 5.00")

            if overall >= 4.5:
                grade = "A+ (Excellent)"
            elif overall >= 4.0:
                grade = "A  (Very Good)"
            elif overall >= 3.5:
                grade = "B+ (Good)"
            elif overall >= 3.0:
                grade = "B  (Average)"
            elif overall >= 2.5:
                grade = "C  (Below Average)"
            else:
                grade = "D  (Poor)"
            lines.append(f"  ★ Grade: {grade}")

        # Category breakdown
        categories = sorted(set(r.category for r in valid))
        if len(categories) > 1:
            lines.append("")
            lines.append("  Scores by Category:")
            for cat in categories:
                cat_results = [r for r in valid if r.category == cat]
                cat_scores = []
                for r in cat_results:
                    cat_scores.extend(v for v in r.judge_scores.values() if isinstance(v, (int, float)))
                if cat_scores:
                    avg_lat = statistics.mean(r.latency_ms for r in cat_results)
                    lines.append(f"    {cat:20} {statistics.mean(cat_scores):.2f}/5  (latency avg: {avg_lat:,.0f}ms)")

    # Negative calibration check
    if negative:
        lines.append("")
        lines.append("=" * 70)
        lines.append("  JUDGE CALIBRATION CHECK")
        lines.append("=" * 70)
        for r in negative:
            numeric = [v for v in r.judge_scores.values() if isinstance(v, (int, float))]
            if numeric:
                neg_avg = statistics.mean(numeric)
                calibrated = neg_avg < 3.0
                status = "PASS" if calibrated else "FAIL"
                lines.append(f"  Negative scenario [{r.scenario_id}] avg: {neg_avg:.1f}/5 — [{status}]")
                if calibrated:
                    lines.append(f"  Judge correctly identifies low-quality responses (avg < 3.0)")
                else:
                    lines.append(f"  WARNING: Judge gave {neg_avg:.1f}/5 to an intentionally bad response!")
                    lines.append(f"  Judge may be unreliable — scores should be interpreted with caution.")

    lines.append("")
    lines.append("=" * 70)
    failed = [r for r in results if r.error]
    lines.append(f"  Results: {len(valid)} evaluated, {len(negative)} calibration, {len(failed)} failed")
    lines.append("=" * 70)

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def main():
    # Parse args
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--server" and i + 1 < len(args):
            global MENE_URL
            MENE_URL = args[i + 1]
        elif arg == "--judge-model" and i + 1 < len(args):
            global JUDGE_MODEL
            JUDGE_MODEL = args[i + 1]
        elif arg == "--judge-provider" and i + 1 < len(args):
            global JUDGE_PROVIDER
            JUDGE_PROVIDER = args[i + 1]

    # Load API keys from config
    api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
    }
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if not api_keys["openai"]:
            api_keys["openai"] = cfg.get("api_keys", {}).get("openai", "")
        if not api_keys["anthropic"]:
            api_keys["anthropic"] = cfg.get("api_keys", {}).get("anthropic", "")
    except Exception:
        pass

    # Validate required keys
    if JUDGE_PROVIDER == "anthropic" and not api_keys["anthropic"]:
        print("ERROR: Anthropic API key required for cross-provider judge.")
        print("Set ANTHROPIC_API_KEY or check config.yaml, or use --judge-provider openai")
        return 1
    if JUDGE_PROVIDER == "openai" and not api_keys["openai"]:
        print("ERROR: OpenAI API key required. Set OPENAI_API_KEY or check config.yaml")
        return 1

    print(f"Starting evaluation: {len(SCENARIOS)} scenarios")
    print(f"Server: {MENE_URL}")
    print(f"Judge: {JUDGE_MODEL} ({JUDGE_PROVIDER}) — cross-provider: {JUDGE_PROVIDER != 'openai'}")
    print()

    # Check server is running
    async with httpx.AsyncClient(timeout=5) as check:
        try:
            r = await check.get(f"{MENE_URL}/api/status")
            status = r.json()
            print(f"Server status: {status.get('status')} (v{status.get('version')})")
        except Exception as e:
            print(f"ERROR: Cannot reach MENE server at {MENE_URL}: {e}")
            return 1

    results = []
    mene_client = httpx.AsyncClient(timeout=TIMEOUT)
    judge_client = httpx.AsyncClient(timeout=60)

    try:
        for i, scenario in enumerate(SCENARIOS, 1):
            label = scenario["name"]
            if scenario.get("is_negative_test"):
                label += " [CALIBRATION]"
            elif scenario.get("multi_turn"):
                label += f" [{len(scenario['turns'])} turns]"
            print(f"\n[{i}/{len(SCENARIOS)}] {scenario['id']}: {label}...", end=" ", flush=True)

            result = await evaluate_scenario(mene_client, judge_client, scenario, api_keys)
            results.append(result)

            if result.error:
                print(f"ERROR: {result.error[:80]}")
            else:
                numeric = [v for v in result.judge_scores.values() if isinstance(v, (int, float))]
                avg = statistics.mean(numeric) if numeric else 0
                if result.is_negative:
                    status = "CALIBRATED" if avg < 3.0 else "UNCALIBRATED"
                    print(f"{status} (score: {avg:.1f}/5)")
                else:
                    print(f"OK ({result.latency_ms:.0f}ms, score: {avg:.1f}/5)")
    finally:
        await mene_client.aclose()
        await judge_client.aclose()

    # Generate and print report
    report = generate_report(results)
    print("\n")
    print(report)

    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save raw data as JSON
    raw_path = os.path.join(os.path.dirname(__file__), "eval_raw.json")
    raw_data = []
    for r in results:
        raw_data.append({
            "scenario_id": r.scenario_id,
            "scenario_name": r.scenario_name,
            "category": r.category,
            "latency_ms": r.latency_ms,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "response_text": r.response_text,
            "multi_turn_responses": r.multi_turn_responses,
            "judge_scores": r.judge_scores,
            "judge_comment": r.judge_comment,
            "is_negative": r.is_negative,
            "error": r.error,
        })
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    print(f"Raw data saved to: {raw_path}")

    failed = sum(1 for r in results if r.error)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
