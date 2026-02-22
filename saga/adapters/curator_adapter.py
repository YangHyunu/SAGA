"""Curator adapter pattern for Letta (primary) and Direct LLM (fallback)."""
from abc import ABC, abstractmethod
import json
import logging

logger = logging.getLogger(__name__)


class CuratorAdapter(ABC):
    @abstractmethod
    async def run(self, session_id: str, context: dict) -> dict: ...


class LettaCuratorAdapter(CuratorAdapter):
    """Primary curator using Letta SDK with Memory Block."""
    def __init__(self, config):
        self.config = config
        self.client = None
        self.agent = None
        self._initialized = False

    def initialize(self):
        try:
            from letta import create_client
            self.client = create_client()
            self.agent = self.client.create_agent(
                name="saga_curator",
                model=self.config.models.curator,
                memory=self.client.create_block(
                    label="narrative_memory",
                    value="""## 서사 큐레이션 기록
이 블록은 큐레이터가 매 N턴마다 자기편집합니다.
이전 큐레이션 판단을 기억하여 일관된 서사 관리를 합니다.

### 서사 요약
(아직 시작되지 않음)

### 이전 큐레이션 결정
(없음)

### 주의 사항
(없음)
"""
                ),
                system="""당신은 RP 세션의 큐레이터입니다.
N턴마다 호출되어 다음을 수행합니다:
1. 서사 모순 탐지 (죽은 NPC 재등장, 위치 불일치 등)
2. 장기 서사 흐름 정리 및 요약
3. story.md가 너무 길면 압축 제안
4. 이벤트 스케줄링 (복선 회수, 새 이벤트 제안)

[중요] Memory Block을 반드시 업데이트하세요:
- 이번에 발견한 모순, 내린 판단, 서사 요약을 기록
- 이전 판단을 참조하여 일관성 유지

JSON으로 응답하세요:
{"contradictions": [...], "events": [...], "compress_story": bool, "compressed_summary": "...", "narrative_notes": "..."}""",
            )
            self._initialized = True
            logger.info("[Curator] Letta curator initialized successfully")
        except Exception as e:
            logger.warning(f"[Curator] Letta initialization failed: {e}. Will use fallback.")
            self._initialized = False

    async def run(self, session_id: str, context: dict) -> dict:
        if not self._initialized:
            raise RuntimeError("Letta curator not initialized")

        prompt = self._build_prompt(session_id, context)
        response = self.client.send_message(
            agent_id=self.agent.id,
            message=prompt,
            role="user"
        )
        return self._parse_response(response)

    def _build_prompt(self, session_id, context):
        return f"""세션 {session_id}의 큐레이션 요청입니다. 현재 턴: {context.get('turn_number', '?')}

[그래프 상태]
{context.get('graph_summary', '없음')}

[최근 에피소드 기억]
{context.get('episodes_text', '없음')}

[최근 턴 로그]
{json.dumps(context.get('turn_logs', []), ensure_ascii=False, indent=2)}

[탐지된 모순]
{json.dumps(context.get('contradictions', []), ensure_ascii=False) if context.get('contradictions') else '없음'}

분석 및 실행:
1. 서사 모순이 있는지? 있으면 수정 방법은?
2. 앞으로 촉발될 이벤트가 있는지?
3. story.md 압축 필요?
4. [필수] Memory Block에 이번 큐레이션 판단을 기록하세요."""

    def _parse_response(self, response):
        # Letta response parsing — extract JSON from agent response
        try:
            if hasattr(response, 'messages'):
                for msg in response.messages:
                    text = getattr(msg, 'text', str(msg))
                    # Try to find JSON in the response
                    if '{' in text:
                        start = text.index('{')
                        end = text.rindex('}') + 1
                        return json.loads(text[start:end])
            return {"contradictions": [], "events": [], "compress_story": False}
        except (json.JSONDecodeError, ValueError):
            return {"contradictions": [], "events": [], "compress_story": False}


class DirectLLMCuratorAdapter(CuratorAdapter):
    """Fallback curator using direct LLM call (no Memory Block continuity)."""
    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config

    async def run(self, session_id: str, context: dict) -> dict:
        prompt = f"""그래프 상태:\n{context.get('graph_summary', '')}\n\n에피소드:\n{context.get('episodes_text', '')}\n\n턴 로그:\n{json.dumps(context.get('turn_logs', []), ensure_ascii=False)}"""

        try:
            response = await self.llm_client.call_llm(
                model=self.config.models.curator,
                messages=[
                    {"role": "system", "content": "당신은 RP 서사 큐레이터입니다.\n분석 후 JSON으로 응답하세요:\n{\"contradictions\": [...], \"events\": [...], \"compress_story\": bool, \"compressed_summary\": \"...\"}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            if '{' in response:
                start = response.index('{')
                end = response.rindex('}') + 1
                return json.loads(response[start:end])
        except Exception as e:
            logger.error(f"[Curator Fallback] Error: {e}")
        return {"contradictions": [], "events": [], "compress_story": False}
