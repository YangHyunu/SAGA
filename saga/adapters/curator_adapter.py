"""Curator adapter pattern for Letta (primary) and Direct LLM (fallback)."""
from abc import ABC, abstractmethod
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Initial values for each Memory Block label
_BLOCK_INITIAL_VALUES: Dict[str, str] = {
    "narrative_summary": "## 서사 요약\n(아직 시작되지 않음)",
    "curation_decisions": "## 큐레이션 판단 기록\n(없음)",
    "contradiction_log": "## 모순 탐지/해결 기록\n(없음)",
}

_SYSTEM_PROMPT = """당신은 RP 세션의 큐레이터입니다.
N턴마다 호출되어 다음을 수행합니다:
1. 서사 모순 탐지 (죽은 NPC 재등장, 위치 불일치 등)
2. 장기 서사 흐름 정리 및 요약
3. story.md가 너무 길면 압축 제안
4. 이벤트 스케줄링 (복선 회수, 새 이벤트 제안)

[Memory Block 업데이트 규칙]
- narrative_summary: 매 큐레이션마다 전체 서사 요약을 최신 상태로 갱신하세요.
- curation_decisions: 이번 큐레이션 판단 내용을 추가하세요. 최근 5건만 유지하세요.
- contradiction_log: 탐지된 모순과 해결 방법을 기록하세요. 해결된 항목은 [해결됨]으로 표시하세요.

[출력 형식]
반드시 아래 JSON 형식으로만 응답하세요 (코드블록 포함 가능):
{"contradictions": [...], "events": [...], "compress_story": false, "compressed_summary": "...", "narrative_notes": "..."}"""


class CuratorAdapter(ABC):
    @abstractmethod
    async def run(self, session_id: str, context: dict) -> dict: ...


class LettaCuratorAdapter(CuratorAdapter):
    """Primary curator using letta-client REST SDK with per-session agents and multiple Memory Blocks."""

    def __init__(self, config):
        self.config = config
        self.client: Optional[Any] = None
        self._agents: Dict[str, Any] = {}  # session_id -> Letta Agent
        self._initialized = False

    def initialize(self):
        """Connect to Letta server only. Agent creation is lazy (per session)."""
        try:
            from letta_client import Letta
        except ImportError:
            logger.warning("[Curator] letta-client package not installed. Will use fallback.")
            self._initialized = False
            return

        try:
            base_url = self.config.curator.letta_base_url
            self.client = Letta(base_url=base_url)
            # Verify connection by listing agents
            list(self.client.agents.list())
            self._initialized = True
            logger.info(f"[Curator] Letta client connected successfully (base_url={base_url})")
        except Exception as e:
            logger.warning(f"[Curator] Letta client connection failed: {e}. Will use fallback.")
            self._initialized = False

    def _get_or_create_agent(self, session_id: str) -> Any:
        """Return cached agent for session_id, or find/create one on the server."""
        if session_id in self._agents:
            return self._agents[session_id]

        agent_name = f"saga_curator_{session_id}"

        # Search for existing agent on the server
        try:
            existing_agents = list(self.client.agents.list())
            for agent in existing_agents:
                if getattr(agent, "name", None) == agent_name:
                    self._agents[session_id] = agent
                    logger.info(f"[Curator] Reusing existing agent '{agent_name}' (id={agent.id})")
                    return agent
        except Exception as e:
            logger.warning(f"[Curator] Could not list agents: {e}")

        # Build memory_blocks as list of dicts
        memory_blocks: List[Dict[str, str]] = []
        schema: list = self.config.curator.memory_block_schema
        for label in schema:
            initial_value = _BLOCK_INITIAL_VALUES.get(label, f"## {label}\n(없음)")
            memory_blocks.append({"label": label, "value": initial_value})

        agent = self.client.agents.create(
            name=agent_name,
            model=self.config.curator.letta_model,
            embedding=self.config.curator.letta_embedding,
            memory_blocks=memory_blocks,
            system=_SYSTEM_PROMPT,
            include_base_tools=True,
        )
        self._agents[session_id] = agent
        logger.info(f"[Curator] Created new agent '{agent_name}' (id={agent.id}) with {len(memory_blocks)} memory block(s)")
        return agent

    async def run(self, session_id: str, context: dict) -> dict:
        if not self._initialized:
            raise RuntimeError("Letta curator not initialized")

        agent = self._get_or_create_agent(session_id)
        prompt = self._build_prompt(session_id, context)
        response = self.client.agents.messages.create(
            agent_id=agent.id,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_response(response)

    def _build_prompt(self, session_id: str, context: dict) -> str:
        return (
            f"세션 {session_id}의 큐레이션 요청입니다. 현재 턴: {context.get('turn_number', '?')}\n\n"
            f"[그래프 상태]\n{context.get('graph_summary', '없음')}\n\n"
            f"[최근 에피소드 기억]\n{context.get('episodes_text', '없음')}\n\n"
            f"[최근 턴 로그]\n{json.dumps(context.get('turn_logs', []), ensure_ascii=False, indent=2)}\n\n"
            f"[탐지된 모순]\n"
            + (json.dumps(context.get('contradictions', []), ensure_ascii=False)
               if context.get('contradictions') else '없음')
            + "\n\n분석 및 실행:\n"
            "1. 서사 모순이 있는지? 있으면 수정 방법은?\n"
            "2. 앞으로 촉발될 이벤트가 있는지?\n"
            "3. story.md 압축 필요?\n"
            "4. [필수] Memory Block에 이번 큐레이션 판단을 기록하세요."
        )

    def _parse_response(self, response) -> dict:
        """Parse Letta response: JSON code block -> direct parse -> brace matching."""
        default = {"contradictions": [], "events": [], "compress_story": False,
                   "compressed_summary": "", "narrative_notes": ""}

        # Collect all text from response messages
        texts: list[str] = []
        try:
            # letta-client returns a list of message objects
            items = response if isinstance(response, list) else getattr(response, "messages", [response])
            for msg in items:
                text = getattr(msg, "content", None) or getattr(msg, "text", None) or str(msg)
                texts.append(text)
        except Exception as e:
            logger.warning(f"[Curator] Failed to extract message text: {e}")
            return default

        full_text = "\n".join(texts)

        # 1) Try ```json ... ``` code block
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", full_text)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError as e:
                logger.debug(f"[Curator] Code block JSON parse failed: {e}")

        # 2) Try direct JSON parse of stripped text
        stripped = full_text.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.debug(f"[Curator] Direct JSON parse failed: {e}")

        # 3) Brace matching (find outermost {...})
        start_idx = full_text.find("{")
        if start_idx != -1:
            depth = 0
            for i, ch in enumerate(full_text[start_idx:], start=start_idx):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = full_text[start_idx:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError as e:
                            logger.debug(f"[Curator] Brace-matched JSON parse failed: {e}")
                        break

        logger.warning(f"[Curator] Could not parse JSON from response. Raw text (first 500 chars): {full_text[:500]!r}")
        return default


class DirectLLMCuratorAdapter(CuratorAdapter):
    """Fallback curator using direct LLM call (no Memory Block continuity)."""

    def __init__(self, llm_client, config):
        self.llm_client = llm_client
        self.config = config

    async def run(self, session_id: str, context: dict) -> dict:
        prompt = (
            f"그래프 상태:\n{context.get('graph_summary', '')}\n\n"
            f"에피소드:\n{context.get('episodes_text', '')}\n\n"
            f"턴 로그:\n{json.dumps(context.get('turn_logs', []), ensure_ascii=False)}"
        )

        try:
            response = await self.llm_client.call_llm(
                model=self.config.models.curator,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 RP 서사 큐레이터입니다.\n"
                            "분석 후 JSON으로 응답하세요:\n"
                            '{"contradictions": [...], "events": [...], "compress_story": bool, '
                            '"compressed_summary": "...", "narrative_notes": ""}'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            # Brace matching for fallback
            if "{" in response:
                start = response.index("{")
                depth = 0
                for i, ch in enumerate(response[start:], start=start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return json.loads(response[start:i + 1])
        except Exception as e:
            logger.error(f"[Curator Fallback] Error: {e}")
        return {"contradictions": [], "events": [], "compress_story": False,
                "compressed_summary": "", "narrative_notes": ""}
