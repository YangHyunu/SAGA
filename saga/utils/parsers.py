"""Parsing utilities for state blocks, .md files, and world data."""
import re
import json
import logging

logger = logging.getLogger(__name__)

# Regex for state block extraction (lenient: handles spaces, varied backticks)
# Matches both closed (```state...```) and unclosed (```state... EOF) blocks
STATE_BLOCK_PATTERN = re.compile(
    r'`{2,3}\s*state\s*\n(.*?)(?:`{2,3}|$)',
    re.DOTALL
)

def strip_state_block(response_text: str) -> str:
    """Remove state block from response text before returning to user."""
    # Remove ```state...``` block
    cleaned = STATE_BLOCK_PATTERN.sub('', response_text)
    # Also remove the instruction header if present
    cleaned = re.sub(r'\[--- SAGA State Tracking ---\].*?```', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_llm_json(text: str) -> dict | None:
    """LLM 응답에서 JSON dict 추출. 3단계: 직접 파싱 → 코드블록 → 균형 중괄호 매칭."""
    if not text:
        return None
    text = text.strip()
    # 1) Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    # 2) Code block (```json ... ``` or ``` ... ```)
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    # 3) Balanced brace matching (outermost { ... })
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i + 1])
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        # 4) Truncated JSON recovery: try closing open strings/braces
        fragment = text[start:]
        for suffix in ('"}', '"}', '"}]}', '"}]}'  , '"]}', '"}'):
            try:
                result = json.loads(fragment + suffix)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                continue
        # 5) Last resort: strip to last complete key-value, close braces
        last_comma = fragment.rfind(',')
        if last_comma > 0:
            try:
                result = json.loads(fragment[:last_comma] + '}')
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def format_turn_narrative(turn_number: int, user_input: str, response_text: str, state_block: dict) -> str:
    """Format a turn into a narrative summary for episode storage."""
    clean_response = strip_state_block(response_text)
    # Truncate for storage
    if len(clean_response) > 500:
        clean_response = clean_response[:500] + "..."

    summary_parts = [f"Turn {turn_number}"]
    if state_block.get("location"):
        summary_parts.append(f"장소: {state_block['location']}")
    if state_block.get("npc_met"):
        summary_parts.append(f"만남: {', '.join(state_block['npc_met'])}")
    if state_block.get("items_gained"):
        summary_parts.append(f"획득: {', '.join(state_block['items_gained'])}")
    if state_block.get("event_trigger"):
        summary_parts.append(f"이벤트: {state_block['event_trigger']}")

    header = " | ".join(summary_parts)
    return f"{header}\n{clean_response[:300]}"
