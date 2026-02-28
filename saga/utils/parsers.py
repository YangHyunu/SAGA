"""Parsing utilities for state blocks, .md files, and world data."""
import re
import json
import yaml
import logging

logger = logging.getLogger(__name__)

# Regex for state block extraction (lenient: handles spaces, varied backticks)
STATE_BLOCK_PATTERN = re.compile(
    r'`{2,3}\s*state\s*\n(.*?)`{2,3}',
    re.DOTALL
)

def parse_state_block(response_text: str) -> dict | None:
    """Parse state block from LLM response using regex. Returns dict or None."""
    match = STATE_BLOCK_PATTERN.search(response_text)
    if not match:
        return None

    block_text = match.group(1).strip()
    result = {}

    try:
        for line in block_text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()

            if key == 'location':
                result['location'] = value
            elif key == 'location_moved':
                result['location_moved'] = value.lower() in ('true', 'yes', '1')
            elif key == 'hp_change':
                try:
                    result['hp_change'] = int(value)
                except ValueError:
                    result['hp_change'] = 0
            elif key == 'items_gained':
                result['items_gained'] = _parse_list(value)
            elif key == 'items_lost':
                result['items_lost'] = _parse_list(value)
            elif key == 'items_transferred':
                result['items_transferred'] = _parse_transfer_list(value)
            elif key == 'npc_met':
                result['npc_met'] = _parse_list(value)
            elif key == 'npc_separated':
                result['npc_separated'] = _parse_list(value)
            elif key == 'relationship_changes':
                result['relationship_changes'] = _parse_relationship_changes(value)
            elif key == 'mood':
                result['mood'] = value
            elif key == 'event_trigger':
                result['event_trigger'] = None if value.lower() in ('null', 'none', '') else value
            elif key == 'notes':
                result['notes'] = value

        return result if result else None
    except Exception as e:
        logger.warning(f"State block parse error: {e}")
        return None


def _parse_list(value: str) -> list[str]:
    """Parse [item1, item2] or item1, item2 format.

    Handles parenthesized/bracketed commas correctly:
    e.g. "Yui (유이, 18), Mina" -> ["Yui (유이, 18)", "Mina"]
    """
    value = value.strip('[]')
    if not value or value.lower() in ('없음', 'none', '[]'):
        return []
    # Split on commas NOT inside parentheses or brackets
    items = re.split(r',\s*(?![^()\[\]]*[)\]])', value)
    return [item.strip().strip('"').strip("'") for item in items if item.strip()]


def _parse_transfer_list(value: str) -> list[dict]:
    """Parse [{item: x, to: y}] format."""
    try:
        # Try JSON parse first
        parsed = json.loads(value.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: try regex for {item: x, to: y} patterns
    results = []
    pattern = re.compile(r'\{[^}]*item\s*:\s*([^,}]+)\s*,\s*to\s*:\s*([^}]+)\}')
    for match in pattern.finditer(value):
        results.append({"item": match.group(1).strip().strip('"\''), "to": match.group(2).strip().strip('"\'')})
    return results


def _parse_relationship_changes(value: str) -> list[dict]:
    """Parse [{from, to, type, delta}] format."""
    try:
        parsed = json.loads(value.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def strip_state_block(response_text: str) -> str:
    """Remove state block from response text before returning to user."""
    # Remove ```state...``` block
    cleaned = STATE_BLOCK_PATTERN.sub('', response_text)
    # Also remove the instruction header if present
    cleaned = re.sub(r'\[--- SAGA State Tracking ---\].*?```', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


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


def parse_characters_md(filepath: str) -> list[dict]:
    """Parse CHARACTERS.md into list of character dicts."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []

    characters = []
    current_char = None

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            if current_char:
                characters.append(current_char)
            name = line[3:].strip()
            current_char = {"name": name, "is_player": False, "hp": 100, "max_hp": 100, "location": "unknown", "mood": "neutral", "traits": [], "custom": {}}
        elif current_char and line.startswith('- '):
            kv = line[2:]
            if ':' in kv:
                key, _, val = kv.partition(':')
                key = key.strip().lower()
                val = val.strip()
                if key in ('플레이어', 'player', 'is_player'):
                    current_char['is_player'] = val.lower() in ('true', 'yes', '예', 'o')
                elif key in ('hp',):
                    try: current_char['hp'] = int(val)
                    except: pass
                elif key in ('max_hp', '최대hp'):
                    try: current_char['max_hp'] = int(val)
                    except: pass
                elif key in ('위치', 'location'):
                    current_char['location'] = val
                elif key in ('성격', 'traits', '특성'):
                    current_char['traits'] = [t.strip() for t in val.split(',')]
                elif key in ('기분', 'mood'):
                    current_char['mood'] = val
                else:
                    current_char['custom'][key] = val

    if current_char:
        characters.append(current_char)
    return characters


def parse_lorebook_md(filepath: str) -> list[dict]:
    """Parse LOREBOOK.md into list of lorebook entry dicts."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return []

    entries = []
    current_entry = None
    current_text_lines = []

    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('## '):
            if current_entry:
                current_entry['text'] = '\n'.join(current_text_lines).strip()
                entries.append(current_entry)
            name = stripped[3:].strip()
            current_entry = {"name": name, "type": "lore", "layer": "A1", "tags": [], "text": ""}
            current_text_lines = []
        elif current_entry and stripped.startswith('- '):
            kv = stripped[2:]
            if ':' in kv:
                key, _, val = kv.partition(':')
                key = key.strip().lower()
                val = val.strip()
                if key in ('타입', 'type'):
                    current_entry['type'] = val
                elif key in ('레이어', 'layer'):
                    current_entry['layer'] = val
                elif key in ('태그', 'tags'):
                    current_entry['tags'] = [t.strip() for t in val.split(',')]
                else:
                    current_text_lines.append(stripped)
            else:
                current_text_lines.append(stripped)
        elif current_entry:
            current_text_lines.append(line)

    if current_entry:
        current_entry['text'] = '\n'.join(current_text_lines).strip()
        entries.append(current_entry)

    return entries
