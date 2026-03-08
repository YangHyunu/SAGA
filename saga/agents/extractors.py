"""Extraction strategies for Sub-B post-turn state parsing.

Provides regex_flash_extract as a callable injectable into PostTurnExtractor.
"""
import logging
from saga.utils.parsers import parse_state_block, parse_llm_json

logger = logging.getLogger(__name__)


async def regex_flash_extract(
    assistant_text: str,
    session_id: str,
    sqlite_db,
    llm_client,
    config,
) -> dict | None:
    """Try regex extraction first; fall back to Flash LLM if regex fails.

    Args:
        assistant_text: Raw LLM response text to extract state from.
        session_id: Current session ID for character name lookup.
        sqlite_db: SQLiteDB instance.
        llm_client: LLMClient instance.
        config: SagaConfig with models.extraction field.

    Returns:
        dict of 12 extracted fields, or None if both methods fail.
    """
    # 1. Try regex via parse_state_block
    state_block = parse_state_block(assistant_text)
    if state_block is not None:
        return state_block

    # 2. Flash LLM fallback with character name hints
    try:
        existing_chars = await sqlite_db.get_session_characters(session_id)
        char_names = [c["name"] for c in existing_chars if c.get("name")]
        char_hint = ""
        if char_names:
            char_hint = (
                f"\n\nIMPORTANT: These characters already exist in this session: "
                f"{', '.join(char_names)}. When referring to the same character, use EXACTLY "
                f"the same name form as listed above. Do not use nicknames, honorifics, or alternate spellings."
            )

        clean_text = ''.join(c if c.isprintable() or c in '\n\r' else ' ' for c in assistant_text)
        result = await llm_client.call_llm(
            model=config.models.extraction,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract game state changes from the RP response. "
                        "Return ONLY valid JSON (no markdown, no explanation) with keys: "
                        "location, location_moved, hp_change, items_gained, items_lost, "
                        "items_transferred, npc_met, npc_separated, relationship_changes, "
                        "mood, event_trigger, notes. "
                        f"Use defaults (false, 0, [], null) for missing values.{char_hint}"
                    ),
                },
                {"role": "user", "content": clean_text},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        parsed = parse_llm_json(result)
        if parsed is None:
            logger.warning(f"[Extractor] Flash returned unparseable: {result[:200]}")
        return parsed
    except Exception as e:
        logger.error(f"[Extractor] Flash extraction failed: {e}")
        return None
