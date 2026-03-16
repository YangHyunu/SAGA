"""Sub-B extraction: Flash 서사 요약.

프레젠테이션 S4에서 12필드 regex 추출은 '시행착오'로 분류.
현재는 Flash 미니 서사 요약 (summary + npcs + scene_type + key_event)으로 전환.
향후 scriptstate 수신 시 extract_fn 교체로 대응 (P3-a 참조).
"""
import logging
from saga.utils.parsers import parse_llm_json

logger = logging.getLogger(__name__)


async def narrative_extract(
    assistant_text: str,
    session_id: str,
    llm_client,
    config,
) -> dict | None:
    """Flash로 서사 요약 추출. 4필드 미니 요약.

    Returns:
        dict with keys: summary, npcs_mentioned, scene_type, key_event
        or None if extraction fails.
    """
    try:
        clean_text = ''.join(
            c if c.isprintable() or c in '\n\r' else ' '
            for c in assistant_text
        )
        result = await llm_client.call_llm(
            model=config.models.extraction,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "이 RP 대화에서 일어난 서사적 사건을 요약하세요. "
                        "JSON만 반환 (마크다운, 설명 금지):\n"
                        '{"summary": "2-3문장 요약", '
                        '"npcs_mentioned": ["장면에 직접 등장하여 행동 또는 대사가 있는 NPC의 고유 이름만. '
                        '다음은 제외: 무명 엑스트라(마을 사람, 여인, 병사 등), '
                        '언급만 된 인물(대화 속 이름만 나옴, 실제 장면에 없음), '
                        '역사적/신화적 배경 인물. 원문 그대로 표기, 번역 금지"], '
                        '"scene_type": "combat|dialogue|exploration|event", '
                        '"key_event": "핵심 사건 한 줄 또는 null"}'
                    ),
                },
                {"role": "user", "content": clean_text},
            ],
            temperature=0.1,
            max_tokens=1024,
            response_mime_type="application/json",
        )
        parsed = parse_llm_json(result)
        if parsed is None:
            logger.warning(f"[Extractor] Flash returned unparseable: {result[:200]}")
        return parsed
    except Exception as e:
        logger.error(f"[Extractor] Flash narrative extraction failed: {e}")
        return None
