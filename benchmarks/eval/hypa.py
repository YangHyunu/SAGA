"""HypaV3 재현 (뮈토스 하이파 V5 export 고정, exp 경로 단일).

규범 스펙: docs/superpowers/plans/2026-08-09-refs/hypav3-algorithm.md
Task 4: 설정 로드 + 토크나이저 + chats 변환 계약.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")


@dataclass
class HypaSettings:
    max_chats_per_summary: int = 8
    query_chat_count: int = 3
    memory_tokens_ratio: float = 0.39
    extra_summarization_ratio: float = 0.01
    recent_memory_ratio: float = 0.6
    similar_memory_ratio: float = 0.4
    do_not_summarize_user_message: bool = False
    summarization_prompt: str = ""
    # export에 없는 필드 (HV3:1803 기본값) — hypav3.ts:105-116
    summary_chunk_separator: str = "\\n\\n"


# export 필드명 -> HypaSettings 필드명 (§1 표, 화이트리스트 병합 대상만)
_FIELD_MAP = {
    "maxChatsPerSummary": ("max_chats_per_summary", int),
    "queryChatCount": ("query_chat_count", int),
    "memoryTokensRatio": ("memory_tokens_ratio", float),
    "extraSummarizationRatio": ("extra_summarization_ratio", float),
    "recentMemoryRatio": ("recent_memory_ratio", float),
    "similarMemoryRatio": ("similar_memory_ratio", float),
    "doNotSummarizeUserMessage": ("do_not_summarize_user_message", bool),
    "summarizationPrompt": ("summarization_prompt", str),
}


def load_hypa_settings(export_path: str) -> HypaSettings:
    """뮈토스 하이파 V5 export → HypaSettings.

    "키 존재 + typeof 일치" 화이트리스트 병합 (hypav3.ts:1814-1824).
    summary_chunk_separator는 export에 없으므로 항상 코드 기본값을 쓴다.
    """
    with open(export_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_settings = raw.get("data", {}).get("settings", {})

    kwargs: Dict[str, Any] = {}
    for export_key, (field_name, expected_type) in _FIELD_MAP.items():
        if export_key in raw_settings and isinstance(raw_settings[export_key], expected_type):
            kwargs[field_name] = raw_settings[export_key]
    return HypaSettings(**kwargs)


def tok_chat(chat: Dict[str, Any]) -> int:
    """tokenizer.tokenize_chat 재현 — 비-gpt 경로 (index.svelte.ts:287-293).

    name 항은 벤치 chats에 name 필드가 없어 생략 (스펙 이탈 아님).
    """
    return len(_ENC.encode(chat["content"])) + 3


def to_risu_chats(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """벤치 히스토리 -> RisuAI chats 배열 (memo 계약).

    memo = f"m{i}" — 히스토리 내 메시지 인덱스(greeting 포함 0부터).
    인덱스 기반이므로 edit_at(content 변형)·리롤(같은 자리 교체)에도 memo가
    불변 — start_idx 매칭(hypav3.ts:214-229)이 깨지지 않는다.
    """
    return [
        {"role": msg["role"], "content": msg["content"], "memo": f"m{i}"}
        for i, msg in enumerate(history)
    ]
