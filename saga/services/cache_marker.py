"""3-breakpoint prompt caching marker for Anthropic/Claude models."""
from saga.core import dependencies as deps


def is_anthropic_model(model: str | None) -> bool:
    """Check if the model (or config narration model) is Anthropic/Claude."""
    return "claude" in (model or "").lower() or "claude" in deps.config.models.narration.lower()


def build_cacheable_messages(original_messages, md_prefix, dynamic_suffix, lorebook_delta=""):
    """Build messages with 3-breakpoint prompt caching structure.

    BP1: system prompt (절대 안 변함 — SystemStabilizer가 보장)
    BP2: 대화 히스토리 중간 지점 assistant (이전 턴 내용은 안 변함)
    BP3: 대화 히스토리 마지막 assistant (직전 턴까지 안 변함)
    Dynamic: md_prefix + lorebook_delta + dynamic_suffix → 마지막 user 메시지에 prepend (캐시 밖)
    """
    messages = list(original_messages)
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    is_claude = "claude" in deps.config.models.narration.lower()

    if system_idx is not None and is_claude and deps.config.prompt_caching.enabled:
        cache_ctrl = {"type": "ephemeral"}
        if deps.config.prompt_caching.cache_ttl:
            cache_ctrl["ttl"] = deps.config.prompt_caching.cache_ttl

        messages[system_idx] = dict(messages[system_idx])
        messages[system_idx]["cache_control"] = cache_ctrl

        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]

        if len(assistant_indices) >= 2:
            from saga.message_compressor import _CHUNK_USER_PREFIX
            summary_asst_indices = [
                idx for idx in assistant_indices
                if idx > 0 and messages[idx - 1].get("role") == "user"
                and str(messages[idx - 1].get("content", "")).startswith(_CHUNK_USER_PREFIX)
            ]
            if summary_asst_indices:
                mid_idx = summary_asst_indices[0]
            else:
                mid_idx = assistant_indices[len(assistant_indices) // 2]
            messages[mid_idx] = dict(messages[mid_idx])
            messages[mid_idx]["cache_control"] = cache_ctrl

            last_idx = assistant_indices[-1]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = cache_ctrl

        elif len(assistant_indices) == 1:
            last_idx = assistant_indices[0]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = cache_ctrl

        context_block = ""
        if md_prefix:
            context_block += f"[--- SAGA Context Cache ---]\n{md_prefix}\n\n"
        if lorebook_delta:
            context_block += f"[--- Active Lorebook ---]\n{lorebook_delta}\n\n"
        if dynamic_suffix:
            context_block += f"[--- SAGA Dynamic ---]\n{dynamic_suffix}"

        if context_block:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                messages[last_user_idx] = dict(messages[last_user_idx])
                messages[last_user_idx]["content"] = context_block + "\n\n" + messages[last_user_idx]["content"]
            else:
                messages.append({
                    "role": "user",
                    "content": context_block,
                })

    else:
        if system_idx is not None:
            messages[system_idx] = dict(messages[system_idx])
            content = messages[system_idx]["content"]
            if md_prefix or dynamic_suffix:
                content += f"\n\n[--- SAGA Dynamic Context ---]\n{md_prefix}\n\n{dynamic_suffix}"
            messages[system_idx]["content"] = content
        else:
            sys_content = f"[--- SAGA Dynamic Context ---]\n{md_prefix}\n\n{dynamic_suffix}"
            messages.insert(0, {"role": "system", "content": sys_content})

    return messages
