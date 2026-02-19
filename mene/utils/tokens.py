"""Token counting and budget management using tiktoken."""
import tiktoken

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.encoding_for_model("gpt-4")
    return _encoder

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))

def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += count_tokens(msg.get("content", ""))
        total += count_tokens(msg.get("role", ""))
    total += 2  # reply priming
    return total

def truncate_to_budget(text: str, max_tokens: int) -> str:
    encoder = _get_encoder()
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])
