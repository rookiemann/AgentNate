"""Token estimation for dynamic max_tokens calculation."""

CHARS_PER_TOKEN = 4  # Conservative estimate


def estimate_tokens(text: str) -> int:
    """Estimate tokens from text (4 chars/token)."""
    return len(text) // CHARS_PER_TOKEN + 1 if text else 0


def estimate_message_tokens(message: dict) -> int:
    """Estimate tokens for a message including images."""
    tokens = 4  # Role overhead
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):  # Multimodal
        for part in content:
            if part.get("type") == "text":
                tokens += estimate_tokens(part.get("text", ""))
            elif part.get("type") == "image_url":
                tokens += 200  # Conservative image estimate
    return tokens


def estimate_messages_tokens(messages: list) -> int:
    """Total tokens for message list."""
    return 3 + sum(estimate_message_tokens(m) for m in messages)


def calculate_safe_max_tokens(
    context_length: int,
    input_tokens: int,
    requested_max_tokens: int,
    safety_buffer_percent: float = 0.05,
    min_output_tokens: int = 100,
) -> int:
    """Calculate safe max_tokens that won't exceed context."""
    safety_buffer = int(context_length * safety_buffer_percent)
    available = context_length - input_tokens - safety_buffer
    available = max(available, min_output_tokens)
    return max(min(requested_max_tokens, available), min_output_tokens)
