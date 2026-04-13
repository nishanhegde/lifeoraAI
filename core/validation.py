"""
Input validation for user queries.
All validation raises ValidationError so callers can catch it at the API boundary.
"""
import re
from .exceptions import ValidationError

# Hard limits
MAX_QUERY_LENGTH = 2_000   # characters
MIN_QUERY_LENGTH = 3       # characters

VALID_TOPIC_FILTERS = {"nutrition", "exercise", "lifestyle", None}

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+|all\s+)?(instructions?|prompts?|rules?|system)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"disregard\s+(all\s+)?(previous\s+|your\s+)?(instructions?|rules?)",
    r"jailbreak",
    r"<\s*/?system\s*>",          # XML-style role injection
    r"\[INST\]|\[/INST\]",        # LLaMA instruction tags
    r"###\s*(Human|Assistant|System)\s*:",  # role delimiters
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def validate_query(question: str) -> str:
    """
    Validate and clean a user query.
    Returns the stripped query string.
    Raises ValidationError with a user-friendly message on failure.
    """
    if not isinstance(question, str):
        raise ValidationError("Question must be a string.")

    question = question.strip()

    if len(question) < MIN_QUERY_LENGTH:
        raise ValidationError("Question is too short. Please ask a complete question.")

    if len(question) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Question is too long ({len(question)} characters). "
            f"Please keep it under {MAX_QUERY_LENGTH} characters."
        )

    if _INJECTION_RE.search(question):
        raise ValidationError(
            "Your question contains patterns that look like instruction overrides. "
            "Please ask a genuine health or lifestyle question."
        )

    return question


def validate_topic_filter(topic: str | None) -> str | None:
    """Raise ValidationError if topic is not a recognised filter value."""
    if topic not in VALID_TOPIC_FILTERS:
        raise ValidationError(
            f"Unknown topic filter '{topic}'. "
            f"Valid options: {', '.join(t for t in VALID_TOPIC_FILTERS if t)}."
        )
    return topic


def validate_config(config: dict) -> None:
    """
    Validate config.yaml contents at startup.
    Raises ConfigError with a clear message if anything is wrong.
    """
    from .exceptions import ConfigError

    provider = config.get("llm_provider")
    if provider not in ("ollama", "claude"):
        raise ConfigError(
            f"config.yaml: llm_provider must be 'ollama' or 'claude', got: {provider!r}"
        )

    if provider == "ollama":
        if not config.get("ollama", {}).get("model"):
            raise ConfigError("config.yaml: ollama.model is required when llm_provider is 'ollama'")

    if provider == "claude":
        if not config.get("claude", {}).get("model"):
            raise ConfigError("config.yaml: claude.model is required when llm_provider is 'claude'")

    emb = config.get("embedding", {})
    chunk_size = emb.get("chunk_size", 300)
    if not isinstance(chunk_size, int) or chunk_size < 50:
        raise ConfigError("config.yaml: embedding.chunk_size must be an integer >= 50")

    ret = config.get("retrieval", {})
    top_k = ret.get("top_k", 5)
    if not isinstance(top_k, int) or top_k < 1:
        raise ConfigError("config.yaml: retrieval.top_k must be a positive integer")
