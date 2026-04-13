"""
All LifeoraAI exceptions in one place.
Callers catch LifeoraError to handle any application error uniformly.
Specific subclasses let callers handle only what they care about.
"""


class LifeoraError(Exception):
    """Base for all LifeoraAI errors."""


class ValidationError(LifeoraError):
    """Input failed validation (too long, empty, unsafe content)."""


class ProviderError(LifeoraError):
    """LLM provider call failed after all retries."""

    def __init__(self, provider_name: str, message: str, cause: Exception = None):
        self.provider_name = provider_name
        self.cause = cause
        super().__init__(f"[{provider_name}] {message}")


class ProviderUnavailableError(ProviderError):
    """Provider is not reachable or not configured (Ollama not running, missing API key)."""


class RetrievalError(LifeoraError):
    """Vector store query failed."""


class IngestError(LifeoraError):
    """Document ingestion failed."""


class ConfigError(LifeoraError):
    """config.yaml is missing required fields or has invalid values."""
