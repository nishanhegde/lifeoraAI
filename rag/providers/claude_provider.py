import logging
import os
import threading

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from core.exceptions import ProviderError, ProviderUnavailableError
from .base import LLMProvider, GenerationConfig

logger = logging.getLogger(__name__)


def _retryable_anthropic_error(exc: Exception) -> bool:
    """Return True for transient errors worth retrying."""
    try:
        import anthropic
        return isinstance(exc, (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ))
    except ImportError:
        return False


class ClaudeProvider(LLMProvider):
    """
    Path B: Claude API via Anthropic SDK.
    Set env var: ANTHROPIC_API_KEY=sk-ant-...
    Switch via config.yaml: llm_provider: "claude"

    Thread-safe: client is initialised once under a lock.
    Retries rate-limit and transient API errors with exponential backoff.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", config: GenerationConfig = None):
        self.model = model
        self.config = config or GenerationConfig()
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        # Double-checked locking — safe under concurrent requests
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import anthropic
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        raise ProviderUnavailableError(
                            self.name,
                            "ANTHROPIC_API_KEY is not set. "
                            "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...",
                        )
                    self._client = anthropic.Anthropic(
                        api_key=api_key,
                        timeout=float(self.config.timeout_seconds),
                    )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._generate_with_retry(system_prompt, user_prompt)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderError(self.name, f"Generation failed: {exc}", cause=exc) from exc

    @retry(
        retry=retry_if_exception_type(_retryable_anthropic_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.config.temperature,
        )
        return message.content[0].text

    def is_available(self) -> bool:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderUnavailableError(
                self.name,
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...",
            )
        return True

    @property
    def name(self) -> str:
        return f"Claude/{self.model}"
