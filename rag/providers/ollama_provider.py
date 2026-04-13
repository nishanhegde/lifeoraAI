import logging
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


class OllamaProvider(LLMProvider):
    """
    Path A: Local LLM via Ollama. Zero cost, full privacy, runs offline.
    Install: https://ollama.com  →  ollama pull llama3.2:3b

    Thread-safe: client is initialised once under a lock.
    Retries transient connection errors with exponential backoff.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        config: GenerationConfig = None,
    ):
        self.model = model
        self.base_url = base_url
        self.config = config or GenerationConfig()
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        # Double-checked locking — safe under concurrent requests
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import ollama
                    self._client = ollama.Client(
                        host=self.base_url,
                        timeout=self.config.timeout_seconds,
                    )
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._generate_with_retry(system_prompt, user_prompt)
        except Exception as exc:
            raise ProviderError(self.name, f"Generation failed: {exc}", cause=exc) from exc

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _generate_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_client()
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        return response["message"]["content"]

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            models = client.list()
            available_names = [m["name"] for m in models.get("models", [])]
            found = any(self.model in name for name in available_names)
            if not found:
                raise ProviderUnavailableError(
                    self.name,
                    f"Model '{self.model}' not found. "
                    f"Run: ollama pull {self.model}. "
                    f"Available: {available_names or 'none'}",
                )
            return True
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                self.name,
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is 'ollama serve' running?",
                cause=exc,
            ) from exc

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"
