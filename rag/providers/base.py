from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: int = 60      # max wait for a single generation call
    max_retries: int = 3           # attempts before raising ProviderError


class LLMProvider(ABC):
    """
    Abstract LLM provider interface.
    All providers (Ollama, Claude, future Lifeora SLM) implement this.
    Swap providers via config.yaml — no code changes needed.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response given a system and user prompt.
        Raises ProviderError after max_retries exhausted.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is reachable.
        Called at startup — raises ProviderUnavailableError if not ready.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for logging and metrics."""
        ...
