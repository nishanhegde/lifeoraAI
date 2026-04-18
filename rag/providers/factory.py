import logging

from core.exceptions import ConfigError
from .base import LLMProvider, GenerationConfig
from .ollama_provider import OllamaProvider
from .claude_provider import ClaudeProvider

logger = logging.getLogger(__name__)


def get_provider(config: dict, check_available: bool = True) -> LLMProvider:
    """
    Build an LLM provider from config.yaml values.
    Calls is_available() immediately — raises ProviderUnavailableError at startup
    rather than silently failing on the first user request.
    """
    provider_name = config.get("llm_provider", "ollama")

    if provider_name == "ollama":
        cfg = config.get("ollama", {})
        gen_config = GenerationConfig(
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 1024),
            timeout_seconds=cfg.get("timeout_seconds", 60),
        )
        provider = OllamaProvider(
            model=cfg.get("model", "llama3.2:3b"),
            base_url=cfg.get("base_url", "http://localhost:11434"),
            config=gen_config,
        )

    elif provider_name == "claude":
        cfg = config.get("claude", {})
        gen_config = GenerationConfig(
            temperature=cfg.get("temperature", 0.2),
            max_tokens=cfg.get("max_tokens", 1024),
            timeout_seconds=cfg.get("timeout_seconds", 60),
        )
        provider = ClaudeProvider(
            model=cfg.get("model", "claude-sonnet-4-6"),
            config=gen_config,
        )

    else:
        raise ConfigError(
            f"Unknown llm_provider: '{provider_name}'. Valid options: ollama, claude"
        )

    if check_available:
        provider.is_available()
        logger.info("LLM provider ready: %s", provider.name)
    else:
        logger.warning("Skipping availability check for %s", provider.name)
    return provider
