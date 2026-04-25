from __future__ import annotations

from src.core.config import Settings
from src.llm.openai_adapter import OpenAIAdapter

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMFactory:
    @staticmethod
    def from_config(settings: Settings) -> OpenAIAdapter:
        provider = settings.llm_provider.lower()
        if provider == "openai":
            base_url = settings.llm_base_url
        elif provider == "groq":
            base_url = settings.llm_base_url or GROQ_BASE_URL
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        return OpenAIAdapter(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=base_url,
            request_timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
        )
