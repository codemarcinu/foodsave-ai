from typing import Any, Dict, Optional


class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError("LLMClient.generate() not implemented")

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("LLMClient.embed() not implemented")

    async def chat(
        self, messages: list[Dict[str, str]], **kwargs: Any
    ) -> Optional[str]:
        raise NotImplementedError("LLMClient.chat() not implemented")
