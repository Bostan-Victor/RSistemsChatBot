from __future__ import annotations

from typing import Any

from openai import OpenAI


class LLMClient:
    """Placeholder for OpenAI client wrapper."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        client = OpenAI(api_key=self.api_key)
        response: Any = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        return content or ""
