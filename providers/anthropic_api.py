"""
Anthropic API Provider
=======================
Wraps the Anthropic Python SDK for Claude models.
"""

import anthropic
from config import ANTHROPIC_API_KEY, MAX_TOKENS, TEMPERATURE
from providers.base import BaseProvider


class AnthropicProvider(BaseProvider):

    def __init__(self, model_id: str, display_name: str):
        super().__init__(model_id, display_name)
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
