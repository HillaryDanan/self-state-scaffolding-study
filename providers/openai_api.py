"""
OpenAI API Provider
====================
Wraps the OpenAI Python SDK.

Note: Newer OpenAI models (gpt-5+) use max_completion_tokens
instead of the deprecated max_tokens parameter.
"""

from openai import OpenAI
from config import OPENAI_API_KEY, MAX_TOKENS, TEMPERATURE
from providers.base import BaseProvider


class OpenAIProvider(BaseProvider):

    def __init__(self, model_id: str, display_name: str):
        super().__init__(model_id, display_name)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content
