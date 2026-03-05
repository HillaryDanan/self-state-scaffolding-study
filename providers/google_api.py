"""
Google Gemini API Provider
============================
Wraps the google-genai SDK for Gemini models.

Uses the google.genai client (google-genai package).
Gemini 3 models require thinking_level config; we set "minimal"
to keep inference deterministic and comparable across providers.
"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, MAX_TOKENS, TEMPERATURE
from providers.base import BaseProvider


class GoogleProvider(BaseProvider):

    def __init__(self, model_id: str, display_name: str):
        super().__init__(model_id, display_name)
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                # Gemini 3+ requires thinking_level; minimal for determinism
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        return response.text
