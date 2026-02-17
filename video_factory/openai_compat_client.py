from __future__ import annotations

import os
import requests


class OpenAICompatClient:
    """Cliente mínimo para endpoints compatíveis com OpenAI."""

    def __init__(self, base_url: str | None, api_key: str | None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            raise ValueError("LLM_API_BASE não configurado")
        if not self.api_key:
            raise ValueError("LLM_API_KEY/OPENAI_API_KEY não configurado")

    def chat(self, prompt: str, model: str | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
