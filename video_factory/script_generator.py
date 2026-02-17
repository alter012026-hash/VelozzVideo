from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.config import OLLAMA_HOST, LLM_MODEL, LLM_TIMEOUT, LLM_API_BASE, LLM_API_KEY  # noqa: E402
from video_factory.openai_compat_client import OpenAICompatClient  # noqa: E402

log = logging.getLogger(__name__)


def _ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"timeout": LLM_TIMEOUT}}
    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def generate_scenes(prompt: str) -> Dict[str, Any]:
    """
    Gera roteiro em JSON com título e cenas.
    Usa Ollama local; se falhar, tenta endpoint OpenAI compatível.
    """
    system_prompt = (
        "Gere um roteiro curto para vídeo. Responda SOMENTE JSON no formato: "
        '{"title": "...", "scenes": [{"id": "1", "text": "...", "visualPrompt": "..."}]} '
        "Limite máximo 5 cenas."
    )
    full_prompt = f"{system_prompt}\n\nTema: {prompt}"

    try:
        raw = _ollama_generate(full_prompt)
        log.info("Roteiro gerado via Ollama")
    except Exception as exc:  # noqa: BLE001
        log.warning("Falha Ollama, tentando OpenAI compatível: %s", exc)
        client = OpenAICompatClient(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
        raw = client.chat(full_prompt)

    try:
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        log.error("Não foi possível converter resposta em JSON: %s", exc)
        return {"title": "Roteiro inválido", "scenes": []}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gera roteiro usando LLM local ou compatível.")
    parser.add_argument("tema", help="Tema ou tópico do vídeo")
    args = parser.parse_args()

    roteiro = generate_scenes(args.tema)
    print(json.dumps(roteiro, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
