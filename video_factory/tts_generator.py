from __future__ import annotations

import asyncio
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tempfile
import edge_tts
import pyttsx3

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.config import EDGE_VOICE, ASSETS_DIR, TTS_PROVIDER  # noqa: E402

log = logging.getLogger(__name__)


async def _edge_tts(text: str, outfile: Path) -> Path:
    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    with outfile.open("wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    return outfile


def synthesize(text: str, filename: str = "tts_output.mp3") -> Path:
    """Gera audio TTS com Edge; retorna caminho do arquivo."""
    out_dir = ASSETS_DIR / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / filename
    asyncio.run(_edge_tts(text, outfile))
    log.info("Audio salvo em %s", outfile)
    return outfile


async def synthesize_to_bytes(text: str, voice: str | None = None) -> bytes:
    """Retorna audio em bytes (primeiro provider disponível)."""
    data, _, _ = await synthesize_to_bytes_with_metadata(text, voice)
    return data


def _pyttsx3_voice_id(engine, voice_hint: str | None) -> str | None:
    if not voice_hint:
        return None
    hint = voice_hint.lower()
    for v in engine.getProperty("voices"):
        name = (v.name or "").lower()
        lang_raw = v.languages[0] if (getattr(v, "languages", None)) else ""
        if isinstance(lang_raw, (bytes, bytearray)):
            lang_raw = lang_raw.decode(errors="ignore")
        lang = str(lang_raw).lower().replace("-", "") if lang_raw else ""
        if hint in name or hint in lang or hint in (v.id or "").lower():
            return v.id
    return None


def _pyttsx3_first_match(engine, pref_lang: str = "pt") -> str | None:
    pref_lang = (pref_lang or "").lower()
    voices = engine.getProperty("voices") or []
    # 1) exact lang match (starts with pref_lang)
    for v in voices:
        lang_raw = v.languages[0] if (getattr(v, "languages", None)) else ""
        if isinstance(lang_raw, (bytes, bytearray)):
            lang_raw = lang_raw.decode(errors="ignore")
        lang = str(lang_raw).lower()
        if lang.startswith(pref_lang):
            return v.id
    # 2) first available
    return voices[0].id if voices else None


def _offline_tts_bytes(text: str, voice: str | None = None) -> bytes:
    engine = pyttsx3.init()
    vid = _pyttsx3_voice_id(engine, voice or EDGE_VOICE) or _pyttsx3_first_match(engine, "pt")
    if vid:
        engine.setProperty("voice", vid)
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.95))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = Path(tmp.name)
    try:
        engine.save_to_file(text, str(tmp_path))
        engine.runAndWait()
        return tmp_path.read_bytes()
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


async def synthesize_to_bytes_with_metadata(text: str, voice: str | None = None) -> Tuple[bytes, List[Dict[str, Any]], str]:
    """
    Gera audio e retorna metadados de palavras para legendas.
    Tenta providers na ordem configurada em TTS_PROVIDER (ex.: "edge,offline").
    Retorna (bytes, metadata, extension).
    """
    providers = [p.strip().lower() for p in (TTS_PROVIDER or "edge").split(",") if p.strip()]
    last_err: Exception | None = None
    for provider in providers:
        try:
            if provider == "edge":
                communicate = edge_tts.Communicate(text, voice or EDGE_VOICE, boundary="WordBoundary")
                buffer = BytesIO()
                metadata: List[Dict[str, Any]] = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer.write(chunk["data"])
                    else:
                        metadata.append(chunk)
                return buffer.getvalue(), metadata, "mp3"
            if provider in {"offline", "pyttsx3"}:
                data = _offline_tts_bytes(text, voice)
                return data, [], "wav"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("Provider TTS '%s' falhou: %s", provider, exc)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Nenhum provider TTS configurado.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gera narracao via Edge TTS.")
    parser.add_argument("texto")
    parser.add_argument("--out", default="tts_demo.mp3")
    args = parser.parse_args()
    path = synthesize(args.texto, args.out)
    print(path)


if __name__ == "__main__":
    main()
