from __future__ import annotations

import asyncio
import logging
import sys
from io import BytesIO
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.config import EDGE_VOICE, ASSETS_DIR  # noqa: E402

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
    """Retorna audio MP3 em bytes sem gravar em disco."""
    communicate = edge_tts.Communicate(text, voice or EDGE_VOICE)
    buffer = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buffer.write(chunk["data"])
    return buffer.getvalue()


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
