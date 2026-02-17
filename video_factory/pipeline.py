from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Garante que o pacote seja encontrado mesmo rodando de dentro de video_factory/
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from video_factory.render_pipeline import RenderRequest, RenderScene, render_script  # noqa: E402
from video_factory.script_generator import generate_scenes  # noqa: E402
from video_factory.config import EDGE_VOICE  # noqa: E402


def _build_request(topic: str, aspect: str, duration: int, scenes: int | None) -> RenderRequest:
    roteiro = generate_scenes(topic, duration_minutes=duration, format_ratio=aspect, num_scenes_override=scenes)
    scenes = roteiro.get("scenes", [])
    if not scenes:
        raise RuntimeError("Nenhuma cena retornada pelo LLM.")

    render_scenes = []
    for idx, scene in enumerate(scenes):
        vp = scene.get("visualPrompt")
        if isinstance(vp, list):
            vp = " ".join(str(x) for x in vp)
        if vp is None:
            vp = ""
        txt = scene.get("text") or vp or topic
        render_scenes.append(
            RenderScene(
                id=str(scene.get("id", idx + 1)),
                text=str(txt),
                visualPrompt=str(vp),
                localImage=scene.get("localImage"),
                animationType=scene.get("animationType"),
            )
        )

    return RenderRequest(
        scenes=render_scenes,
        format=aspect,
        voice=EDGE_VOICE,
        scriptTitle=roteiro.get("title") or topic,
    )


def run(topic: str, aspect: str = "16:9", duration: int = 1, scenes: int | None = None) -> Path:
    request = _build_request(topic, aspect, duration, scenes)
    return asyncio.run(render_script(request))


def main():
    parser = argparse.ArgumentParser(description="Renderiza vídeo com MoviePy + Edge TTS.")
    parser.add_argument("--topic", required=True, help="Tema do vídeo")
    parser.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    parser.add_argument("--duration", type=int, choices=[1, 5], default=1, help="Duração em minutos")
    parser.add_argument("--scenes", type=int, default=None, help="Forçar número de cenas (sobrescreve padrão)")
    parser.add_argument("--fast", action="store_true", help="Atalho: força poucas cenas (5) e duração 1min")
    args = parser.parse_args()
    duration = 1 if args.fast else args.duration
    scenes = 5 if args.fast else args.scenes
    outfile = run(args.topic, args.aspect, duration, scenes)
    print(json.dumps({"output": str(outfile)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
