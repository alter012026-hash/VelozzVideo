from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from moviepy.editor import concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory.config import ASPECT, FPS, CODEC, AUDIO_CODEC, ASSETS_DIR  # noqa: E402

log = logging.getLogger(__name__)


def stitch_captions(captions: List[str], duration_per_scene: float = 4.0) -> Path:
    """
    Cria vídeo simples com blocos de texto. Útil para smoke test do pipeline.
    """
    w, h = (1080, 1920) if ASPECT == "9:16" else (1920, 1080)
    clips = []
    for text in captions:
        bg = ColorClip(size=(w, h), color=(8, 10, 20), duration=duration_per_scene)
        txt = TextClip(text, fontsize=48, color="white", size=(w - 200, h - 200), method="caption")
        txt = txt.set_duration(duration_per_scene).set_pos("center")
        clips.append(CompositeVideoClip([bg, txt]))

    final = concatenate_videoclips(clips, method="compose")
    final = final.set_fps(FPS)

    out_dir = ASSETS_DIR / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / "preview.mp4"
    final.write_videofile(outfile.as_posix(), codec=CODEC, audio_codec=AUDIO_CODEC, fps=FPS)
    log.info("Vídeo salvo em %s", outfile)
    return outfile
