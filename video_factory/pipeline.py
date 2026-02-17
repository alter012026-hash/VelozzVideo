from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, concatenate_videoclips, ImageClip
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

from video_factory import config  # noqa: E402
from video_factory.script_generator import generate_scenes  # noqa: E402
from video_factory.tts_generator import synthesize  # noqa: E402


def _wrap_text(text: str, width: int = 40) -> str:
    words = text.split()
    lines = []
    line = []
    for w in words:
        test = " ".join(line + [w])
        if len(test) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line = test.split() if isinstance(test, str) else [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def render_text_image(text: str, size=(1920, 1080), bg_color=(8, 10, 20), text_color=(255, 255, 255)):
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    wrapped = _wrap_text(text, width=60)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    draw.multiline_text((x, y), wrapped, font=font, fill=text_color, align="center", spacing=6)
    return np.array(img)


def build_video_from_scenes(scenes, aspect: str, voice_prefix: str = "voice") -> Path:
    w, h = (1080, 1920) if aspect == "9:16" else (1920, 1080)
    clips = []
    for idx, scene in enumerate(scenes):
        audio_path = synthesize(scene["text"], f"{voice_prefix}_{idx+1}.mp3")
        audio_clip = AudioFileClip(audio_path)
        frame = render_text_image(scene["text"], size=(w, h))
        video_clip = ImageClip(frame).with_duration(audio_clip.duration)
        video_clip = video_clip.with_audio(audio_clip)
        clips.append(video_clip)

    final = concatenate_videoclips(clips, method="compose").with_fps(config.FPS)
    out_dir = config.ASSETS_DIR / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"auto_render_{uuid.uuid4().hex[:8]}.mp4"
    final.write_videofile(outfile.as_posix(), codec=config.CODEC, audio_codec=config.AUDIO_CODEC, fps=config.FPS)
    return outfile


def run(topic: str, aspect: str = "16:9") -> Path:
    roteiro = generate_scenes(topic)
    scenes = roteiro.get("scenes", [])
    if not scenes:
        raise RuntimeError("Nenhuma cena retornada pelo LLM.")
    return build_video_from_scenes(scenes, aspect)


def main():
    parser = argparse.ArgumentParser(description="Automatiza vídeo: roteiro (Ollama) -> TTS (Edge) -> MP4.")
    parser.add_argument("--topic", required=True, help="Tema do vídeo")
    parser.add_argument("--aspect", choices=["16:9", "9:16"], default="16:9")
    args = parser.parse_args()
    outfile = run(args.topic, args.aspect)
    print(json.dumps({"output": str(outfile)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
