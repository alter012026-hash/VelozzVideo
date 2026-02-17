from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from moviepy.video.fx.SlideIn import SlideIn
from moviepy.video.fx.CrossFadeIn import CrossFadeIn
from moviepy.video.fx.CrossFadeOut import CrossFadeOut
from moviepy.video.fx.Resize import Resize
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.Clip import Clip as MoviepyClip

from video_factory.config import (
    ANIMATION_INTENSITY,
    ANIMATION_STYLE,
    ANIMATION_TYPES,
    ASPECT,
    AUDIO_CODEC,
    CODEC,
    FADE_DURATION,
    FPS,
    TRANSITION_DURATION,
    TRANSITION_STYLE,
    TRANSITION_TYPES,
    VIDEO_FORMATS,
    ASSETS_DIR,
)
from video_factory.tts_metadata import WordTiming

logger = logging.getLogger(__name__)


def _ensure_dirs() -> None:
    for path in (ASSETS_DIR / "images", ASSETS_DIR / "audio", ASSETS_DIR / "video"):
        path.mkdir(parents=True, exist_ok=True)


_ensure_dirs()

if not hasattr(MoviepyClip, "set_duration"):
    def _compat_set_duration(self, duration, change_end=True):
        """Compat shim when moviepy removes ``set_duration``."""
        self.duration = duration
        if change_end:
            if duration is None:
                self.end = None
            elif self.start is not None:
                self.end = self.start + duration
            else:
                self.end = duration
        else:
            if duration is None:
                raise ValueError("Cannot change clip start when new duration is None")
            if self.end is None:
                raise ValueError("Cannot change clip start: end is undefined")
            self.start = self.end - duration

        for child in (getattr(self, "mask", None), getattr(self, "audio", None)):
            if child is None:
                continue
            if hasattr(child, "set_duration"):
                child.set_duration(duration, change_end=change_end)
            elif hasattr(child, "with_duration"):
                child = child.with_duration(duration, change_end=change_end)
        return self

    MoviepyClip.set_duration = _compat_set_duration


class VideoBuilder:
    """
    Motor de montagem de vídeos que combina MoviePy, transições, efeitos e narração sincronizada.
    """

    def __init__(
        self,
        format_ratio: str = "16:9",
        transition_style: Optional[str] = None,
        transition_types: Optional[List[str]] = None,
        transition_duration: Optional[float] = None,
        color_filter: Optional[str] = None,
        color_strength: float = 0.35,
    ):
        if format_ratio not in VIDEO_FORMATS:
            raise ValueError(f"Formato inválido: {format_ratio}")
        self.format_ratio = format_ratio
        self.width = VIDEO_FORMATS[format_ratio]["width"]
        self.height = VIDEO_FORMATS[format_ratio]["height"]
        self.fps = FPS
        self.transition_style = (transition_style or TRANSITION_STYLE or "mixed").lower()
        self.transition_types = transition_types or TRANSITION_TYPES
        self.transition_duration = (
            float(transition_duration) if transition_duration is not None else float(TRANSITION_DURATION or FADE_DURATION)
        )
        self.color_filter = (color_filter or "").lower()
        self.color_strength = max(0.0, min(1.5, color_strength))

    def _resize_to_cover(self, clip: Any) -> Any:
        """
        Redimensiona mantendo proporção: preenche o quadro e corta o excedente
        para evitar distorção.
        """
        target_aspect = self.width / self.height
        clip_aspect = (clip.w / clip.h) if getattr(clip, "h", 0) else target_aspect

        if clip_aspect > target_aspect:
            clip = clip.resized(height=self.height)
            clip = clip.cropped(x_center=clip.w / 2, width=self.width)
        else:
            clip = clip.resized(width=self.width)
            clip = clip.cropped(y_center=clip.h / 2, height=self.height)
        return clip

    def _pick_transition(self, index: int) -> str:
        style = self.transition_style
        if style == "mixed":
            types = self.transition_types or ["fade"]
            return types[index % len(types)]
        return style

    def _transition_duration(self, clip_duration: float) -> float:
        if clip_duration <= 0:
            return 0.0
        base = self.transition_duration or FADE_DURATION
        return max(0.05, min(base, clip_duration * 0.25))

    def _apply_transition_effects(self, clip: Any, index: int) -> Any:
        transition = self._pick_transition(index)
        duration = self._transition_duration(getattr(clip, "duration", 0) or 0)
        if duration <= 0 or transition == "none":
            return clip

        if transition == "fade":
            return clip.with_effects([FadeIn(duration), FadeOut(duration)])
        if transition == "crossfade":
            return clip.with_effects([CrossFadeIn(duration), CrossFadeOut(duration)])

        if transition.startswith("slide_"):
            side = transition.split("_", 1)[1]
            side = {"up": "top", "down": "bottom"}.get(side, side)
            if side not in {"left", "right", "top", "bottom"}:
                return clip.with_effects([FadeIn(duration), FadeOut(duration)])
            return clip.with_effects(
                [
                    SlideIn(duration, side),
                    FadeIn(min(0.2, duration)),
                    FadeOut(min(0.2, duration)),
                ]
            )

        return clip.with_effects([FadeIn(duration), FadeOut(duration)])

    def _apply_color_filter(self, clip: Any) -> Any:
        mode = (self.color_filter or "none").lower()
        strength = max(0.0, float(self.color_strength))
        if mode in ("none", "off") or strength <= 0:
            return clip

        def _cinematic(img):
            arr = img.astype(np.float32)
            arr *= 1.06
            arr[..., 0] *= 1.03
            arr[..., 2] *= 0.97
            return arr

        def _cool(img):
            arr = img.astype(np.float32)
            arr[..., 1] *= 0.98
            arr[..., 2] *= 1.08
            arr *= 1.02
            return arr

        def _warm(img):
            arr = img.astype(np.float32)
            arr[..., 0] *= 1.08
            arr[..., 2] *= 0.95
            return arr

        def _bw(img):
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            return np.stack([gray, gray, gray], axis=-1)

        def _vibrant(img):
            arr = img.astype(np.float32)
            mean = arr.mean(axis=2, keepdims=True)
            arr = mean + (arr - mean) * 1.18
            arr *= 1.04
            return arr

        def _vhs(img):
            arr = img.astype(np.float32)
            noise = np.random.normal(0, 6, arr.shape)
            arr = arr + noise
            arr[..., 0] *= 1.02
            arr[..., 2] *= 0.94
            return arr

        def _matte(img):
            arr = img.astype(np.float32)
            arr = arr ** 0.96
            arr *= 1.01
            return np.clip(arr + 12, 0, 255)

        filters = {
            "cinematic": _cinematic,
            "cool": _cool,
            "teal": _cool,
            "warm": _warm,
            "bw": _bw,
            "b&w": _bw,
            "grayscale": _bw,
            "vibrant": _vibrant,
            "vhs": _vhs,
            "matte": _matte,
        }
        fn = filters.get(mode)
        if not fn:
            return clip

        alpha = min(1.5, strength)

        def blend(img):
            base = img.astype(np.float32)
            fx = fn(base)
            out = base * (1 - alpha) + fx * alpha
            return np.clip(out, 0, 255).astype(np.uint8)

        return clip.transform(lambda gf, t: blend(gf(t)))

    def _add_background_music(
        self,
        video: CompositeVideoClip,
        music_path: Path,
        volume: float = 0.25,
        fade_duration: float = 2.0,
    ) -> CompositeVideoClip:
        if not music_path.exists():
            logger.warning("Música de fundo não encontrada: %s", music_path)
            return video

        music = AudioFileClip(str(music_path))
        fade = min(fade_duration, music.duration * 0.3, video.duration * 0.2)
        music = music.fx(AudioFadeIn(fade)).fx(AudioFadeOut(fade)).volumex(volume)
        loops = []
        t = 0.0
        while t < video.duration:
            loops.append(music.with_start(t))
            t += music.duration
        looped = CompositeAudioClip(loops).subclip(0, video.duration)
        final_audio = CompositeAudioClip([video.audio, looped] if video.audio else [looped])
        return video.with_audio(final_audio)

    def build_video(
        self,
        visual_paths: List[Path],
        audio_paths: List[Path],
        output_path: Path,
        background_music_path: Optional[Path] = None,
        music_volume: float = 0.25,
        scene_effects: Optional[Sequence[str]] = None,
        scene_word_timings: Optional[Sequence[List[WordTiming]]] = None,
    ) -> Path:
        if len(visual_paths) != len(audio_paths):
            raise ValueError("Número de visuais e áudios precisa ser igual.")

        duration = 0
        clips = []
        for index, (visual_path, audio_path) in enumerate(zip(visual_paths, audio_paths)):
            audio_clip = AudioFileClip(str(audio_path))
            animation = scene_effects[index] if scene_effects and index < len(scene_effects) else None
            video_clip = self._build_visual_clip(
                visual_path=visual_path,
                duration=audio_clip.duration,
                animation_type=animation,
            )
            video_clip = video_clip.with_audio(audio_clip)
            timings = scene_word_timings[index] if scene_word_timings and index < len(scene_word_timings) else []
            if timings:
                video_clip = self._apply_karaoke_overlay(video_clip, timings)
            video_clip = self._apply_transition_effects(video_clip, index)
            clips.append(video_clip)
            duration += audio_clip.duration

        final_video = concatenate_videoclips(clips, method="compose")
        if background_music_path:
            final_video = self._add_background_music(
                final_video, background_music_path, volume=music_volume
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec=CODEC,
            audio_codec=AUDIO_CODEC,
            threads=4,
            preset="medium",
        )
        final_video.close()
        for clip in clips:
            clip.close()
        return output_path

    def _apply_karaoke_overlay(self, clip: ImageClip, timings: List[WordTiming]) -> CompositeVideoClip:
        base_text = " ".join([w.text.strip().upper() for w in timings if w.text.strip()])
        if not base_text or clip.duration <= 0:
            return clip

        font_size = max(28, int(self.height * 0.035))
        y_pos = max(10, int(self.height * 0.78))

        base_clip = self._make_text_clip(
            base_text,
            font_size=font_size,
            color="white",
            stroke_color="#000000",
            stroke_width=3,
            opacity=0.85,
        ).with_position(("center", y_pos)).with_duration(clip.duration)

        highlight_clips = []
        for word in timings:
            txt = word.text.strip()
            if not txt:
                continue
            start = max(0.0, min(word.start, clip.duration))
            end = max(start + 0.05, min(clip.duration, start + word.duration))
            if end <= start:
                continue
            highlight = self._make_text_clip(
                txt.upper(),
                font_size=font_size + 6,
                color="#ffd166",
                stroke_color="#111111",
                stroke_width=3,
                opacity=1.0,
            ).with_position(("center", y_pos)).with_start(start).with_duration(end - start)
            highlight_clips.append(highlight)

        overlay = CompositeVideoClip([base_clip, *highlight_clips], size=(self.width, self.height)).with_duration(clip.duration)
        combined = CompositeVideoClip([clip, overlay.with_position((0, 0))], size=(self.width, self.height)).with_duration(clip.duration)
        if clip.audio:
            combined = combined.with_audio(clip.audio)
        return combined

    def _make_text_clip(
        self,
        text: str,
        font_size: int,
        color: str,
        stroke_color: str,
        stroke_width: int,
        opacity: float = 1.0,
    ) -> TextClip:
        base_kwargs = dict(
            text=text,
            font_size=font_size,
            color=color,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            method="label",
            text_align="center",
            horizontal_align="center",
            vertical_align="center",
        )
        fonts = ["Arial-Bold", "DejaVuSans-Bold", "Arial"]
        for font_name in fonts:
            try:
                clip = TextClip(font=font_name, **base_kwargs)
                clip = clip.with_opacity(opacity)
                return clip
            except Exception:
                continue
        clip = TextClip(**base_kwargs)
        return clip.with_opacity(opacity)

    def _build_visual_clip(
        self,
        visual_path: Path,
        duration: float,
        animation_type: Optional[str] = None,
    ) -> ImageClip:
        ext = visual_path.suffix.lower()
        if ext in [".mp4", ".mov", ".mkv", ".webm", ".avi"]:
            clip = VideoFileClip(str(visual_path)).without_audio()
            clip_duration = float(getattr(clip, "duration", 0) or 0)
            if clip_duration and clip_duration < duration:
                loops = int(math.ceil(duration / clip_duration))
                clip = concatenate_videoclips([clip] * loops, method="chain")
            clip = clip.subclipped(0, duration)
            clip = self._resize_to_cover(clip)
        else:
            clip = ImageClip(str(visual_path)).with_duration(duration)
            clip = self._resize_to_cover(clip)
        animation = animation_type or self._default_animation_type(ext)
        clip = self._apply_animation(clip, animation)
        return self._apply_color_filter(clip)

    def _default_animation_type(self, ext: str) -> str:
        return (
            ANIMATION_STYLE
            if ANIMATION_STYLE and ANIMATION_STYLE != "mixed"
            else ANIMATION_TYPES[hash(ext) % len(ANIMATION_TYPES)] if ANIMATION_TYPES else "zoom_in"
        )

    def _apply_animation(self, clip: ImageClip, animation_type: Optional[str]) -> ImageClip:
        animation_type = (animation_type or "").lower()
        if animation_type == "zoom_out":
            return self._apply_zoom(clip, zoom_in=False)
        if animation_type == "zoom_in_fast":
            return self._apply_zoom(clip, zoom_in=True, intensity=ANIMATION_INTENSITY * 1.5)
        if animation_type == "zoom_out_fast":
            return self._apply_zoom(clip, zoom_in=False, intensity=ANIMATION_INTENSITY * 1.5)
        if animation_type == "kenburns":
            return self._apply_kenburns(clip)
        if animation_type == "pan_left":
            return self._apply_pan(clip, direction="left")
        if animation_type == "pan_right":
            return self._apply_pan(clip, direction="right")
        if animation_type == "pan_up":
            return self._apply_pan(clip, direction="up")
        if animation_type == "pan_down":
            return self._apply_pan(clip, direction="down")
        if animation_type == "rotate_left":
            return self._apply_rotate(clip, direction=-1)
        if animation_type == "rotate_right":
            return self._apply_rotate(clip, direction=1)
        if animation_type == "sway":
            return self._apply_sway(clip)
        if animation_type in {"warp", "warp_in", "warp_out"}:
            direction = -1 if animation_type == "warp_out" else 1
            return self._apply_warp(clip, direction=direction)
        if animation_type == "pulse":
            return self._apply_pulse(clip)
        return self._apply_zoom(clip, zoom_in=True)

    def _apply_zoom(self, clip: ImageClip, zoom_in: bool = True, intensity: float = ANIMATION_INTENSITY) -> ImageClip:
        intensity = max(0.02, float(intensity))

        def zoom_effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration, 0.0, 1.0) if clip.duration else 0.0
            delta = intensity * (progress if zoom_in else (1.0 - progress))
            factor = 1.0 + delta
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * factor), int(pil_img.height * factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(zoom_effect, keep_duration=True)

    def _apply_pan(self, clip: ImageClip, direction: str = "left") -> ImageClip:
        direction = direction.lower()
        zoom_boost = 1.0 + (ANIMATION_INTENSITY * 0.6)

        def pan_effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration, 0.0, 1.0) if clip.duration else 0.0
            h, w = frame.shape[:2]
            scaled_w = int(w * zoom_boost)
            scaled_h = int(h * zoom_boost)
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((scaled_w, scaled_h), Image.LANCZOS)
            max_x = max(0, scaled_w - w)
            max_y = max(0, scaled_h - h)

            offset_x = int(max_x * progress)
            offset_y = int(max_y * progress)
            if direction == "right":
                offset_x = max_x - offset_x
            if direction == "up":
                offset_y = max_y - offset_y
            if direction == "down":
                offset_y = offset_y

            cropped = pil_img.crop((offset_x, offset_y, offset_x + w, offset_y + h))
            return np.array(cropped)

        return clip.transform(pan_effect, keep_duration=True)

    def _apply_kenburns(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            progress = np.clip(t / clip.duration if clip.duration else 0.0, 0.0, 1.0)
            # smoothstep para movimento suave
            progress = progress * progress * (3 - 2 * progress)
            scale = 1.0 + ANIMATION_INTENSITY * (0.8 + progress)
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

            max_x = max(0, new_size[0] - frame.shape[1])
            max_y = max(0, new_size[1] - frame.shape[0])
            x = int(max_x * (0.2 + 0.6 * progress))
            y = int(max_y * (0.15 + 0.7 * progress))
            return np.array(pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0])))

        return clip.transform(effect, keep_duration=True)

    def _apply_rotate(self, clip: ImageClip, direction: int = 1) -> ImageClip:
        direction = 1 if direction >= 0 else -1
        max_angle = 4.0 * ANIMATION_INTENSITY * 10

        def effect(gf, t):
            frame = gf(t)
            angle = direction * max_angle * math.sin(2 * math.pi * (t / max(clip.duration or 0.001, 0.001)))
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(effect, keep_duration=True)

    def _apply_sway(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            h, w = frame.shape[:2]
            shift = int(np.sin(2 * np.pi * (t / max(clip.duration or 0.001, 0.001))) * w * 0.02)
            return np.roll(frame, shift, axis=1)

        return clip.transform(effect, keep_duration=True)

    def _apply_warp(self, clip: ImageClip, direction: int = 1) -> ImageClip:
        direction = 1 if direction >= 0 else -1
        wobble = ANIMATION_INTENSITY * 0.35

        def effect(gf, t):
            frame = gf(t)
            phase = 2 * np.pi * (t / max(clip.duration or 0.001, 0.001))
            scale = 1.0 + wobble * np.sin(phase) * direction
            angle = 3.0 * wobble * 50 * np.sin(phase * 0.5) * direction
            from PIL import Image

            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((int(pil_img.width * scale), int(pil_img.height * scale)), Image.LANCZOS)
            pil_img = pil_img.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            cropped = pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0]))
            return np.array(cropped)

        return clip.transform(effect, keep_duration=True)

    def _apply_pulse(self, clip: ImageClip) -> ImageClip:
        def effect(gf, t):
            frame = gf(t)
            progress = np.sin(2 * np.pi * (t / max(clip.duration or 0.001, 0.001)))
            factor = 1.0 + ANIMATION_INTENSITY * 0.3 * progress
            from PIL import Image

            pil_img = Image.fromarray(frame)
            new_size = (int(pil_img.width * factor), int(pil_img.height * factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            x = (pil_img.width - frame.shape[1]) // 2
            y = (pil_img.height - frame.shape[0]) // 2
            return np.array(pil_img.crop((x, y, x + frame.shape[1], y + frame.shape[0])))

        return clip.transform(effect, keep_duration=True)
