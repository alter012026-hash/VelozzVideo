from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

TICKS_PER_SECOND = 10_000_000


@dataclass
class WordTiming:
    text: str
    start: float
    duration: float


def _clean_token(text: Any) -> str:
    value = str(text or "").strip()
    value = value.replace("\n", " ").replace("\r", " ")
    value = value.replace("�", "")
    return " ".join(value.split())


def word_timings_from_chunks(chunks: Iterable[Dict[str, Any]]) -> List[WordTiming]:
    raw: List[WordTiming] = []
    for chunk in chunks:
        if chunk.get("type") != "WordBoundary":
            continue
        text = _clean_token(chunk.get("text", ""))
        if not text:
            continue
        try:
            offset = int(chunk.get("offset", 0))
            duration = int(chunk.get("duration", 0))
        except Exception:
            continue
        start = max(0.0, offset / TICKS_PER_SECOND)
        length = max(duration / TICKS_PER_SECOND, 0.05)
        raw.append(WordTiming(text=text, start=start, duration=length))

    if not raw:
        return []

    raw.sort(key=lambda item: item.start)
    timings: List[WordTiming] = []
    for idx, token in enumerate(raw):
        next_start = raw[idx + 1].start if idx + 1 < len(raw) else None
        if next_start is not None and next_start > token.start:
            span = max(0.05, next_start - token.start)
            duration = min(1.6, max(token.duration, span))
        else:
            duration = max(0.05, token.duration)
        timings.append(WordTiming(text=token.text, start=token.start, duration=duration))
    return timings


def approximate_word_timings(text: str, total_duration: float) -> List[WordTiming]:
    clean = " ".join(str(text or "").replace("\n", " ").split())
    if not clean:
        return []
    words = [_clean_token(w) for w in clean.split(" ")]
    words = [w for w in words if w]
    if not words:
        return []

    duration = max(0.25, float(total_duration or 0.0))
    total_weight = sum(max(1, len(w)) for w in words)
    cursor = 0.0
    out: List[WordTiming] = []
    for i, word in enumerate(words):
        weight = max(1, len(word))
        base = duration * (weight / total_weight)
        hold = max(0.07, base)
        if i == len(words) - 1:
            hold = max(0.07, duration - cursor)
        out.append(WordTiming(text=word, start=max(0.0, cursor), duration=hold))
        cursor += hold
    return out
