from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

TICKS_PER_SECOND = 10_000_000


@dataclass
class WordTiming:
    text: str
    start: float
    duration: float


def word_timings_from_chunks(chunks: Iterable[Dict[str, Any]]) -> List[WordTiming]:
    timings: List[WordTiming] = []
    for chunk in chunks:
        if chunk.get("type") != "WordBoundary":
            continue
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        try:
            offset = int(chunk.get("offset", 0))
            duration = int(chunk.get("duration", 0))
        except Exception:
            continue
        start = offset / TICKS_PER_SECOND
        length = max(duration / TICKS_PER_SECOND, 0.05)
        timings.append(WordTiming(text=text, start=start, duration=length))
    timings.sort(key=lambda item: item.start)
    return timings
