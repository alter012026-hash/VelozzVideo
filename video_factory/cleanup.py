from __future__ import annotations

"""
Rotinas de limpeza de cache/artefatos antigos gerados nos renders.
Remove imagens/áudios/vídeos temporários mais antigos que o limite
configurado em CACHE_MAX_AGE_DAYS e também arquivos temporários
TEMP_MPY_wvf_snd.mp4 deixados pelo MoviePy.
"""

import logging
import time
from pathlib import Path
from typing import Iterable, Tuple

from video_factory import config

log = logging.getLogger(__name__)


def _iter_targets() -> Iterable[Tuple[Path, str]]:
    base = config.ASSETS_DIR
    yield base / "images", "*.png"
    yield base / "audio", "*"
    yield base / "video", "*.mp4"
    # glob no root para temp do MoviePy
    yield config.ROOT_DIR, "*TEMP_MPY_wvf_snd.mp4"


def purge_old_assets(max_age_days: int | None = None) -> dict:
    """
    Remove arquivos mais antigos que max_age_days (default: config.CACHE_MAX_AGE_DAYS).
    Retorna um resumo com contagem de removidos e mantidos.
    """
    max_age_days = max_age_days or config.CACHE_MAX_AGE_DAYS
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    kept = 0
    errors: list[str] = []

    for folder, pattern in _iter_targets():
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.glob(pattern):
            if path.is_dir():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
                else:
                    kept += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")
                log.warning("Falha ao limpar %s: %s", path, exc)

    summary = {"removed": removed, "kept": kept, "max_age_days": max_age_days, "errors": errors}
    log.info("Limpeza de cache: %s", summary)
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(purge_old_assets(), indent=2, ensure_ascii=False))
