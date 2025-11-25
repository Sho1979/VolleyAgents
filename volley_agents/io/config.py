"""Utility per leggere/scrivere configurazioni persistenti (ROI, parametri agenti)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


RoiTuple = Tuple[int, int, int, int]


def _normalize_roi(raw_roi: Any) -> Optional[RoiTuple]:
    """
    Converte un oggetto ROI generico (lista, dict, tuple) in un tuple di int.
    Restituisce None se il formato non è compatibile.
    """
    if isinstance(raw_roi, dict):
        keys = ("x", "y", "w", "h")
        if not all(k in raw_roi for k in keys):
            return None
        return tuple(int(raw_roi[k]) for k in keys)  # type: ignore

    if isinstance(raw_roi, Sequence) and not isinstance(raw_roi, (str, bytes)) and len(raw_roi) == 4:
        return tuple(int(v) for v in raw_roi)  # type: ignore

    return None


def load_scoreboard_roi_config(
    path: Path | str,
    *,
    expected_video: Optional[str] = None,
) -> Optional[RoiTuple]:
    """
    Carica la ROI principale del tabellone da file JSON.

    Args:
        path: percorso del file JSON (scoreboard_config.json, ecc.)
        expected_video: opzionale, nome del video da verificare per evitare mismatch

    Returns:
        Tuple (x, y, w, h) oppure None se il file non esiste/non è valido.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return None

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if expected_video:
        video_name = data.get("video_name") or Path(data.get("video_path", "")).name
        if video_name and video_name != Path(expected_video).name:
            return None

    roi = (
        data.get("scoreboard_main_roi")
        or data.get("scoreboard_roi")
        or data.get("roi")
    )
    return _normalize_roi(roi)


def save_scoreboard_roi_config(
    path: Path | str,
    roi: RoiTuple,
    *,
    video_path: Optional[Path | str] = None,
    led_color: Optional[str] = "red",
) -> Path:
    """
    Salva la ROI principale del tabellone in un file JSON riutilizzabile.

    Restituisce il percorso del file salvato.
    """
    cfg_path = Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    video_name = None
    video_path_str = None
    if video_path:
        video_path = Path(video_path)
        video_name = video_path.name
        video_path_str = str(video_path)

    payload: Dict[str, Any] = {
        "version": 1,
        "format": "scoreboard_v3_roi",
        "scoreboard_main_roi": list(map(int, roi)),
        "led_color": led_color,
        "video_name": video_name,
        "video_path": video_path_str,
    }

    cfg_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cfg_path


__all__ = [
    "RoiTuple",
    "load_scoreboard_roi_config",
    "save_scoreboard_roi_config",
]

