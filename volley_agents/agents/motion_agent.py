from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from volley_agents.core.event import Event, EventType

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - opzionale
    cv2 = None  # type: ignore

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline


@dataclass
class FrameSample:
    """
    Wrapper minimale per i frame video con timestamp.
    """

    time: float
    frame: np.ndarray


@dataclass
class OpticalFlowConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    magnitude_threshold: float = 1.8
    hit_ratio: float = 1.35
    gap_threshold: float = 0.35
    gap_min_duration: float = 0.35
    min_hit_cooldown: float = 0.25


class MotionAgent:
    """
    Usa l'optical flow (Farneback) per dedurre impatti e pause.
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None):
        self.config = config or OpticalFlowConfig()
        self._last_hit_time: Optional[float] = None

    def run(
        self,
        frames: Sequence[FrameSample],
        timeline: Optional["Timeline"] = None,
    ) -> List[Event]:
        events = self.analyze(frames)
        if timeline is not None:
            timeline.extend(events)
        return events

    def analyze(self, frames: Sequence[FrameSample]) -> List[Event]:
        if len(frames) < 2:
            return []

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) non disponibile: installa opencv-python.")

        cfg = self.config
        events: List[Event] = []

        prev_gray = self._to_gray(frames[0].frame)
        gap_start: Optional[float] = None
        for sample in frames[1:]:
            curr_gray = self._to_gray(sample.frame)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                cfg.pyr_scale,
                cfg.levels,
                cfg.winsize,
                cfg.iterations,
                cfg.poly_n,
                cfg.poly_sigma,
                0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            mean_mag = float(np.mean(magnitude))
            left_mag = float(np.mean(magnitude[:, : magnitude.shape[1] // 2]))
            right_mag = float(np.mean(magnitude[:, magnitude.shape[1] // 2 :]))

            hit_event = self._detect_hit(sample.time, left_mag, right_mag)
            if hit_event:
                events.append(hit_event)

            if mean_mag <= cfg.gap_threshold:
                gap_start = gap_start or sample.time
            else:
                if gap_start is not None and sample.time - gap_start >= cfg.gap_min_duration:
                    events.append(
                        Event(
                            time=(gap_start + sample.time) / 2,
                            type=EventType.MOTION_GAP,
                            confidence=0.6,
                            extra={"duration": sample.time - gap_start, "mean_mag": mean_mag},
                        )
                    )
                gap_start = None

            prev_gray = curr_gray

        return events

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            if cv2 is None:
                raise RuntimeError("cv2 richiesto per convertire frame RGB in scala di grigi.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("Frame non supportato: atteso HxW o HxWx3.")

    def _detect_hit(self, time_sec: float, left_mag: float, right_mag: float) -> Optional[Event]:
        cfg = self.config
        cooldown_ok = (
            self._last_hit_time is None or time_sec - self._last_hit_time >= cfg.min_hit_cooldown
        )
        if not cooldown_ok:
            return None

        if left_mag >= cfg.magnitude_threshold and left_mag >= right_mag * cfg.hit_ratio:
            self._last_hit_time = time_sec
            confidence = min(1.0, left_mag / (cfg.magnitude_threshold * 1.5))
            return Event(
                time=time_sec,
                type=EventType.HIT_LEFT,
                confidence=confidence,
                extra={"left_mag": left_mag, "right_mag": right_mag},
            )

        if right_mag >= cfg.magnitude_threshold and right_mag >= left_mag * cfg.hit_ratio:
            self._last_hit_time = time_sec
            confidence = min(1.0, right_mag / (cfg.magnitude_threshold * 1.5))
            return Event(
                time=time_sec,
                type=EventType.HIT_RIGHT,
                confidence=confidence,
                extra={"left_mag": left_mag, "right_mag": right_mag},
            )

        return None


__all__ = [
    "FrameSample",
    "MotionAgent",
    "OpticalFlowConfig",
]

