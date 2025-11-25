from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from volley_agents.core.event import Event, EventType

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - opzionale
    cv2 = None  # type: ignore

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline

# Import FrameSample per evitare TYPE_CHECKING
from volley_agents.agents.motion_agent import FrameSample


@dataclass
class ServeAgentConfig:
    """
    Parametri per la rilevazione delle battute nella zona battuta (ROI SX/DX).
    """

    # Soglie per motion nella ROI
    roi_motion_threshold: float = 4.0  # maggiore selettività sulla ROI
    roi_motion_ratio: float = 2.0  # richiede ROI molto più attiva del resto
    roi_motion_percentile: float = 95.0  # usa percentile invece di media per rilevare spike

    # Matching con HIT
    max_hit_delay: float = 1.0  # max secondi tra serve e hit
    min_serve_cooldown: float = 3.0  # min secondi tra serve consecutivi

    # Calibrazione con referee
    use_referee_calibration: bool = True  # usa REF_SERVE_READY/RELEASE per calibrare timing
    referee_serve_window: float = 3.0  # finestra temporale dopo REF_SERVE_READY per cercare serve
    referee_release_window: float = 1.0  # finestra temporale dopo REF_SERVE_RELEASE per cercare serve

    # Parametri optical flow (riutilizzati da MotionAgent)
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 21
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

    # Logging
    enable_logging: bool = False
    log_callback: Optional[Callable[[str], None]] = None


class ServeAgent:
    """
    Analizza la zona battuta (ROI SX/DX) per rilevare motion spike
    e matchare con HIT per identificare SERVE_START.
    """

    def __init__(self, config: Optional[ServeAgentConfig] = None):
        self.config = config or ServeAgentConfig()
        self._last_serve_time: Optional[float] = None
        self._referee_ready_times: List[float] = []  # timestamp di REF_SERVE_READY
        self._referee_release_times: List[float] = []  # timestamp di REF_SERVE_RELEASE

    def _log(self, message: str):
        """Log interno."""
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(message)
            else:
                print(message)

    def run(
        self,
        frames: Sequence["FrameSample"],
        roi_left: Optional[Tuple[int, int, int, int]] = None,
        roi_right: Optional[Tuple[int, int, int, int]] = None,
        timeline: Optional["Timeline"] = None,
    ) -> List[Event]:
        """
        Analizza i frame per rilevare battute nella zona battuta.

        Args:
            frames: Sequenza di FrameSample da analizzare
            roi_left: ROI zona battuta sinistra (x, y, w, h) o None per disattivare
            roi_right: ROI zona battuta destra (x, y, w, h) o None per disattivare
            timeline: Timeline opzionale per estrarre eventi HIT esistenti

        Returns:
            Lista di eventi SERVE_START rilevati
        """
        events = self.analyze(frames, roi_left, roi_right, timeline)
        if timeline is not None:
            timeline.extend(events)
        return events

    def analyze(
        self,
        frames: Sequence["FrameSample"],
        roi_left: Optional[Tuple[int, int, int, int]] = None,
        roi_right: Optional[Tuple[int, int, int, int]] = None,
        timeline: Optional["Timeline"] = None,
    ) -> List[Event]:
        """
        Analizza i frame per motion spike nella ROI e matcha con HIT.
        """
        if len(frames) < 2:
            return []

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) non disponibile: installa opencv-python.")

        # Se le ROI non sono definite, disattivarsi automaticamente
        if roi_left is None and roi_right is None:
            return []

        cfg = self.config
        events: List[Event] = []

        # Estrai eventi dalla timeline se disponibile
        hits: List[Event] = []
        referee_ready: List[Event] = []
        referee_release: List[Event] = []
        if timeline is not None:
            all_events = timeline.sorted()
            hits = [
                e
                for e in all_events
                if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)
            ]
            if cfg.use_referee_calibration:
                referee_ready = [
                    e for e in all_events if e.type == EventType.REF_SERVE_READY
                ]
                referee_release = [
                    e for e in all_events if e.type == EventType.REF_SERVE_RELEASE
                ]
                # Memorizza timestamp referee per calibrazione
                self._referee_ready_times = [e.time for e in referee_ready]
                self._referee_release_times = [e.time for e in referee_release]
                if referee_ready:
                    self._log(f"🎯 ServeAgent: trovati {len(referee_ready)} REF_SERVE_READY per calibrazione")
                if referee_release:
                    self._log(f"🎯 ServeAgent: trovati {len(referee_release)} REF_SERVE_RELEASE per calibrazione")

        prev_gray = self._to_gray(frames[0].frame)
        for sample in frames[1:]:
            curr_gray = self._to_gray(sample.frame)

            # Calcola optical flow
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

            h, w = magnitude.shape

            # Estrai motion dalla ROI sinistra se definita (usa percentile per rilevare spike)
            roi_left_mag = 0.0
            roi_left_mag_percentile = 0.0
            if roi_left is not None:
                x, y, roi_w, roi_h = roi_left
                # Assicura che la ROI sia dentro il frame
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                roi_w = min(roi_w, w - x)
                roi_h = min(roi_h, h - y)
                if roi_w > 0 and roi_h > 0:
                    roi_left_region = magnitude[y : y + roi_h, x : x + roi_w]
                    if roi_left_region.size > 0:
                        roi_left_mag = float(np.mean(roi_left_region))
                        # Usa percentile invece di media per rilevare spike localizzati
                        roi_left_mag_percentile = float(
                            np.percentile(roi_left_region, cfg.roi_motion_percentile)
                        )

            # Estrai motion dalla ROI destra se definita (usa percentile per rilevare spike)
            roi_right_mag = 0.0
            roi_right_mag_percentile = 0.0
            if roi_right is not None:
                x, y, roi_w, roi_h = roi_right
                # Assicura che la ROI sia dentro il frame
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                roi_w = min(roi_w, w - x)
                roi_h = min(roi_h, h - y)
                if roi_w > 0 and roi_h > 0:
                    roi_right_region = magnitude[y : y + roi_h, x : x + roi_w]
                    if roi_right_region.size > 0:
                        roi_right_mag = float(np.mean(roi_right_region))
                        # Usa percentile invece di media per rilevare spike localizzati
                        roi_right_mag_percentile = float(
                            np.percentile(roi_right_region, cfg.roi_motion_percentile)
                        )

            # Motion globale per normalizzazione (media di tutto il frame esclusa le ROI)
            # Per semplicità, usiamo la media di tutto il frame
            center_mag = float(np.mean(magnitude)) if magnitude.size > 0 else 1.0

            # Rileva serve nella ROI sinistra se definita
            if roi_left is not None:
                serve_event = self._detect_serve(
                    sample.time,
                    roi_left_mag,
                    roi_left_mag_percentile,  # usa percentile per rilevazione spike
                    roi_right_mag if roi_right is not None else 0.0,
                    center_mag,
                    "left",
                    hits,
                    referee_ready,
                    referee_release,
                )
                if serve_event:
                    events.append(serve_event)
                    self._last_serve_time = serve_event.time
                    self._log(
                        f"🎯 ServeAgent: rilevato SERVE_START left @ {serve_event.time:.2f}s "
                        f"(conf={serve_event.confidence:.2f}, mag={roi_left_mag:.2f}, "
                        f"percentile={roi_left_mag_percentile:.2f})"
                    )

            # Rileva serve nella ROI destra se definita
            if roi_right is not None:
                serve_event = self._detect_serve(
                    sample.time,
                    roi_right_mag,
                    roi_right_mag_percentile,  # usa percentile per rilevazione spike
                    roi_left_mag if roi_left is not None else 0.0,
                    center_mag,
                    "right",
                    hits,
                    referee_ready,
                    referee_release,
                )
                if serve_event:
                    events.append(serve_event)
                    self._last_serve_time = serve_event.time
                    self._log(
                        f"🎯 ServeAgent: rilevato SERVE_START right @ {serve_event.time:.2f}s "
                        f"(conf={serve_event.confidence:.2f}, mag={roi_right_mag:.2f}, "
                        f"percentile={roi_right_mag_percentile:.2f})"
                    )

            prev_gray = curr_gray

        return events

    def _detect_serve(
        self,
        time_sec: float,
        roi_mag: float,
        roi_mag_percentile: float,
        other_roi_mag: float,
        center_mag: float,
        side: str,
        hits: List[Event],
        referee_ready: List[Event],
        referee_release: List[Event],
    ) -> Optional[Event]:
        """
        Rileva serve nella ROI specificata con calibrazione referee.

        Args:
            time_sec: Timestamp corrente
            roi_mag: Magnitude motion media nella ROI target
            roi_mag_percentile: Magnitude motion percentile (95%) nella ROI target (per spike)
            other_roi_mag: Magnitude motion nell'altra ROI (per confronto)
            center_mag: Magnitude motion nel centro (per normalizzazione)
            side: "left" o "right"
            hits: Lista di eventi HIT dalla timeline
            referee_ready: Lista di eventi REF_SERVE_READY
            referee_release: Lista di eventi REF_SERVE_RELEASE

        Returns:
            Event SERVE_START se rilevato, None altrimenti
        """
        cfg = self.config

        # Cooldown check
        cooldown_ok = (
            self._last_serve_time is None
            or time_sec - self._last_serve_time >= cfg.min_serve_cooldown
        )
        if not cooldown_ok:
            return None

        # CALIBRAZIONE REFEREE: verifica se siamo in una finestra temporale dopo REF_SERVE_READY/RELEASE
        in_referee_window = False
        if cfg.use_referee_calibration:
            # Cerca REF_SERVE_READY precedente
            for ref_ready in referee_ready:
                dt = time_sec - ref_ready.time
                if 0 <= dt <= cfg.referee_serve_window:
                    in_referee_window = True
                    break

            # Cerca REF_SERVE_RELEASE precedente (finestra più stretta)
            if not in_referee_window:
                for ref_release in referee_release:
                    dt = time_sec - ref_release.time
                    if 0 <= dt <= cfg.referee_release_window:
                        in_referee_window = True
                        break

        # Usa percentile invece di media per rilevare spike localizzati
        # Il percentile è più sensibile a picchi di movimento
        motion_signal = roi_mag_percentile if roi_mag_percentile > 0 else roi_mag

        # Verifica soglia motion nella ROI (usa percentile)
        threshold = cfg.roi_motion_threshold
        if in_referee_window:
            # In finestra referee: soglia più bassa (siamo più confidenti)
            threshold *= 0.8

        if motion_signal < threshold:
            return None

        # Verifica che la ROI sia significativamente più attiva dell'altra
        if motion_signal < other_roi_mag * cfg.roi_motion_ratio:
            return None

        # Verifica che la ROI sia più attiva del centro (normalizzazione)
        if center_mag > 0 and motion_signal < center_mag * cfg.roi_motion_ratio:
            return None

        # Match con HIT successivo entro max_hit_delay
        hit_match = None
        for hit in hits:
            dt = hit.time - time_sec
            if 0 < dt <= cfg.max_hit_delay:
                # Verifica che il lato corrisponda
                if (side == "left" and hit.type == EventType.HIT_LEFT) or (
                    side == "right" and hit.type == EventType.HIT_RIGHT
                ):
                    hit_match = hit
                    break

        # Calcola confidence basata su:
        # - Match HIT (alto confidence se match)
        # - Finestra referee (aumenta confidence se in finestra)
        # - Percentile motion (maggiore percentile = maggiore confidence)
        confidence = 0.6  # base

        if hit_match is not None:
            confidence += 0.25  # match HIT

        if in_referee_window:
            confidence += 0.15  # in finestra referee

        # Normalizza confidence basata su percentile motion
        if motion_signal > threshold * 2.0:
            confidence += 0.1  # motion molto alto

        confidence = min(1.0, confidence)  # clamp a 1.0

        return Event(
            time=time_sec,
            type=EventType.SERVE_START,
            confidence=confidence,
            extra={
                "side": side,
                "roi_mag": roi_mag,
                "roi_mag_percentile": roi_mag_percentile,
                "center_mag": center_mag,
                "hit_matched": hit_match is not None,
                "hit_time": hit_match.time if hit_match else None,
                "referee_calibrated": in_referee_window,
            },
        )

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Converte frame a scala di grigi."""
        if frame.ndim == 2:
            return frame.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 3:
            if cv2 is None:
                raise RuntimeError("cv2 richiesto per convertire frame RGB in scala di grigi.")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raise ValueError("Frame non supportato: atteso HxW o HxWx3.")


__all__ = [
    "ServeAgent",
    "ServeAgentConfig",
]

