"""
BallAgent: tracking palla con YOLO o TrackNet.

Output:
- Posizione palla (x, y) per ogni frame
- Velocità e direzione
- Eventi: tocco terra, fuori, attraversa rete
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING, Callable

import numpy as np

from volley_agents.core.event import Event, EventType
from volley_agents.agents.motion_agent import FrameSample

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline
    from volley_agents.calibration.field_auto import FieldAutoCalibrator

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class BallAgentConfig:
    """Configurazione per BallAgent."""

    # Modello da usare
    model_type: str = "yolo"  # "yolo" o "tracknet"
    model_path: str = "yolov8n.pt"  # Path al modello

    # Classe palla per detection
    # Per COCO dataset: 32 = "sports ball"
    # Per modelli custom: specificare la classe corrispondente (es: 0 per YOLOv10 volleyball)
    ball_class_id: int = 32  # Classe COCO di default
    use_custom_ball_class: bool = False  # Se True, usa ball_class_id invece di 32

    # Soglie detection
    confidence_threshold: float = 0.3

    # Parametri Kalman Filter per smoothing
    use_kalman: bool = True
    process_noise: float = 0.1
    measurement_noise: float = 0.5

    # Ground detection
    ground_y_threshold: float = 0.9  # Y normalizzato per considerare "terra"

    # Zone detection thresholds (normalizzati 0-1)
    service_left_threshold: float = 0.15  # X < 0.15 = service_left
    service_right_threshold: float = 0.85  # X > 0.85 = service_right
    net_left_threshold: float = 0.45  # 0.45 < X < 0.55 = net
    net_right_threshold: float = 0.55

    # Velocità massima ragionevole (pixel/frame)
    max_speed: float = 200.0

    # Logging
    enable_logging: bool = False
    log_callback: Optional[Callable[[str], None]] = None
    log_every_n_frames: int = 50  # Log dettagli ogni N frame


class BallTracker:
    """
    Tracker interno con Kalman Filter per smoothing.
    """

    def __init__(self, config: BallAgentConfig):
        self.config = config
        self._last_pos: Optional[Tuple[float, float]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._lost_frames: int = 0

    def update(self, detection: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Aggiorna il tracker con nuova detection.

        Args:
            detection: (x, y) della palla o None se non rilevata

        Returns:
            Posizione stimata (x, y) o None
        """
        if detection is not None:
            if self._last_pos is not None:
                # Calcola velocità
                dx = detection[0] - self._last_pos[0]
                dy = detection[1] - self._last_pos[1]

                # Filtra velocità anomale
                speed = np.sqrt(dx * dx + dy * dy)
                if speed < self.config.max_speed:
                    self._velocity = (dx, dy)

            self._last_pos = detection
            self._lost_frames = 0
            return detection

        else:
            # Prediction quando palla non rilevata
            self._lost_frames += 1

            if self._last_pos is not None and self._lost_frames < 5:
                # Predici con velocità
                pred_x = self._last_pos[0] + self._velocity[0]
                pred_y = self._last_pos[1] + self._velocity[1]
                return (pred_x, pred_y)

            return None


class BallAgent:
    """
    Agente per tracking palla.

    Uso:
        agent = BallAgent()
        events = agent.run(frames, timeline)
    """

    def __init__(self, config: Optional[BallAgentConfig] = None):
        self.config = config or BallAgentConfig()
        self._model = None
        self._tracker = BallTracker(self.config)

        # Statistiche per summary finale
        self._stats = {
            "total_frames": 0,
            "detected_frames": 0,
            "zones": {"left": 0, "right": 0, "service_left": 0, "service_right": 0, "net": 0, "ground": 0},
            "events": {"detected": 0, "ground": 0, "cross_net": 0, "serve": 0, "out": 0},
        }

        if YOLO_AVAILABLE and self.config.model_type == "yolo":
            try:
                self._model = YOLO(self.config.model_path)
                ball_class_info = (
                    f"custom class {self.config.ball_class_id}"
                    if self.config.use_custom_ball_class
                    else f"COCO class 32 (sports ball)"
                )
                self._log(
                    f"[BallAgent] ✅ Modello YOLO caricato: {self.config.model_path} (ball: {ball_class_info})"
                )
            except Exception as e:
                self._log(f"[BallAgent] ⚠️ Errore nel caricare modello YOLO: {e}")
                self._model = None

    def _log(self, message: str):
        """Log interno."""
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(message)
            else:
                print(message)

    def get_zone(self, x: float, y: float, w: int, h: int) -> str:
        """
        Determina la zona della palla nel campo.

        Args:
            x, y: Coordinate palla in pixel
            w, h: Dimensioni frame (width, height)

        Returns:
            Zona: "ground", "service_left", "service_right", "net", "left", "right"
        """
        x_norm = x / w if w > 0 else 0.5
        y_norm = y / h if h > 0 else 0.5

        # Terra
        if y_norm > self.config.ground_y_threshold:
            return "ground"

        # Zone servizio
        if x_norm < self.config.service_left_threshold:
            return "service_left"
        if x_norm > self.config.service_right_threshold:
            return "service_right"

        # Rete (centro campo)
        if self.config.net_left_threshold < x_norm < self.config.net_right_threshold:
            return "net"

        # Lati
        return "left" if x_norm < 0.5 else "right"

    def run(
        self,
        frames: List[FrameSample],
        timeline: Optional["Timeline"] = None,
        field_calibrator: Optional["FieldAutoCalibrator"] = None,
    ) -> List[Event]:
        """
        Analizza frames e traccia la palla.

        Args:
            frames: Lista di FrameSample
            timeline: Timeline per aggiungere eventi
            field_calibrator: Calibratore campo per coordinate metri

        Returns:
            Lista di eventi palla
        """
        events = []

        if self._model is None:
            self._log("[BallAgent] ⚠️ Modello non caricato, salto tracking palla")
            return events

        # Reset statistiche
        self._stats = {
            "total_frames": len(frames),
            "detected_frames": 0,
            "zones": {"left": 0, "right": 0, "service_left": 0, "service_right": 0, "net": 0, "ground": 0},
            "events": {"detected": 0, "ground": 0, "cross_net": 0, "serve": 0, "out": 0},
        }

        positions = []

        # Determina classe palla da usare
        target_class = self.config.ball_class_id if self.config.use_custom_ball_class else 32

        for idx, frame_sample in enumerate(frames):
            t = frame_sample.time
            frame = frame_sample.frame
            self._stats["total_frames"] += 1

            # Detection con YOLO
            results = self._model(frame, verbose=False)

            ball_pos = None
            ball_conf = 0.0
            ball_box = None

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Usa classe target (COCO 32 o custom)
                    if cls == target_class and conf > self.config.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2

                        if conf > ball_conf:
                            ball_pos = (cx, cy)
                            ball_conf = conf
                            ball_box = (x1, y1, x2, y2)

            # Aggiorna tracker
            tracked_pos = self._tracker.update(ball_pos)

            if tracked_pos is not None:
                self._stats["detected_frames"] += 1
                h, w = frame.shape[:2]
                zone = self.get_zone(tracked_pos[0], tracked_pos[1], w, h)
                self._stats["zones"][zone] = self._stats["zones"].get(zone, 0) + 1

                positions.append((t, tracked_pos[0], tracked_pos[1], ball_conf, zone))

                # Converti in metri se calibrato
                meters_pos = None
                if field_calibrator is not None:
                    meters_pos = field_calibrator.pixel_to_meters(tracked_pos[0], tracked_pos[1])

                self._stats["events"]["detected"] += 1
                events.append(
                    Event(
                        time=t,
                        type=EventType.BALL_DETECTED,
                        confidence=ball_conf,
                        extra={
                            "x": tracked_pos[0],
                            "y": tracked_pos[1],
                            "zone": zone,
                            "x_meters": meters_pos[0] if meters_pos else None,
                            "y_meters": meters_pos[1] if meters_pos else None,
                        },
                    )
                )

                # Log dettagliato ogni N frame
                if self.config.enable_logging and idx % self.config.log_every_n_frames == 0:
                    self._log(
                        f"  Frame {idx}/{len(frames)} | t={t:.2f}s | "
                        f"Palla: ({tracked_pos[0]:.0f}, {tracked_pos[1]:.0f}) | "
                        f"Zone: {zone} | Conf: {ball_conf:.2f}"
                    )

                # Check ground touch
                if zone == "ground":
                    self._stats["events"]["ground"] += 1
                    events.append(
                        Event(
                            time=t,
                            type=EventType.BALL_TOUCH_GROUND,
                            confidence=ball_conf * 0.8,
                            extra={
                                "x": tracked_pos[0],
                                "y": tracked_pos[1],
                                "zone": zone,
                            },
                        )
                    )

        # Analisi traiettoria per eventi derivati
        trajectory_events = self._analyze_trajectory(positions, frames)
        events.extend(trajectory_events)

        # Update stats per eventi traiettoria
        for evt in trajectory_events:
            if evt.type == EventType.BALL_CROSS_NET:
                self._stats["events"]["cross_net"] += 1
            elif evt.type == EventType.BALL_SERVE:
                self._stats["events"]["serve"] += 1
            elif evt.type == EventType.BALL_OUT:
                self._stats["events"]["out"] += 1

        if timeline is not None:
            timeline.extend(events)

        # Log summary finale
        if self.config.enable_logging:
            self._log_summary()

        return events

    def _log_summary(self):
        """Log statistiche finali."""
        stats = self._stats
        total = stats["total_frames"]
        detected = stats["detected_frames"]
        detection_rate = (detected / total * 100) if total > 0 else 0.0

        self._log(f"[BallAgent] 📊 Summary:")
        self._log(f"  Detection rate: {detection_rate:.1f}% ({detected}/{total} frame)")
        self._log(f"  Eventi generati: {sum(stats['events'].values())}")
        self._log(f"    - BALL_DETECTED: {stats['events']['detected']}")
        self._log(f"    - BALL_TOUCH_GROUND: {stats['events']['ground']}")
        self._log(f"    - BALL_CROSS_NET: {stats['events']['cross_net']}")
        self._log(f"    - BALL_SERVE: {stats['events']['serve']}")
        self._log(f"    - BALL_OUT: {stats['events']['out']}")
        self._log(f"  Zone breakdown:")
        for zone, count in stats["zones"].items():
            if count > 0:
                pct = (count / detected * 100) if detected > 0 else 0.0
                self._log(f"    - {zone}: {count} ({pct:.1f}%)")

    def _analyze_trajectory(
        self, positions: List[Tuple], frames: List[FrameSample]
    ) -> List[Event]:
        """
        Analizza traiettoria per eventi derivati:
        - BALL_CROSS_NET: quando palla attraversa la rete (centro campo)
        - BALL_SERVE: quando palla parte da zona servizio
        - BALL_OUT: quando palla esce dai limiti campo
        """
        events = []

        if len(positions) < 2:
            return events

        # Estrai dimensioni frame per normalizzazione
        if not frames:
            return events
        h, w = frames[0].frame.shape[:2]

        # Traccia attraversamento rete
        prev_zone = None
        prev_side = None  # "left" o "right"

        for idx, pos_data in enumerate(positions):
            if len(pos_data) < 5:
                continue

            t, x, y, conf, zone = pos_data
            current_side = "left" if x / w < 0.5 else "right"

            # BALL_CROSS_NET: rileva quando attraversa centro (45%-55%)
            if prev_side is not None and prev_side != current_side:
                # Palla ha attraversato il centro
                prev_x = positions[idx - 1][1] if idx > 0 else x
                # Verifica che sia effettivamente passata per la zona rete
                if (
                    self.config.net_left_threshold * w <= (x + prev_x) / 2 <= self.config.net_right_threshold * w
                ):
                    events.append(
                        Event(
                            time=t,
                            type=EventType.BALL_CROSS_NET,
                            confidence=conf,
                            extra={
                                "x": x,
                                "y": y,
                                "direction": f"{prev_side}->{current_side}",
                            },
                        )
                    )

            # BALL_SERVE: rileva quando palla parte da zona servizio e si muove verso centro
            if zone in ("service_left", "service_right"):
                # Cerca movimento verso centro nei frame successivi
                lookahead = min(5, len(positions) - idx - 1)
                for j in range(1, lookahead + 1):
                    if idx + j >= len(positions):
                        break
                    next_t, next_x, next_y, next_conf, next_zone = positions[idx + j]
                    # Serve: parte da servizio e si sposta verso centro/altra metà
                    if (
                        zone == "service_left"
                        and next_x > x
                        and (next_zone in ("left", "net", "right") or next_x / w > self.config.service_left_threshold)
                    ) or (
                        zone == "service_right"
                        and next_x < x
                        and (next_zone in ("right", "net", "left") or next_x / w < self.config.service_right_threshold)
                    ):
                        # Serve rilevato
                        events.append(
                            Event(
                                time=t,
                                type=EventType.BALL_SERVE,
                                confidence=min(conf, next_conf),
                                extra={
                                    "x": x,
                                    "y": y,
                                    "zone": zone,
                                    "next_x": next_x,
                                    "next_zone": next_zone,
                                },
                            )
                        )
                        break  # Evita duplicati

            # BALL_OUT: rileva quando palla esce dai limiti campo (bordi frame o zone anomale)
            # Considera "out" se palla va molto oltre i bordi normali del campo
            margin = 0.05  # 5% margine
            if x < -margin * w or x > (1 + margin) * w or y < -margin * h or y > (1 + margin) * h:
                events.append(
                    Event(
                        time=t,
                        type=EventType.BALL_OUT,
                        confidence=conf * 0.7,
                        extra={
                            "x": x,
                            "y": y,
                            "zone": zone,
                        },
                    )
                )

            prev_zone = zone
            prev_side = current_side

        return events


__all__ = [
    "BallAgent",
    "BallAgentConfig",
]

