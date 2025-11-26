"""ActionAgent: Riconoscimento azioni pallavolo.

Usa modello YOLO per classificare:
- serve: battuta
- receive: ricezione
- set: palleggio/alzata
- spike: schiacciata/attacco
- block: muro
- dig: difesa/bagher

Richiede: models/action/best.pt (YOLO trained on volleyball actions)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:  # pragma: no cover - dipendenza opzionale
    YOLO_AVAILABLE = False


class ActionType(Enum):
    """Tipi di azione pallavolo."""

    SERVE = "serve"
    RECEIVE = "receive"  # Ricezione
    SET = "set"  # Palleggio/alzata
    SPIKE = "spike"  # Schiacciata
    BLOCK = "block"  # Muro
    DIG = "dig"  # Difesa/bagher
    UNKNOWN = "unknown"


@dataclass
class ActionDetection:
    """Singola azione rilevata."""

    action_type: ActionType
    time: float
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    player_id: Optional[str] = None
    side: str = "unknown"  # left/right/center
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Action({self.action_type.value}@{self.time:.1f}s, "
            f"conf={self.confidence:.2f}, side={self.side})"
        )


@dataclass
class ActionAgentConfig:
    """Configurazione ActionAgent."""

    model_path: str = "volley_agents/models/action/best.pt"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4

    # Mapping classi YOLO -> ActionType
    class_mapping: Dict[int, ActionType] = field(
        default_factory=lambda: {
            0: ActionType.SERVE,
            1: ActionType.RECEIVE,
            2: ActionType.SET,
            3: ActionType.SPIKE,
            4: ActionType.BLOCK,
            5: ActionType.DIG,
        }
    )

    # Zone campo per determinare side
    left_zone_x: float = 0.45  # x < 45% = left
    right_zone_x: float = 0.55  # x > 55% = right

    # Logging
    enable_logging: bool = True
    log_callback: Optional[Callable[[str], None]] = None


class ActionAgent:
    """
    Agente per riconoscimento azioni pallavolo.

    Usa YOLO per detection + classificazione azioni in ogni frame.
    Aggrega detections temporalmente per generare eventi azione.
    """

    def __init__(self, config: Optional[ActionAgentConfig] = None):
        self.config = config or ActionAgentConfig()
        self.model: Any = None
        self._load_model()

    def log(self, msg: str) -> None:
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(msg)
            else:
                print(msg)

    def _load_model(self) -> None:
        """Carica modello YOLO."""
        if not YOLO_AVAILABLE:
            self.log("[ActionAgent] ⚠️ ultralytics non installato, uso mock")
            return

        try:
            import os

            if os.path.exists(self.config.model_path):
                self.model = YOLO(self.config.model_path)
                self.log(f"[ActionAgent] ✅ Modello caricato: {self.config.model_path}")
            else:
                self.log(f"[ActionAgent] ⚠️ Modello non trovato: {self.config.model_path}")
        except Exception as e:  # pragma: no cover - errore runtime modello
            self.log(f"[ActionAgent] ❌ Errore caricamento: {e}")

    def detect_frame(self, frame: np.ndarray, time: float) -> List[ActionDetection]:
        """
        Rileva azioni in un singolo frame.

        Args:
            frame: Frame BGR
            time: Timestamp in secondi

        Returns:
            Lista di ActionDetection
        """
        if self.model is None:
            return []

        detections: List[ActionDetection] = []

        try:
            results = self.model(frame, verbose=False, conf=self.config.confidence_threshold)

            for r in results:
                if getattr(r, "boxes", None) is None:
                    continue

                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Map class to ActionType
                    action_type = self.config.class_mapping.get(cls_id, ActionType.UNKNOWN)

                    # Determina side basandosi sulla posizione x
                    center_x = (x1 + x2) / 2.0 / frame.shape[1]  # Normalizzato 0-1
                    if center_x < self.config.left_zone_x:
                        side = "left"
                    elif center_x > self.config.right_zone_x:
                        side = "right"
                    else:
                        side = "center"

                    detections.append(
                        ActionDetection(
                            action_type=action_type,
                            time=time,
                            confidence=conf,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            side=side,
                        )
                    )

        except Exception as e:  # pragma: no cover - errore runtime modello
            self.log(f"[ActionAgent] Errore detection: {e}")

        return detections

    def run(
        self,
        frames: List[np.ndarray],
        fps: float = 5.0,
        start_time: float = 0.0,
    ) -> List[ActionDetection]:
        """
        Analizza tutti i frame e rileva azioni.

        Args:
            frames: Lista frame BGR
            fps: Frame rate
            start_time: Tempo iniziale in secondi

        Returns:
            Lista di ActionDetection aggregate
        """
        if self.model is None:
            self.log("[ActionAgent] ⚠️ Modello non disponibile, skip")
            return []

        self.log(f"[ActionAgent] Analisi {len(frames)} frame...")

        all_detections: List[ActionDetection] = []

        for i, frame in enumerate(frames):
            time = start_time + i / fps
            detections = self.detect_frame(frame, time)
            all_detections.extend(detections)

        # Aggrega detections temporalmente (rimuovi duplicati vicini)
        aggregated = self._aggregate_detections(all_detections)

        self.log(f"[ActionAgent] 📊 {len(aggregated)} azioni rilevate")

        # Conta per tipo
        counts: Dict[str, int] = {}
        for d in aggregated:
            counts[d.action_type.value] = counts.get(d.action_type.value, 0) + 1

        for action, count in sorted(counts.items()):
            self.log(f"  - {action}: {count}")

        return aggregated

    def _aggregate_detections(
        self,
        detections: List[ActionDetection],
        time_window: float = 0.5,
    ) -> List[ActionDetection]:
        """
        Aggrega detections vicine nel tempo (stesso tipo + stessa zona).
        """
        if not detections:
            return []

        # Ordina per tempo
        detections.sort(key=lambda d: d.time)

        aggregated: List[ActionDetection] = []
        current_group: List[ActionDetection] = [detections[0]]

        for det in detections[1:]:
            last = current_group[-1]

            # Se stesso tipo, stesso side, e vicini nel tempo -> raggruppa
            if (
                det.action_type == last.action_type
                and det.side == last.side
                and det.time - last.time < time_window
            ):
                current_group.append(det)
            else:
                # Chiudi gruppo precedente
                best = max(current_group, key=lambda d: d.confidence)
                aggregated.append(best)
                current_group = [det]

        # Ultimo gruppo
        if current_group:
            best = max(current_group, key=lambda d: d.confidence)
            aggregated.append(best)

        return aggregated

    def get_actions_in_rally(
        self,
        all_actions: List[ActionDetection],
        rally_start: float,
        rally_end: float,
    ) -> List[ActionDetection]:
        """Filtra azioni che cadono dentro un rally."""
        return [a for a in all_actions if rally_start <= a.time <= rally_end]

    def get_rally_stats(
        self,
        actions: List[ActionDetection],
    ) -> Dict[str, Any]:
        """Calcola statistiche azioni per un rally."""
        stats: Dict[str, Any] = {
            "total_actions": len(actions),
            "by_type": {},
            "by_side": {"left": 0, "right": 0, "center": 0},
            "sequence": [],
        }

        for a in actions:
            # Count by type
            atype = a.action_type.value
            stats["by_type"][atype] = stats["by_type"].get(atype, 0) + 1

            # Count by side
            stats["by_side"][a.side] = stats["by_side"].get(a.side, 0) + 1

            # Sequence
            stats["sequence"].append(
                {
                    "time": a.time,
                    "action": atype,
                    "side": a.side,
                }
            )

        return stats


def analyze_touch_sequence(actions: List[ActionDetection]) -> Dict[str, Any]:
    """
    Analizza la sequenza di tocchi in un rally.

    Pattern tipici:
    - receive -> set -> spike (attacco classico)
    - receive -> set -> spike -> block (attacco murato)
    - serve -> receive (ace o errore ricezione)
    """
    if not actions:
        return {"pattern": "empty", "touches": 0}

    sequence = [a.action_type.value for a in sorted(actions, key=lambda x: x.time)]

    # Identifica pattern
    pattern = "unknown"
    if sequence == ["serve"]:
        pattern = "ace_or_error"
    elif "serve" in sequence and "receive" in sequence:
        if "spike" in sequence:
            if "block" in sequence:
                pattern = "blocked_attack"
            else:
                pattern = "successful_attack"
        elif "set" in sequence:
            pattern = "setup_only"
        else:
            pattern = "receive_only"

    return {
        "pattern": pattern,
        "touches": len(sequence),
        "sequence": sequence,
    }


__all__ = [
    "ActionAgent",
    "ActionAgentConfig",
    "ActionDetection",
    "ActionType",
    "analyze_touch_sequence",
]


