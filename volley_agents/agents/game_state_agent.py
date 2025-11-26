"""GameStateAgent - Classificazione stato partita con VideoMAE."""

from dataclasses import dataclass
from typing import List, Optional, Callable
from enum import Enum

import numpy as np
import cv2

try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    import torch

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from volley_agents.core.timeline import Timeline
from volley_agents.core.event import Event, EventType
from volley_agents.agents.motion_agent import FrameSample


class GameState(Enum):
    """Stati possibili della partita."""

    NO_PLAY = "no-play"
    PLAY = "play"
    SERVICE = "service"
    UNKNOWN = "unknown"


@dataclass
class GameStateAgentConfig:
    """Configurazione GameStateAgent."""

    model_path: str = "volley_agents/models/game_state"
    num_frames: int = 16
    resize_to: int = 224
    window_seconds: float = 2.0  # Finestra di analisi
    stride_seconds: float = 1.0  # Passo tra finestre
    min_confidence: float = 0.7
    enable_logging: bool = False
    log_callback: Optional[Callable[[str], None]] = None


@dataclass
class GameStateResult:
    """Risultato classificazione."""

    time: float
    state: GameState
    confidence: float


class GameStateAgent:
    """
    Agent per classificare lo stato della partita usando VideoMAE.

    Classifica segmenti video in:
    - NO_PLAY: pausa, timeout, celebrazione
    - PLAY: rally in corso
    - SERVICE: momento della battuta
    """

    def __init__(self, config: Optional[GameStateAgentConfig] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers e torch sono richiesti. Installa con: pip install transformers torch"
            )

        self.config = config or GameStateAgentConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

        self._stats = {
            "segments_analyzed": 0,
            "states": {"play": 0, "no-play": 0, "service": 0},
        }

    def _log(self, msg: str):
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(msg)
            else:
                print(msg)

    def _load_model(self):
        """Carica modello VideoMAE."""

        self._log(f"[GameStateAgent] Caricamento modello da {self.config.model_path}...")

        self.processor = VideoMAEImageProcessor.from_pretrained(self.config.model_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            self.config.model_path
        )
        self.model.to(self.device)
        self.model.eval()

        self._log(f"[GameStateAgent] ✅ Modello caricato su {self.device}")

    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocessa frame: BGR->RGB, resize."""

        processed: List[np.ndarray] = []
        for frame in frames:
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (self.config.resize_to, self.config.resize_to))
                processed.append(resized)
        return processed

    def _select_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Seleziona esattamente num_frames uniformemente, con padding se necessario."""

        target = self.config.num_frames  # 16

        if len(frames) == 0:
            return []

        if len(frames) >= target:
            # Subsample uniformemente
            step = len(frames) / target
            return [frames[int(i * step)] for i in range(target)]

        # Pad ripetendo l'ultimo frame
        selected = frames.copy()
        while len(selected) < target:
            selected.append(frames[-1])
        return selected

    def classify_segment(self, frames: List[np.ndarray]) -> GameStateResult:
        """
        Classifica un segmento di frame.

        Args:
            frames: Lista di frame BGR

        Returns:
            GameStateResult con stato e confidence
        """

        if not frames:
            return GameStateResult(time=0, state=GameState.UNKNOWN, confidence=0.0)

        # Preprocess
        processed = self._preprocess_frames(frames)
        selected = self._select_frames(processed)

        if len(selected) < 8:  # Minimo frame necessari
            return GameStateResult(time=0, state=GameState.UNKNOWN, confidence=0.0)

        # Inference
        inputs = self.processor(selected, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()

        # Map to GameState
        label = self.model.config.id2label[pred_id]
        state_map = {
            "no-play": GameState.NO_PLAY,
            "play": GameState.PLAY,
            "service": GameState.SERVICE,
        }
        state = state_map.get(label.lower(), GameState.UNKNOWN)

        return GameStateResult(time=0, state=state, confidence=confidence)

    def run(
        self,
        frames: List[FrameSample],
        timeline: Optional[Timeline] = None,
    ) -> List[Event]:
        """
        Analizza tutti i frame e genera eventi GAME_STATE.

        Args:
            frames: Lista di FrameSample dal video
            timeline: Timeline opzionale per aggiungere eventi

        Returns:
            Lista di eventi con stato partita
        """

        if not frames:
            return []

        events: List[Event] = []
        fps = 1.0 / (frames[1].time - frames[0].time) if len(frames) > 1 else 5.0

        window_frames = int(self.config.window_seconds * fps)
        stride_frames = int(self.config.stride_seconds * fps)

        self._log(
            f"[GameStateAgent] Analisi {len(frames)} frame (window={window_frames}, stride={stride_frames})"
        )

        i = 0
        while i < len(frames):
            # Estrai finestra
            window_end = min(i + window_frames, len(frames))
            window = [f.frame for f in frames[i:window_end]]
            window_time = frames[i].time

            # Classifica
            result = self.classify_segment(window)
            result.time = window_time

            self._stats["segments_analyzed"] += 1
            if result.state != GameState.UNKNOWN:
                self._stats["states"][result.state.value] += 1

            # Crea evento se confidence sufficiente
            if result.confidence >= self.config.min_confidence:
                event = Event(
                    type=EventType.GAME_STATE,
                    time=window_time,
                    confidence=result.confidence,
                    extra={"state": result.state.value, "model": "VideoMAE"},
                )
                events.append(event)

                if result.state == GameState.SERVICE:
                    self._log(
                        f"  🎯 SERVICE @ {window_time:.1f}s (conf={result.confidence:.0%})"
                    )
                elif result.state == GameState.PLAY:
                    self._log(
                        f"  🏐 PLAY @ {window_time:.1f}s (conf={result.confidence:.0%})"
                    )

            i += stride_frames

        self._log_summary()

        if timeline is not None:
            timeline.extend(events)

        return events

    def _log_summary(self):
        """Log statistiche finali."""

        stats = self._stats
        total = stats["segments_analyzed"]
        if total == 0:
            return

        self._log("[GameStateAgent] 📊 Summary:")
        self._log(f"  Segmenti analizzati: {total}")
        for state, count in stats["states"].items():
            pct = count / total * 100
            self._log(f"    {state}: {count} ({pct:.1f}%)")


__all__ = ["GameStateAgent", "GameStateAgentConfig", "GameState", "GameStateResult"]


