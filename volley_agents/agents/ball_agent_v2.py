"""
BallAgentV2: tracking palla con modelli ONNX ottimizzati (VballNet).

Usa modelli pre-trainati specifici per pallavolo con detection rate ~84%.
Mantiene compatibilità con l'interfaccia di BallAgent originale.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Sequence
from pathlib import Path
from collections import deque

import numpy as np

from volley_agents.core.event import Event, EventType
from volley_agents.agents.motion_agent import FrameSample

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class BallAgentV2Config:
    """Configurazione per BallAgentV2."""
    
    # Path al modello ONNX
    model_path: str = "volley_agents/models/ball_tracking/VballNetV1b_seq9_grayscale_best.onnx"
    
    # Dimensioni input modello
    input_width: int = 512
    input_height: int = 288
    
    # Soglia detection heatmap
    detection_threshold: float = 0.5
    
    # Numero frame per batch (seq9 = 9 frame)
    sequence_length: int = 9
    
    # Zone detection (normalizzate 0-1)
    service_left_threshold: float = 0.15
    service_right_threshold: float = 0.85
    net_left_threshold: float = 0.45
    net_right_threshold: float = 0.55
    ground_y_threshold: float = 0.9
    
    # Tracking
    max_speed: float = 200.0  # pixel/frame
    track_length: int = 8
    
    # Logging
    enable_logging: bool = False
    log_callback: Optional[Callable[[str], None]] = None


class BallAgentV2:
    """
    Agente per tracking palla usando modelli ONNX ottimizzati.
    
    Usa VballNet pre-trainato specificamente per pallavolo.
    Detection rate ~84% vs ~30-40% di YOLO generico.
    """
    
    def __init__(self, config: Optional[BallAgentV2Config] = None):
        self.config = config or BallAgentV2Config()
        self._session = None
        self._out_dim = 9
        
        # Stats
        self._stats = {
            "total_frames": 0,
            "detected_frames": 0,
            "events": {"detected": 0, "ground": 0, "cross_net": 0, "serve": 0},
            "zones": {}
        }
        
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) non disponibile")
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime non disponibile: pip install onnxruntime")
        
        self._load_model()
    
    def _log(self, message: str):
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(message)
            else:
                print(message)
    
    def _load_model(self):
        """Carica il modello ONNX."""
        model_path = Path(self.config.model_path)
        
        if not model_path.exists():
            # Prova path relativo alla root del progetto
            alt_paths = [
                Path("volley_agents/models/ball_tracking") / model_path.name,
                Path(__file__).parent.parent / "models" / "ball_tracking" / model_path.name,
            ]
            for alt in alt_paths:
                if alt.exists():
                    model_path = alt
                    break
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modello ONNX non trovato: {self.config.model_path}")
        
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Determina output dimension dal nome del modello
        if "seq9" in str(model_path):
            self._out_dim = 9
        else:
            self._out_dim = 3
        
        self._log(f"[BallAgentV2] ✅ Modello caricato: {model_path.name}")
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Preprocessa i frame per il modello."""
        processed = []
        cfg = self.config
        
        for frame in frames:
            # Converti in grayscale se necessario
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Ridimensiona
            resized = cv2.resize(gray, (cfg.input_width, cfg.input_height))
            
            # Normalizza
            normalized = resized.astype(np.float32) / 255.0
            processed.append(normalized)
        
        return processed
    
    def _postprocess_output(self, output: np.ndarray) -> List[Tuple[int, int, int]]:
        """Estrae posizioni palla dalle heatmap."""
        results = []
        cfg = self.config
        
        for frame_idx in range(self._out_dim):
            heatmap = output[0, frame_idx, :, :]
            
            # Threshold
            _, binary = cv2.threshold(
                heatmap, cfg.detection_threshold, 1.0, cv2.THRESH_BINARY
            )
            
            # Trova contorni
            contours, _ = cv2.findContours(
                (binary * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Prendi il contorno più grande
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    results.append((1, cx, cy))
                else:
                    results.append((0, 0, 0))
            else:
                results.append((0, 0, 0))
        
        return results
    
    def get_zone(self, x: float, y: float, w: int, h: int) -> str:
        """Determina la zona della palla."""
        cfg = self.config
        x_norm = x / w
        y_norm = y / h
        
        if y_norm > cfg.ground_y_threshold:
            return "ground"
        if x_norm < cfg.service_left_threshold:
            return "service_left"
        if x_norm > cfg.service_right_threshold:
            return "service_right"
        if cfg.net_left_threshold < x_norm < cfg.net_right_threshold:
            return "net"
        if x_norm < 0.5:
            return "left"
        return "right"
    
    def run(
        self,
        frames: Sequence[FrameSample],
        timeline=None,
    ) -> List[Event]:
        """
        Analizza i frame per tracking palla.
        
        Args:
            frames: Sequenza di FrameSample
            timeline: Timeline opzionale per aggiungere eventi
            
        Returns:
            Lista di eventi palla
        """
        if len(frames) < 2:
            return []
        
        cfg = self.config
        events: List[Event] = []
        
        # Reset stats
        self._stats = {
            "total_frames": len(frames),
            "detected_frames": 0,
            "events": {"detected": 0, "ground": 0, "cross_net": 0, "serve": 0},
            "zones": {}
        }
        
        # Estrai frame numpy
        raw_frames = [f.frame for f in frames]
        timestamps = [f.time for f in frames]
        
        # Dimensioni originali
        h_orig, w_orig = raw_frames[0].shape[:2]
        
        # Buffer per sequenza
        frame_buffer = deque(maxlen=cfg.sequence_length)
        
        # Riempi buffer iniziale
        initial_processed = self._preprocess_frames(raw_frames[:cfg.sequence_length])
        for pf in initial_processed:
            frame_buffer.append(pf)
        
        # Pad se necessario
        while len(frame_buffer) < cfg.sequence_length:
            frame_buffer.appendleft(frame_buffer[0])
        
        # Posizioni per analisi traiettoria
        positions = []
        prev_side = None
        
        # Processa in batch
        batch_size = cfg.sequence_length
        
        for batch_start in range(0, len(raw_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(raw_frames))
            batch_frames = raw_frames[batch_start:batch_end]
            batch_times = timestamps[batch_start:batch_end]
            
            # Preprocessa batch
            processed = self._preprocess_frames(batch_frames)
            
            # Aggiorna buffer
            for pf in processed:
                frame_buffer.append(pf)
            
            # Prepara input tensor
            input_tensor = np.stack(list(frame_buffer), axis=2)  # (H, W, 9)
            input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, H, W, 9)
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # (1, 9, H, W)
            
            # Inferenza
            inputs = {self._session.get_inputs()[0].name: input_tensor}
            output = self._session.run(None, inputs)[0]
            
            # Postprocess
            predictions = self._postprocess_output(output)
            
            # Genera eventi per ogni frame del batch
            for i, (visibility, x, y) in enumerate(predictions[:len(batch_frames)]):
                if batch_start + i >= len(timestamps):
                    break
                
                t = batch_times[i]
                
                if visibility:
                    # Scala alle coordinate originali
                    x_orig = x * w_orig / cfg.input_width
                    y_orig = y * h_orig / cfg.input_height
                    
                    zone = self.get_zone(x_orig, y_orig, w_orig, h_orig)
                    
                    self._stats["detected_frames"] += 1
                    self._stats["zones"][zone] = self._stats["zones"].get(zone, 0) + 1
                    self._stats["events"]["detected"] += 1
                    
                    # Evento BALL_DETECTED
                    events.append(Event(
                        time=t,
                        type=EventType.BALL_DETECTED,
                        confidence=0.85,  # Confidence fissa basata su detection rate modello
                        extra={
                            "x": x_orig,
                            "y": y_orig,
                            "zone": zone,
                            "model": "VballNet_ONNX"
                        }
                    ))
                    
                    # Track per analisi
                    current_side = "left" if x_orig / w_orig < 0.5 else "right"
                    positions.append((t, x_orig, y_orig, zone, current_side))
                    
                    # BALL_CROSS_NET
                    if prev_side is not None and prev_side != current_side:
                        x_norm = x_orig / w_orig
                        if cfg.net_left_threshold <= x_norm <= cfg.net_right_threshold:
                            self._stats["events"]["cross_net"] += 1
                            events.append(Event(
                                time=t,
                                type=EventType.BALL_CROSS_NET,
                                confidence=0.8,
                                extra={
                                    "x": x_orig,
                                    "y": y_orig,
                                    "direction": f"{prev_side}->{current_side}"
                                }
                            ))
                    
                    # BALL_TOUCH_GROUND
                    if zone == "ground":
                        self._stats["events"]["ground"] += 1
                        events.append(Event(
                            time=t,
                            type=EventType.BALL_TOUCH_GROUND,
                            confidence=0.7,
                            extra={"x": x_orig, "y": y_orig, "zone": zone}
                        ))
                    
                    prev_side = current_side
        
        # Log summary
        if cfg.enable_logging:
            self._log_summary()
        
        if timeline is not None:
            timeline.extend(events)
        
        return events
    
    def _log_summary(self):
        """Log statistiche finali."""
        stats = self._stats
        total = stats["total_frames"]
        detected = stats["detected_frames"]
        rate = (detected / total * 100) if total > 0 else 0
        
        self._log(f"[BallAgentV2] 📊 Summary:")
        self._log(f"  Detection rate: {rate:.1f}% ({detected}/{total} frame)")
        self._log(f"  Eventi generati: {sum(stats['events'].values())}")
        for evt_type, count in stats['events'].items():
            if count > 0:
                self._log(f"    - {evt_type}: {count}")


__all__ = ["BallAgentV2", "BallAgentV2Config"]