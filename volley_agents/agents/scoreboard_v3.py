"""
ScoreboardAgentV3 - Lettura tabellone LED con OCR reale + preprocessing adattivo.

Migliorie rispetto a V2:
- EasyOCR invece di riconoscimento a 7 segmenti
- Preprocessing specifico per LED (canale colore, morphology)
- Detector di cambio frame come fallback rapido
- Template matching opzionale per cifre difficili
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Dict
from collections import deque
from pathlib import Path

import numpy as np
import cv2

from volley_agents.core.event import Event, EventType

_easyocr_reader = None


def get_easyocr_reader():
    """Lazy loading del reader EasyOCR (evita import pesanti all'avvio)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr

            _easyocr_reader = easyocr.Reader(
                ["en"],
                gpu=False,
                verbose=False,
            )
        except ImportError:
            raise ImportError(
                "EasyOCR non installato. Installa con: pip install easyocr"
            )
    return _easyocr_reader


@dataclass
class ScoreboardConfigV3:
    """Configurazione per ScoreboardAgentV3."""

    x: int = 0
    y: int = 0
    w: int = 100
    h: int = 50

    led_color: str = "red"  # "red", "green", "yellow", "auto"
    threshold_low: int = 150
    threshold_high: int = 255
    use_adaptive_threshold: bool = False
    morph_kernel_size: int = 3

    ocr_confidence_min: float = 0.3
    use_template_matching: bool = True
    template_dir: Optional[str] = None

    history_size: int = 10
    min_stable_frames: int = 4
    change_detection_threshold: float = 0.15

    save_debug_images: bool = False
    debug_dir: str = "debug_scoreboard"


@dataclass
class ScoreReading:
    """Singola lettura del punteggio."""

    home: Optional[int] = None
    away: Optional[int] = None
    confidence: float = 0.0
    method: str = "unknown"
    timestamp: float = 0.0


class ScoreboardAgentV3:
    """
    Agente per lettura tabellone LED con OCR reale.

    Pipeline:
    1. Estrai ROI dal frame
    2. Preprocessing LED (canale colore, morphology)
    3. OCR con EasyOCR
    4. Fallback a template matching se necessario
    5. Detector di cambio frame come backup
    6. Stabilizzazione temporale e emissione SCORE_CHANGE
    """

    def __init__(
        self,
        cfg: ScoreboardConfigV3,
        enable_logging: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg
        self.enable_logging = enable_logging
        self.log_callback = log_callback

        self._history: deque = deque(maxlen=cfg.history_size)
        self._last_emitted: Optional[Tuple[int, int]] = None
        self._frame_count: int = 0

        self._templates: Dict[int, np.ndarray] = {}
        self._templates_loaded: bool = False

        self._last_roi_gray: Optional[np.ndarray] = None
        self._change_detected_at: Optional[float] = None

        self._debug_count: int = 0
        self._warned_roi_unset: bool = False
        self._warned_roi_default: bool = False
        if cfg.save_debug_images:
            Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def process_frame(self, frame_bgr: np.ndarray, t: float) -> List[Event]:
        """Processa un frame e ritorna eventi SCORE_CHANGE se rilevati."""
        self._frame_count += 1

        roi = self._extract_roi(frame_bgr)
        if roi is None or roi.size == 0:
            return []

        processed = self._preprocess_led(roi)
        frame_changed = self._detect_frame_change(processed, t)

        reading = self._read_score_ocr(processed, t)

        if (reading.home is None or reading.away is None) and self.cfg.use_template_matching:
            reading = self._read_score_template(processed, t)

        if reading.home is None and frame_changed:
            reading = ScoreReading(
                home=None,
                away=None,
                confidence=0.5,
                method="change_detection",
                timestamp=t,
            )
            self._log(f"Frame {self._frame_count}: cambio rilevato ma cifre non lette")

        if reading.home is not None and reading.away is not None:
            return self._update_history(reading)

        return []

    # --------------------------------------------------------------------- #
    # ROI extraction
    # --------------------------------------------------------------------- #

    def _extract_roi(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Estrae la ROI del tabellone dal frame."""
        cfg = self.cfg
        h_frame, w_frame = frame_bgr.shape[:2]

        if (cfg.w <= 1 or cfg.h <= 1) and not self._warned_roi_unset:
            self._log(
                "ROI tabellone non configurata (w/h troppo piccoli). "
                "Usa tools/scoreboard/calibrate_scoreboard_v3.py per calibrare x,y,w,h."
            )
            self._warned_roi_unset = True
            return None

        if (
            cfg.x == 0
            and cfg.y == 0
            and cfg.w == 100
            and cfg.h == 50
            and not self._warned_roi_default
        ):
            self._log(
                "ROI tabellone sta usando i valori di default (0,0,100,50). "
                "Configura i valori reali del tabellone per abilitare l'OCR."
            )
            self._warned_roi_default = True

        x = max(0, min(cfg.x, w_frame - 1))
        y = max(0, min(cfg.y, h_frame - 1))
        w = max(1, min(cfg.w, w_frame - x))
        h = max(1, min(cfg.h, h_frame - y))

        roi = frame_bgr[y : y + h, x : x + w]
        return roi if roi.size > 0 else None

    # --------------------------------------------------------------------- #
    # LED preprocessing
    # --------------------------------------------------------------------- #

    def _preprocess_led(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocessing migliorato per display LED (bias rosso di default).

        1. Isola canale dominante con sottrazione degli altri canali
        2. Riduce flicker/rumore con blur leggero
        3. Threshold (adaptive opzionale, Otsu come default)
        4. Morfologia aggressiva
        5. Rimozione componenti connesse piccole
        """

        cfg = self.cfg
        b = roi_bgr[:, :, 0].astype(np.float32)
        g = roi_bgr[:, :, 1].astype(np.float32)
        r = roi_bgr[:, :, 2].astype(np.float32)

        if cfg.led_color in ("red", "auto"):
            channel = r - np.maximum(g, b) * 0.7
            channel = np.clip(channel, 0, 255).astype(np.uint8)
        elif cfg.led_color == "green":
            channel = g - np.maximum(r, b) * 0.7
            channel = np.clip(channel, 0, 255).astype(np.uint8)
        elif cfg.led_color == "yellow":
            channel = (r + g) / 2 - b * 0.5
            channel = np.clip(channel, 0, 255).astype(np.uint8)
        else:
            channel = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        channel = cv2.GaussianBlur(channel, (3, 3), 0)

        if cfg.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                channel,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=21,
                C=-10,
            )
        else:
            _, binary = cv2.threshold(
                channel,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (cfg.morph_kernel_size, cfg.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = self._remove_small_components(binary, min_area=12)

        if cfg.save_debug_images and self._debug_count < 10:
            cv2.imwrite(f"{cfg.debug_dir}/roi_{self._debug_count}.png", roi_bgr)
            cv2.imwrite(f"{cfg.debug_dir}/channel_{self._debug_count}.png", channel)
            cv2.imwrite(f"{cfg.debug_dir}/binary_{self._debug_count}.png", binary)
            self._debug_count += 1

        return binary

    def _select_color_channel(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Seleziona il canale colore dominante del LED."""
        cfg = self.cfg

        if cfg.led_color == "auto":
            return self._detect_dominant_channel(roi_bgr)

        if cfg.led_color == "red":
            r = roi_bgr[:, :, 2].astype(np.float32)
            b = roi_bgr[:, :, 0].astype(np.float32)
            return np.clip(r - b * 0.5, 0, 255).astype(np.uint8)

        if cfg.led_color == "green":
            return roi_bgr[:, :, 1]

        if cfg.led_color == "yellow":
            r = roi_bgr[:, :, 2].astype(np.float32)
            g = roi_bgr[:, :, 1].astype(np.float32)
            return np.clip((r + g) / 2, 0, 255).astype(np.uint8)

        return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    def _detect_dominant_channel(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Rileva automaticamente il canale colore dominante del LED."""
        b_mean = np.mean(roi_bgr[:, :, 0])
        g_mean = np.mean(roi_bgr[:, :, 1])
        r_mean = np.mean(roi_bgr[:, :, 2])

        if r_mean > g_mean and r_mean > b_mean:
            r = roi_bgr[:, :, 2].astype(np.float32)
            b = roi_bgr[:, :, 0].astype(np.float32)
            return np.clip(r - b * 0.5, 0, 255).astype(np.uint8)
        if g_mean > r_mean and g_mean > b_mean:
            return roi_bgr[:, :, 1]
        return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    def _remove_small_components(self, binary: np.ndarray, min_area: int = 12) -> np.ndarray:
        """Rimuove componenti connesse troppo piccole per limitare il rumore."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        result = np.zeros_like(binary)
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                result[labels == idx] = 255
        return result

    # --------------------------------------------------------------------- #
    # Cambio frame detection
    # --------------------------------------------------------------------- #

    def _detect_frame_change(self, binary: np.ndarray, t: float) -> bool:
        """Rileva cambi significativi nella ROI."""
        if self._last_roi_gray is None:
            self._last_roi_gray = binary.copy()
            return False

        diff = cv2.absdiff(binary, self._last_roi_gray)
        change_ratio = np.sum(diff > 128) / diff.size
        self._last_roi_gray = binary.copy()

        if change_ratio > self.cfg.change_detection_threshold:
            self._change_detected_at = t
            return True

        return False

    # --------------------------------------------------------------------- #
    # OCR con EasyOCR
    # --------------------------------------------------------------------- #

    def _read_score_ocr(self, binary: np.ndarray, t: float) -> ScoreReading:
        """Legge il punteggio usando EasyOCR."""
        try:
            reader = get_easyocr_reader()
        except ImportError as exc:
            self._log(f"EasyOCR non disponibile: {exc}")
            return ScoreReading(timestamp=t)

        scale = 3
        upscaled = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        inverted = cv2.bitwise_not(upscaled)

        results = reader.readtext(
            inverted,
            allowlist="0123456789",
            paragraph=False,
            min_size=10,
            text_threshold=0.5,
            low_text=0.3,
        )

        if not results:
            return ScoreReading(timestamp=t, method="ocr")

        digits_with_pos = []
        for bbox, text, conf in results:
            if conf < self.cfg.ocr_confidence_min:
                continue
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            for digit in text:
                if digit.isdigit():
                    digits_with_pos.append((x_center, int(digit), conf))

        if len(digits_with_pos) < 2:
            return ScoreReading(timestamp=t, method="ocr")

        digits_with_pos.sort(key=lambda x: x[0])
        mid_x = upscaled.shape[1] / 2
        home_digits = [d for d in digits_with_pos if d[0] < mid_x]
        away_digits = [d for d in digits_with_pos if d[0] >= mid_x]

        home_score = self._digits_to_score(home_digits)
        away_score = self._digits_to_score(away_digits)

        avg_conf = float(np.mean([d[2] for d in digits_with_pos])) if digits_with_pos else 0.0

        if self._frame_count % 30 == 0:
            self._log(f"OCR: home={home_score}, away={away_score}, conf={avg_conf:.2f}")

        return ScoreReading(
            home=home_score,
            away=away_score,
            confidence=avg_conf,
            method="ocr",
            timestamp=t,
        )

    def _digits_to_score(self, digits: List[Tuple[float, int, float]]) -> Optional[int]:
        if not digits:
            return None

        digit_values = [d[1] for d in digits]
        if len(digit_values) == 1:
            return digit_values[0]
        if len(digit_values) >= 2:
            return digit_values[0] * 10 + digit_values[1]
        return None

    # --------------------------------------------------------------------- #
    # Template matching fallback
    # --------------------------------------------------------------------- #

    def _load_templates(self):
        if self._templates_loaded:
            return

        cfg = self.cfg
        if not cfg.template_dir:
            self._templates_loaded = True
            return

        template_path = Path(cfg.template_dir)
        if not template_path.exists():
            self._log(f"Directory template non trovata: {template_path}")
            self._templates_loaded = True
            return

        for digit in range(10):
            for ext in (".png", ".jpg", ".bmp"):
                file_path = template_path / f"{digit}{ext}"
                if not file_path.exists():
                    continue
                template = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self._templates[digit] = template
                    break

        self._log(f"Caricati {len(self._templates)} template")
        self._templates_loaded = True

    def _read_score_template(self, binary: np.ndarray, t: float) -> ScoreReading:
        self._load_templates()
        if not self._templates:
            return ScoreReading(timestamp=t, method="template")

        h, w = binary.shape
        mid_x = w // 2
        home_score = self._match_digits(binary[:, :mid_x])
        away_score = self._match_digits(binary[:, mid_x:])

        return ScoreReading(
            home=home_score,
            away=away_score,
            confidence=0.7 if home_score is not None and away_score is not None else 0.0,
            method="template",
            timestamp=t,
        )

    def _match_digits(self, roi: np.ndarray) -> Optional[int]:
        if roi.size == 0 or not self._templates:
            return None

        best_matches = []
        for digit, template in self._templates.items():
            tpl = template
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                scale = min(roi.shape[0] / tpl.shape[0], roi.shape[1] / tpl.shape[1]) * 0.8
                tpl = cv2.resize(tpl, None, fx=scale, fy=scale)
            if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                continue

            result = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.6:
                best_matches.append((max_loc[0], digit, max_val))

        if not best_matches:
            return None

        best_matches.sort(key=lambda x: x[0])
        filtered = []
        min_distance = 10
        for match in best_matches:
            if not filtered or match[0] - filtered[-1][0] > min_distance:
                filtered.append(match)
            elif match[2] > filtered[-1][2]:
                filtered[-1] = match

        if len(filtered) == 1:
            return filtered[0][1]
        if len(filtered) >= 2:
            return filtered[0][1] * 10 + filtered[1][1]
        return None

    # --------------------------------------------------------------------- #
    # Stabilizzazione temporale
    # --------------------------------------------------------------------- #

    def _update_history(self, reading: ScoreReading) -> List[Event]:
        cfg = self.cfg
        score = (reading.home, reading.away)
        self._history.append(score)

        if len(self._history) < cfg.min_stable_frames:
            return []

        from collections import Counter

        counter = Counter(self._history)
        most_common, count = counter.most_common(1)[0]

        if count < cfg.min_stable_frames:
            return []

        if most_common[0] is None or most_common[1] is None:
            return []

        if most_common == self._last_emitted:
            return []

        if self._last_emitted is not None:
            old_home, old_away = self._last_emitted
            new_home, new_away = most_common
            home_diff = new_home - old_home
            away_diff = new_away - old_away

            is_point = (home_diff == 1 and away_diff == 0) or (home_diff == 0 and away_diff == 1)
            is_set_reset = (home_diff < 0 or away_diff < 0)
            if not (is_point or is_set_reset):
                self._log(f"Cambio non coerente ignorato: {self._last_emitted} -> {most_common}")
                return []

        self._last_emitted = most_common
        home, away = most_common
        self._log(
            f"SCORE_CHANGE: {home}-{away} (method={reading.method}, stable={count} frames)"
        )

        return [
            Event(
                time=reading.timestamp,
                type=EventType.SCORE_CHANGE,
                confidence=reading.confidence,
                extra={
                    "home": home,
                    "away": away,
                    "score_left": home,
                    "score_right": away,
                    "method": reading.method,
                },
            )
        ]

    # --------------------------------------------------------------------- #
    # Logging helper
    # --------------------------------------------------------------------- #

    def _log(self, msg: str):
        if not self.enable_logging:
            return
        if self.log_callback:
            self.log_callback(f"[ScoreboardV3] {msg}")
        else:
            print(f"[ScoreboardV3] {msg}")


def create_digit_templates(
    video_path: str,
    roi: Tuple[int, int, int, int],
    output_dir: str,
    timestamps: List[float],
    led_color: str = "red",
) -> None:
    """
    Helper per creare template cifre da frame di esempio.
    """
    import os

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cfg = ScoreboardConfigV3(
        x=roi[0],
        y=roi[1],
        w=roi[2],
        h=roi[3],
        led_color=led_color,
    )
    agent = ScoreboardAgentV3(cfg, enable_logging=True)

    for i, t in enumerate(timestamps):
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        roi_img = agent._extract_roi(frame)
        if roi_img is None:
            continue

        processed = agent._preprocess_led(roi_img)
        cv2.imwrite(f"{output_dir}/sample_{i}_t{t:.0f}.png", processed)
        print(f"Salvato sample_{i}_t{t:.0f}.png")

    cap.release()
    print(
        f"\nOra ritaglia manualmente le cifre 0-9 e salvale come 0.png, 1.png, ecc. in {output_dir}"
    )


__all__ = [
    "ScoreboardAgentV3",
    "ScoreboardConfigV3",
    "create_digit_templates",
]

