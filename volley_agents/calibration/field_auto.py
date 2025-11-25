"""
FieldAutoCalibrator: auto-detection linee campo con Hough Transform.

Output:
- 4 angoli campo in pixel
- Matrice omografia 3x3
- Fattore pixel/metri
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from volley_agents.core.event import Event, EventType


@dataclass
class FieldAutoConfig:
    """Configurazione per auto-calibrazione campo."""

    # Parametri Hough Transform (ottimizzati per volleyball)
    hough_threshold: int = 50  # Abbassato da 100 per maggiore sensibilità
    hough_min_line_length: int = 50  # Abbassato da 100
    hough_max_line_gap: int = 20  # Aumentato da 10 per linee spezzate

    # Parametri Canny edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Fallback detection
    use_color_fallback: bool = True  # Usa detection basata su colore se Hough fallisce

    # Colore linee campo (bianco tipicamente)
    line_color_lower: Tuple[int, int, int] = (200, 200, 200)  # BGR
    line_color_upper: Tuple[int, int, int] = (255, 255, 255)  # BGR

    # Dimensioni campo reali (metri)
    field_length: float = 18.0
    field_width: float = 9.0


class FieldAutoCalibrator:
    """
    Auto-calibratore campo da singolo frame.

    Uso:
        calibrator = FieldAutoCalibrator()
        events = calibrator.calibrate(frame)

        # Accesso diretto ai dati
        corners = calibrator.get_corners()  # 4 punti in pixel
        H = calibrator.get_homography()     # matrice 3x3
        ppm = calibrator.get_pixels_per_meter()
    """

    def __init__(self, config: Optional[FieldAutoConfig] = None):
        self.config = config or FieldAutoConfig()
        self._corners: Optional[np.ndarray] = None
        self._homography: Optional[np.ndarray] = None
        self._ppm: Optional[float] = None

    def _preprocess_for_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocessing migliorato per rilevare linee bianche su campo colorato.

        Args:
            frame: Frame BGR da OpenCV

        Returns:
            Immagine edges binaria
        """
        if cv2 is None:
            return None

        # Converti in HSV per isolare il bianco
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Maschera per linee bianche (alta luminosità, bassa saturazione)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Applica morphology per pulire
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # Edge detection sulla maschera
        cfg = self.config
        edges = cv2.Canny(white_mask, cfg.canny_low, cfg.canny_high)

        return edges

    def _detect_court_lines(self, edges: np.ndarray) -> Tuple[List, List]:
        """
        Rileva linee del campo con parametri ottimizzati per volleyball.

        Args:
            edges: Immagine edges binaria

        Returns:
            (h_lines, v_lines): Liste di linee orizzontali e verticali
        """
        if cv2 is None:
            return [], []

        cfg = self.config

        # Parametri più permissivi
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )

        if lines is None:
            return [], []

        # Filtra linee per angolo (orizzontali e verticali del campo)
        h_lines = []
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Accetta linee quasi orizzontali (0-15° o 165-180°) o quasi verticali (75-105°)
            if angle < 15 or angle > 165:
                h_lines.append(line[0])
            elif 75 < angle < 105:
                v_lines.append(line[0])

        return h_lines, v_lines

    def _detect_court_by_color(self, frame: np.ndarray) -> Optional[dict]:
        """
        Rileva il campo basandosi sul colore (arancione/blu tipico volleyball).
        Fallback quando Hough Transform fallisce.

        Args:
            frame: Frame BGR da OpenCV

        Returns:
            Dict con informazioni campo o None se non trovato
        """
        if cv2 is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]

        # Range per arancione (campo indoor tipico)
        lower_orange = np.array([8, 100, 100])
        upper_orange = np.array([25, 255, 255])

        # Range per blu (altro lato campo)
        lower_blue = np.array([100, 80, 80])
        upper_blue = np.array([130, 255, 255])

        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_court = cv2.bitwise_or(mask_orange, mask_blue)

        # Pulisci la maschera con morphologia
        kernel = np.ones((10, 10), np.uint8)
        mask_court = cv2.morphologyEx(mask_court, cv2.MORPH_CLOSE, kernel)
        mask_court = cv2.morphologyEx(mask_court, cv2.MORPH_OPEN, kernel)

        # Trova contorni
        contours, _ = cv2.findContours(mask_court, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Prendi il contorno più grande (il campo)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Il campo deve occupare almeno il 30% del frame
        if area < (h * w * 0.3):
            return None

        x, y, cw, ch = cv2.boundingRect(largest)

        # La rete è circa al centro orizzontale
        net_x = x + cw // 2

        print(
            f"✅ Calibrazione colore: campo trovato @ ({x}, {y}, {cw}x{ch}), net_x={net_x}"
        )

        return {
            "court_bounds": (x, y, x + cw, y + ch),
            "net_x": net_x,
            "left_zone": {"x1": x, "y1": y, "x2": net_x, "y2": y + ch},
            "right_zone": {"x1": net_x, "y1": y, "x2": x + cw, "y2": y + ch},
            "service_left": {"x1": x, "y1": y, "x2": x + cw // 6, "y2": y + ch},
            "service_right": {"x1": x + cw - cw // 6, "y1": y, "x2": x + cw, "y2": y + ch},
            "calibration_method": "color_detection",
            "confidence": 0.7,
        }

    def calibrate(self, frame: np.ndarray) -> List[Event]:
        """
        Calibra il campo da un singolo frame.

        Args:
            frame: Frame BGR da OpenCV

        Returns:
            Lista di eventi di calibrazione
        """
        if cv2 is None:
            return []

        events = []
        cfg = self.config

        # 1. Preprocessing migliorato per linee
        edges = self._preprocess_for_lines(frame)
        if edges is None:
            # Fallback a color detection
            if cfg.use_color_fallback:
                print("⚠️ Hough fallito (edges=None), provo color detection...")
                result = self._detect_court_by_color(frame)
                if result:
                    events.append(
                        Event(
                            time=0.0,
                            type=EventType.FIELD_LINES_DETECTED,
                            confidence=result.get("confidence", 0.6),
                            extra={
                                "h_lines": 0,
                                "v_lines": 0,
                                "calibration_method": result.get("calibration_method", "color_fallback"),
                                "net_x": result["net_x"],
                                "court_bounds": result["court_bounds"],
                            },
                        )
                    )
                    return events
                else:
                    # Ultimo fallback: preset
                    print("⚠️ Calibrazione campo fallita, uso preset")
            return events

        # 2. Hough Transform per linee
        h_lines, v_lines = self._detect_court_lines(edges)

        # 3. Se non abbastanza linee, prova fallback
        if len(h_lines) < 2 or len(v_lines) < 2:
            if cfg.use_color_fallback:
                print(
                    f"⚠️ Hough fallito (linee insufficienti: {len(h_lines)}H/{len(v_lines)}V), "
                    "provo color detection..."
                )
                result = self._detect_court_by_color(frame)
                if result:
                    events.append(
                        Event(
                            time=0.0,
                            type=EventType.FIELD_LINES_DETECTED,
                            confidence=result.get("confidence", 0.6),
                            extra={
                                "h_lines": len(h_lines),
                                "v_lines": len(v_lines),
                                "calibration_method": result.get("calibration_method", "color_fallback"),
                                "net_x": result["net_x"],
                                "court_bounds": result["court_bounds"],
                            },
                        )
                    )
                    return events

            # Ultimo fallback: preset
            print("⚠️ Calibrazione campo fallita, uso preset")
            return events

        events.append(
            Event(
                time=0.0,
                type=EventType.FIELD_LINES_DETECTED,
                confidence=0.8,
                extra={
                    "h_lines": len(h_lines),
                    "v_lines": len(v_lines),
                },
            )
        )

        # 5. Trova 4 angoli campo
        corners = self._find_corners(h_lines, v_lines, frame.shape)

        if corners is not None:
            self._corners = corners
            events.append(
                Event(
                    time=0.0,
                    type=EventType.FIELD_CORNERS_DETECTED,
                    confidence=0.9,
                    extra={
                        "corners": corners.tolist(),
                    },
                )
            )

            # 6. Calcola omografia
            self._calculate_homography()

            if self._homography is not None:
                events.append(
                    Event(
                        time=0.0,
                        type=EventType.FIELD_HOMOGRAPHY_READY,
                        confidence=1.0,
                        extra={
                            "homography": self._homography.tolist(),
                            "pixels_per_meter": self._ppm,
                        },
                    )
                )

        return events

    def _filter_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Separa linee orizzontali e verticali."""
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 20 or angle > 160:  # Orizzontale
                h_lines.append(line[0])
            elif 70 < angle < 110:  # Verticale
                v_lines.append(line[0])

        return h_lines, v_lines

    def _find_corners(
        self,
        h_lines: List,
        v_lines: List,
        frame_shape: Tuple,
    ) -> Optional[np.ndarray]:
        """
        Trova i 4 angoli del campo dalle intersezioni delle linee.

        Returns:
            np.ndarray di shape (4, 2) con i 4 angoli, o None
        """
        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Ordina linee orizzontali per Y (top e bottom)
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]

        # Ordina linee verticali per X (left e right)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)
        left_line = v_sorted[0]
        right_line = v_sorted[-1]

        # Calcola intersezioni
        tl = self._line_intersection(top_line, left_line)
        tr = self._line_intersection(top_line, right_line)
        bl = self._line_intersection(bottom_line, left_line)
        br = self._line_intersection(bottom_line, right_line)

        if None in [tl, tr, bl, br]:
            return None

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _line_intersection(self, line1, line2) -> Optional[Tuple[float, float]]:
        """Calcola intersezione tra due linee."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return (px, py)

    def _calculate_homography(self):
        """Calcola matrice omografia dai 4 angoli."""
        if self._corners is None:
            return

        cfg = self.config

        # Punti destinazione (campo reale in metri, scalato)
        # Usiamo 100 pixel per metro come scala interna
        scale = 100
        dst_points = np.array(
            [
                [0, 0],
                [cfg.field_length * scale, 0],
                [cfg.field_length * scale, cfg.field_width * scale],
                [0, cfg.field_width * scale],
            ],
            dtype=np.float32,
        )

        # Calcola omografia
        self._homography, _ = cv2.findHomography(self._corners, dst_points)

        # Calcola pixel per metro
        if self._homography is not None:
            # Distanza tra corner[0] e corner[1] in pixel
            dist_px = np.linalg.norm(self._corners[1] - self._corners[0])
            self._ppm = dist_px / cfg.field_length

    def get_corners(self) -> Optional[np.ndarray]:
        """Restituisce i 4 angoli del campo in pixel."""
        return self._corners

    def get_homography(self) -> Optional[np.ndarray]:
        """Restituisce la matrice omografia 3x3."""
        return self._homography

    def get_pixels_per_meter(self) -> Optional[float]:
        """Restituisce il fattore di conversione pixel/metro."""
        return self._ppm

    def pixel_to_meters(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """
        Converte coordinate pixel in metri sul campo.

        Args:
            px, py: Coordinate in pixel

        Returns:
            (mx, my): Coordinate in metri, o None se non calibrato
        """
        if self._homography is None:
            return None

        pt = np.array([[px, py]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)

        transformed = cv2.perspectiveTransform(pt, self._homography)
        mx = transformed[0, 0, 0] / 100  # Dividi per scala
        my = transformed[0, 0, 1] / 100

        return (mx, my)


__all__ = [
    "FieldAutoCalibrator",
    "FieldAutoConfig",
]

