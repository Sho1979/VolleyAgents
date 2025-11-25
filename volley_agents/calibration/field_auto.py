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

    # Parametri Hough Transform
    hough_threshold: int = 100
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 10

    # Parametri Canny edge detection
    canny_low: int = 50
    canny_high: int = 150

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

        # 1. Converti in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Edge detection
        edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)

        # 3. Hough Transform per linee
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )

        if lines is None:
            return events

        # 4. Filtra linee (orizzontali e verticali)
        h_lines, v_lines = self._filter_lines(lines)

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

