# volley_agents/agents/scoreboard_agent.py

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from volley_agents.core.event import Event, EventType


@dataclass
class ScoreboardAgentConfig:
    # ROI del tabellone in pixel (x, y, w, h)
    roi_scoreboard: Tuple[int, int, int, int]

    # quante letture consecutive servono per considerare stabile un punteggio
    stability_window: int = 7

    # minimo numero di letture uguali nella finestra per emettere SCORE_CHANGE
    stability_min_count: int = 5

    # fattore di upscaling (per avere cifre leggibili dall'OCR/segmenti)
    upscale_factor: int = 4

    # soglia binarizzazione (0–255)
    threshold: int = 180

    # se il tabellone ha punteggi max 0–99
    max_score: int = 99


class ScoreboardAgent:
    """
    Legge il tabellone LED (rosso/giallo) da una ROI fissa,
    decodifica i punteggi delle due squadre e genera eventi SCORE_CHANGE
    stabilizzati temporalmente.
    """

    def __init__(self, cfg: ScoreboardAgentConfig, enable_logging: bool = False):
        self.cfg = cfg
        self.enable_logging = enable_logging

        # history di letture (left_score, right_score)
        self._history: Deque[Tuple[int, int]] = deque(maxlen=cfg.stability_window)

        # ultimo punteggio emesso come SCORE_CHANGE
        self._last_emitted: Optional[Tuple[int, int]] = None

    # ------------------- API principale -------------------

    def process_frame(self, frame_bgr: np.ndarray, t: float) -> List[Event]:
        """
        Analizza un singolo frame al tempo t.

        Returns:
            Lista di Event (tipicamente vuota, oppure 1 SCORE_CHANGE quando rilevato).
        """
        x, y, w, h = self.cfg.roi_scoreboard
        h_frame, w_frame = frame_bgr.shape[:2]

        # clamp ROI ai limiti del frame
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        roi = frame_bgr[y : y + h, x : x + w]

        left_score, right_score = self._read_scores_from_roi(roi)
        if left_score is None or right_score is None:
            # lettura fallita → non aggiorniamo history per non sporcarla
            return []

        # aggiorna history
        self._history.append((left_score, right_score))

        # controlla stabilità
        stable = self._get_stable_score()
        events: List[Event] = []

        if stable is not None and stable != self._last_emitted:
            # nuovo punteggio stabile → SCORE_CHANGE
            left, right = stable
            self._last_emitted = stable
            extra = {"score_left": int(left), "score_right": int(right)}

            evt = Event(
                time=t,
                type=EventType.SCORE_CHANGE,
                confidence=1.0,  # punteggio già stabilizzato
                extra=extra,
            )
            events.append(evt)

            if self.enable_logging:
                print(f"[ScoreboardAgent] SCORE_CHANGE @ {t:.2f}s -> {left}-{right}")

        return events

    # ------------------- Lettura punteggi -------------------

    def _read_scores_from_roi(self, roi_bgr: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Legge i punteggi left/right dalla ROI del tabellone.

        Strategia:
        - converte in gray
        - upscaling
        - threshold
        - split in due metà (sinistra/destra)
        - per ogni metà, decode 2 cifre (tens + ones) usando segmenti.
        """
        cfg = self.cfg

        # grigio
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # upscaling
        gray_big = cv2.resize(
            gray,
            None,
            fx=cfg.upscale_factor,
            fy=cfg.upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        # threshold: i LED sono molto luminosi -> consideriamo "bianco" il valore alto
        _, thr = cv2.threshold(gray_big, cfg.threshold, 255, cv2.THRESH_BINARY)

        h, w = thr.shape

        # split metà sinistra/destra per i due punteggi
        mid_x = w // 2
        left_img = thr[:, :mid_x]
        right_img = thr[:, mid_x:]

        left_score = self._decode_two_digit_score(left_img)
        right_score = self._decode_two_digit_score(right_img)

        # sanity check
        if left_score is not None and (left_score < 0 or left_score > cfg.max_score):
            left_score = None
        if right_score is not None and (right_score < 0 or right_score > cfg.max_score):
            right_score = None

        return left_score, right_score

    def _decode_two_digit_score(self, img: np.ndarray) -> Optional[int]:
        """
        Decodifica un punteggio a due cifre da una sotto-immagine del tabellone.

        Approccio semplice:
        - split verticale in 2 blocchi (tens, ones)
        - per ciascuno, classifica la cifra 0–9 tramite segmenti predefiniti.
        """
        h, w = img.shape
        if w < 10 or h < 10:
            return None

        # margini per evitare bordi sporchi
        margin_x = int(0.05 * w)
        margin_y = int(0.10 * h)
        core = img[margin_y : h - margin_y, margin_x : w - margin_x]
        h_c, w_c = core.shape

        mid_x = w_c // 2
        tens_img = core[:, :mid_x]
        ones_img = core[:, mid_x:]

        tens_digit = self._decode_single_digit(tens_img)
        ones_digit = self._decode_single_digit(ones_img)

        if tens_digit is None and ones_digit is None:
            return None
        if tens_digit is None:
            tens_digit = 0  # es. 8 → "08"

        return 10 * tens_digit + (ones_digit or 0)

    def _decode_single_digit(self, img: np.ndarray) -> Optional[int]:
        """
        Decodifica una singola cifra 0–9 da un display a 7 segmenti.

        Definiamo 7 regioni (a,b,c,d,e,f,g) e guardiamo se sono accese/spente.
        """
        h, w = img.shape
        if w < 4 or h < 6:
            return None

        # definisci 7 regioni come frazioni
        #   --a--
        #  |     |
        #  f     b
        #  |     |
        #   --g--
        #  |     |
        #  e     c
        #  |     |
        #   --d--

        # helper per media pixel in una sotto-regione
        def region_on(x0, y0, x1, y1) -> bool:
            x0 = max(0, min(x0, w - 1))
            x1 = max(0, min(x1, w - 1))
            y0 = max(0, min(y0, h - 1))
            y1 = max(0, min(y1, h - 1))
            if x1 <= x0 or y1 <= y0:
                return False
            roi = img[y0:y1, x0:x1]
            # siccome il threshold ha portato LED al bianco, guardiamo la % di white
            white_ratio = np.mean(roi == 255)
            return white_ratio > 0.4  # threshold da calibrare se serve

        # coordinate segmenti in termini di frazioni di w,h
        a = region_on(int(0.20 * w), int(0.00 * h), int(0.80 * w), int(0.20 * h))
        d = region_on(int(0.20 * w), int(0.80 * h), int(0.80 * w), int(1.00 * h))
        g = region_on(int(0.20 * w), int(0.40 * h), int(0.80 * w), int(0.60 * h))

        f = region_on(int(0.00 * w), int(0.20 * h), int(0.30 * w), int(0.50 * h))
        e = region_on(int(0.00 * w), int(0.50 * h), int(0.30 * w), int(0.80 * h))

        b = region_on(int(0.70 * w), int(0.20 * h), int(1.00 * w), int(0.50 * h))
        c = region_on(int(0.70 * w), int(0.50 * h), int(1.00 * w), int(0.80 * h))

        pattern = (a, b, c, d, e, f, g)

        # mappa pattern segmenti → cifra (0–9)
        # True=segmento acceso, False=spento
        SEGMENTS = {
            (True,  True,  True,  True,  True,  True,  False): 0,
            (False, True,  True,  False, False, False, False): 1,
            (True,  True,  False, True,  True,  False, True ): 2,
            (True,  True,  True,  True,  False, False, True ): 3,
            (False, True,  True,  False, False, True,  True ): 4,
            (True,  False, True,  True,  False, True,  True ): 5,
            (True,  False, True,  True,  True,  True,  True ): 6,
            (True,  True,  True,  False, False, False, False): 7,
            (True,  True,  True,  True,  True,  True,  True ): 8,
            (True,  True,  True,  True,  False, True,  True ): 9,
        }

        return SEGMENTS.get(pattern, None)

    # ------------------- Stabilità temporale -------------------

    def _get_stable_score(self) -> Optional[Tuple[int, int]]:
        """
        Ritorna il punteggio stabile nella finestra history, se presente.

        Criterio:
        - prendi il valore (left,right) più frequente nella finestra
        - se compare almeno stability_min_count volte, consideralo stabile
        """
        cfg = self.cfg
        if len(self._history) < cfg.stability_min_count:
            return None

        # conteggio frequenze
        counter = {}
        for s in self._history:
            counter[s] = counter.get(s, 0) + 1

        # trova il punteggio più frequente
        stable_score, count = max(counter.items(), key=lambda kv: kv[1])

        if count >= cfg.stability_min_count:
            return stable_score
        return None


__all__ = [
    "ScoreboardAgent",
    "ScoreboardAgentConfig",
]
