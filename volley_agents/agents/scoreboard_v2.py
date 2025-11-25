"""
ScoreboardAgent v2 - Robust LED scoreboard reading with 7-segment digit recognition.

This agent reads the score from a fixed-position LED scoreboard using:
- ROI-based extraction
- 7-segment digit recognition (no neural networks)
- Temporal smoothing to avoid noisy events
- Coherent score change validation (+1 on exactly one team)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from collections import Counter, deque
from pathlib import Path

import numpy as np
import cv2

from volley_agents.core.event import Event, EventType


# Predefined scoreboard profiles for different video types/courts
# ROI coordinates are normalized (0.0-1.0) relative to frame size
# Format: (x, y, w, h) where x,y are top-left corner, w,h are width and height
SCOREBOARD_PROFILES = {
    "montichiari_640x360": {
        "frame_size": (640, 360),  # (width, height)
        # ROI misurata: (353, 50, 388, 74) -> (x=353, y=50, w=35, h=24)
        "main_roi": (0.5516, 0.1389, 0.0547, 0.0667),  # (x, y, w, h) normalized
        "description": "Palestra Montichiari - 640x360",
    },
    # Add more profiles as needed
    # Example:
    # "another_court_1920x1080": {
    #     "frame_size": (1920, 1080),
    #     "main_roi": (0.5, 0.1, 0.3, 0.15),
    #     "description": "Another court - 1920x1080",
    # },
}


def find_scoreboard_profile(frame_width: int, frame_height: int) -> Optional[dict]:
    """
    Find a matching scoreboard profile based on frame dimensions.
    
    Args:
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        Profile dict or None if no match found
    """
    frame_size = (frame_width, frame_height)
    
    for profile_name, profile in SCOREBOARD_PROFILES.items():
        if profile["frame_size"] == frame_size:
            return profile
    
    return None


def normalize_roi_to_absolute(roi_normalized: Tuple[float, float, float, float], 
                                frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """
    Convert normalized ROI (0.0-1.0) to absolute pixel coordinates.
    
    Args:
        roi_normalized: (x, y, w, h) normalized coordinates
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        
    Returns:
        (x, y, w, h) absolute pixel coordinates
    """
    x_norm, y_norm, w_norm, h_norm = roi_normalized
    x = int(x_norm * frame_width)
    y = int(y_norm * frame_height)
    w = int(w_norm * frame_width)
    h = int(h_norm * frame_height)
    return (x, y, w, h)


@dataclass
class ScoreboardConfig:
    """Configuration for ScoreboardAgentV2."""
    
    # Main ROI of the scoreboard in absolute coordinates (pixels)
    x: int
    y: int
    w: int
    h: int
    
    # Sub-ROIs for the 4 digits (all relative to the main ROI)
    # Format: (x_offset, y_offset, width, height) relative to main ROI
    digit_home_tens: Tuple[int, int, int, int]
    digit_home_units: Tuple[int, int, int, int]
    digit_away_tens: Tuple[int, int, int, int]
    digit_away_units: Tuple[int, int, int, int]
    
    # Temporal smoothing parameters
    history_size: int = 15  # number of frames to keep in history
    min_stable_frames: int = 5  # minimum frames with same score before emitting event
    
    # Image processing parameters
    threshold: int = 180  # binary threshold (0-255)
    segment_activation_ratio: float = 0.4  # minimum white pixel ratio for segment to be "on"
    upscale_factor: int = 2  # upscale factor for digit recognition


class ScoreboardAgentV2:
    """
    ScoreboardAgent v2 - Reads LED scoreboard with 7-segment digit recognition.
    
    Features:
    - Fixed camera position (scoreboard always in same position)
    - Reads 4 digits: home tens, home units, away tens, away units
    - Temporal smoothing to avoid noisy readings
    - Emits SCORE_CHANGE only when score changes coherently (+1 on one team)
    """
    
    def __init__(self, cfg: ScoreboardConfig, enable_logging: bool = False, log_callback: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.enable_logging = enable_logging
        self.log_callback = log_callback  # Optional callback for logging (e.g., GUI log)
        self._history: deque = deque(maxlen=cfg.history_size)
        self._last_emitted: Optional[Tuple[int, int]] = None
        self._stable_count: int = 0  # count of consecutive frames with same score
        self._frame_count: int = 0  # counter for debug logging
        self._debug_save_count: int = 0  # counter for saving debug images
        
        # 7-segment pattern mapping
        # Pattern: (a, b, c, d, e, f, g) where True = segment on
        self._SEGMENT_TO_DIGIT = {
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
    
    def _log(self, msg: str):
        """Log a message if logging is enabled."""
        if self.enable_logging:
            if self.log_callback:
                self.log_callback(msg)
            else:
                print(f"[ScoreboardAgentV2] {msg}")

    def process_frame(self, frame_bgr: np.ndarray, t: float) -> List[Event]:
        """
        Process a single frame and return list of events (typically empty or 1 SCORE_CHANGE).
        
        Args:
            frame_bgr: BGR frame from video
            t: timestamp in seconds
            
        Returns:
            List of Event objects (empty or containing SCORE_CHANGE)
        """
        self._frame_count += 1
        score = self._extract_score(frame_bgr)
        
        if score is None:
            # Failed to read score - don't update history to avoid noise
            if self._frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                self._log(f"Frame {self._frame_count} @ {t:.2f}s: Lettura punteggio fallita")
            return []
        
        home, away = score
        if self._frame_count % 30 == 0:  # Log every 30 frames
            self._log(f"Frame {self._frame_count} @ {t:.2f}s: Punteggio letto = {home}-{away}")
        
        return self._update_history(score, t)
    
    def _extract_score(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Extract score from frame by reading the 4 digits from the scoreboard ROI.
        
        Args:
            frame_bgr: BGR frame
            
        Returns:
            Tuple (home_score, away_score) or None if reading failed
        """
        cfg = self.cfg
        
        # Clamp ROI to frame bounds
        h_frame, w_frame = frame_bgr.shape[:2]
        x = max(0, min(cfg.x, w_frame - 1))
        y = max(0, min(cfg.y, h_frame - 1))
        w = max(1, min(cfg.w, w_frame - x))
        h = max(1, min(cfg.h, h_frame - y))
        
        # Extract main ROI
        main_roi = frame_bgr[y:y+h, x:x+w]
        if main_roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(main_roi, cv2.COLOR_BGR2GRAY)
        
        # Upscale for better digit recognition
        if cfg.upscale_factor > 1:
            gray = cv2.resize(
                gray,
                None,
                fx=cfg.upscale_factor,
                fy=cfg.upscale_factor,
                interpolation=cv2.INTER_CUBIC
            )
            # Scale sub-ROI coordinates
            scale = cfg.upscale_factor
        else:
            scale = 1
        
        # Binary threshold (LED displays are bright, so high threshold)
        _, binary = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)
        
        # Save debug images occasionally if logging enabled
        if self.enable_logging and self._debug_save_count < 3:
            try:
                debug_dir = Path("tools/scoreboard/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / f"main_roi_{self._debug_save_count}.png"), main_roi)
                cv2.imwrite(str(debug_dir / f"binary_{self._debug_save_count}.png"), binary)
                self._log(f"  Immagini debug salvate in {debug_dir}")
                self._debug_save_count += 1
            except Exception as e:
                self._log(f"  Errore salvataggio debug: {e}")
        
        # Extract and decode each digit
        home_tens = self._decode_digit(binary, cfg.digit_home_tens, scale)
        home_units = self._decode_digit(binary, cfg.digit_home_units, scale)
        away_tens = self._decode_digit(binary, cfg.digit_away_tens, scale)
        away_units = self._decode_digit(binary, cfg.digit_away_units, scale)
        
        # Debug logging
        if self.enable_logging and self._frame_count % 30 == 0:
            self._log(f"  Digit recognition: HOME tens={home_tens}, units={home_units}, AWAY tens={away_tens}, units={away_units}")
        
        # Validate all digits were read
        if None in (home_tens, home_units, away_tens, away_units):
            if self.enable_logging and self._frame_count % 30 == 0:
                failed = []
                if home_tens is None: failed.append("HOME_TENS")
                if home_units is None: failed.append("HOME_UNITS")
                if away_tens is None: failed.append("AWAY_TENS")
                if away_units is None: failed.append("AWAY_UNITS")
                self._log(f"  Lettura fallita per: {', '.join(failed)}")
            return None
        
        home_score = 10 * home_tens + home_units
        away_score = 10 * away_tens + away_units
        
        # Sanity check: scores should be reasonable (0-99)
        if not (0 <= home_score <= 99) or not (0 <= away_score <= 99):
            return None
        
        return (home_score, away_score)
    
    def _decode_digit(self, binary_img: np.ndarray, roi_rel: Tuple[int, int, int, int], scale: int) -> Optional[int]:
        """
        Decode a single digit from a sub-ROI using 7-segment recognition.
        
        Args:
            binary_img: Binary image (main ROI, possibly upscaled)
            roi_rel: Relative ROI (x, y, w, h) in original coordinates
            scale: Scale factor if image was upscaled
            
        Returns:
            Digit 0-9 or None if recognition failed
        """
        x_rel, y_rel, w_rel, h_rel = roi_rel
        x = int(x_rel * scale)
        y = int(y_rel * scale)
        w = int(w_rel * scale)
        h = int(h_rel * scale)
        
        h_img, w_img = binary_img.shape
        if x + w > w_img or y + h > h_img or w < 4 or h < 6:
            return None
        
        # Extract digit ROI
        digit_roi = binary_img[y:y+h, x:x+w]
        if digit_roi.size == 0:
            return None
        
        # Apply margins to avoid edge noise
        margin_x = max(1, int(0.05 * w))
        margin_y = max(1, int(0.10 * h))
        core = digit_roi[margin_y:h-margin_y, margin_x:w-margin_x]
        h_c, w_c = core.shape
        
        if w_c < 4 or h_c < 6:
            return None
        
        # Define 7 segments as regions
        # Layout:
        #   --a--
        #  |     |
        #  f     b
        #  |     |
        #   --g--
        #  |     |
        #  e     c
        #  |     |
        #   --d--
        
        def region_on(x0, y0, x1, y1) -> bool:
            """Check if a region is 'on' (has enough white pixels)."""
            x0 = max(0, min(x0, w_c - 1))
            x1 = max(0, min(x1, w_c - 1))
            y0 = max(0, min(y0, h_c - 1))
            y1 = max(0, min(y1, h_c - 1))
            if x1 <= x0 or y1 <= y0:
                return False
            roi = core[y0:y1, x0:x1]
            white_ratio = np.mean(roi == 255)
            return white_ratio > self.cfg.segment_activation_ratio
        
        # Define segment regions as fractions of digit size
        a = region_on(int(0.20 * w_c), int(0.00 * h_c), int(0.80 * w_c), int(0.20 * h_c))
        d = region_on(int(0.20 * w_c), int(0.80 * h_c), int(0.80 * w_c), int(1.00 * h_c))
        g = region_on(int(0.20 * w_c), int(0.40 * h_c), int(0.80 * w_c), int(0.60 * h_c))
        
        f = region_on(int(0.00 * w_c), int(0.20 * h_c), int(0.30 * w_c), int(0.50 * h_c))
        e = region_on(int(0.00 * w_c), int(0.50 * h_c), int(0.30 * w_c), int(0.80 * h_c))
        
        b = region_on(int(0.70 * w_c), int(0.20 * h_c), int(1.00 * w_c), int(0.50 * h_c))
        c = region_on(int(0.70 * w_c), int(0.50 * h_c), int(1.00 * w_c), int(0.80 * h_c))
        
        pattern = (a, b, c, d, e, f, g)
        
        return self._SEGMENT_TO_DIGIT.get(pattern, None)
    
    def _update_history(self, score: Tuple[int, int], t: float) -> List[Event]:
        """
        Update history with new score and check if we should emit SCORE_CHANGE event.
        
        Args:
            score: Tuple (home, away)
            t: timestamp in seconds
            
        Returns:
            List of events (empty or containing SCORE_CHANGE)
        """
        cfg = self.cfg
        
        # Add to history
        self._history.append(score)
        
        if len(self._history) < cfg.min_stable_frames:
            # Not enough history yet
            return []
        
        # Find mode (most frequent score) in recent history
        counter = Counter(self._history)
        mode_score, mode_count = counter.most_common(1)[0]
        
        # Check if mode is stable enough
        if mode_count < cfg.min_stable_frames:
            return []
        
        # Check if score changed coherently
        if self._last_emitted is None:
            # First emission - just emit current stable score
            self._last_emitted = mode_score
            self._stable_count = mode_count
            return self._create_score_event(mode_score, t)
        
        # Check if mode is different from last emitted
        if mode_score == self._last_emitted:
            # Same score - no change
            return []
        
        # Score changed - validate coherence
        old_home, old_away = self._last_emitted
        new_home, new_away = mode_score
        
        # Coherent change: exactly +1 on one team, other unchanged
        home_diff = new_home - old_home
        away_diff = new_away - old_away
        
        is_coherent = (
            (home_diff == 1 and away_diff == 0) or
            (home_diff == 0 and away_diff == 1)
        )
        
        if not is_coherent:
            # Incoherent change - ignore (might be reading error)
            return []
        
        # Coherent change detected - emit event
        self._last_emitted = mode_score
        self._stable_count = mode_count
        home, away = mode_score
        self._log(f"SCORE_CHANGE rilevato @ {t:.2f}s: {home}-{away} (stabile per {mode_count} frame)")
        return self._create_score_event(mode_score, t)
    
    def _create_score_event(self, score: Tuple[int, int], t: float) -> List[Event]:
        """Create a SCORE_CHANGE event."""
        home, away = score
        # Include both formats for compatibility:
        # - "home"/"away" as requested
        # - "score_left"/"score_right" for compatibility with existing code (MasterCoach)
        return [Event(
            time=t,
            type=EventType.SCORE_CHANGE,
            confidence=1.0,
            extra={
                "home": int(home),
                "away": int(away),
                "score_left": int(home),  # compatibility with MasterCoach
                "score_right": int(away),  # compatibility with MasterCoach
            }
        )]


__all__ = [
    "ScoreboardAgentV2",
    "ScoreboardConfig",
]

