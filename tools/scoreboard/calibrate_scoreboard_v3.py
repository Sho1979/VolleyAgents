"""
Quick ROI calibration tool for ScoreboardAgentV3.

Usage:
    python -m tools.scoreboard.calibrate_scoreboard_v3 \
        --video "path/to/match.mp4" \
        --time 1000.0 \
        --output "scoreboard_config.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV (cv2) is required. Install with: pip install opencv-python")
    sys.exit(1)

from volley_agents.io.config import save_scoreboard_roi_config


def _select_roi(frame, title: str):
    roi = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    return roi


def main():
    parser = argparse.ArgumentParser(description="Calibra la ROI principale per ScoreboardAgentV3.")
    parser.add_argument("--video", required=True, help="Percorso del video.")
    parser.add_argument("--time", type=float, default=0.0, help="Secondo del video da utilizzare per l'anteprima.")
    parser.add_argument(
        "--output",
        default="scoreboard_config.json",
        help="File JSON di output con la ROI (default: scoreboard_config.json).",
    )
    parser.add_argument("--led-color", default="red", choices=["red", "green", "yellow", "auto"], help="Colore LED.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_number = int(args.time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: Cannot read frame at time {args.time}s")
        sys.exit(1)

    print("Seleziona il tabellone e premi SPAZIO/INVIO per confermare, ESC per annullare.")
    roi = _select_roi(frame, "Seleziona il tabellone (Scoreboard ROI)")

    if roi[2] <= 0 or roi[3] <= 0:
        print("ROI non valida o selezione annullata.")
        sys.exit(1)

    roi_tuple = tuple(map(int, roi))
    output_path = Path(args.output)
    save_scoreboard_roi_config(
        output_path,
        roi_tuple,  # type: ignore[arg-type]
        video_path=video_path,
        led_color=args.led_color,
    )

    print(f"ROI salvata in {output_path} -> (x={roi_tuple[0]}, y={roi_tuple[1]}, w={roi_tuple[2]}, h={roi_tuple[3]})")
    print("Passa questi valori a ScoreboardConfigV3 oppure carica il JSON direttamente dalla GUI.")


if __name__ == "__main__":
    main()

