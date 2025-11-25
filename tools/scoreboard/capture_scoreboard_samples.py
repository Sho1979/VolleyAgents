"""
Calibration tool for ScoreboardAgentV2 ROI configuration.

This script helps calibrate the ROI coordinates for the LED scoreboard by:
- Loading a video frame
- Displaying the frame with ROI overlay
- Allowing interactive ROI selection
- Saving sample images of the main ROI and digit sub-ROIs
- Logging the configuration values

Usage:
    python -m tools.scoreboard.capture_scoreboard_samples --video path/to/video.mp4 --time 1000.0
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: OpenCV (cv2) is required. Install with: pip install opencv-python")
    sys.exit(1)


def select_roi_interactive(frame: np.ndarray, title: str) -> tuple:
    """
    Interactive ROI selection using OpenCV.
    
    Args:
        frame: Frame to select ROI from
        title: Window title
        
    Returns:
        Tuple (x, y, w, h) or (0, 0, 0, 0) if cancelled
    """
    roi = cv2.selectROI(title, frame, False)
    cv2.destroyAllWindows()
    
    if roi[2] > 0 and roi[3] > 0:  # w > 0 and h > 0
        return tuple(map(int, roi))
    return (0, 0, 0, 0)


def draw_roi_overlay(frame: np.ndarray, roi: tuple, color=(0, 255, 0), thickness=2):
    """Draw ROI rectangle on frame."""
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ScoreboardAgentV2 ROI configuration"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Time in seconds to extract frame (default: 0.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tools/scoreboard/samples",
        help="Output directory for sample images (default: tools/scoreboard/samples)"
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video and extract frame
    print(f"Loading video: {video_path}")
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
        print(f"ERROR: Cannot read frame at time {args.time}s (frame {frame_number})")
        sys.exit(1)
    
    print(f"Frame extracted at {args.time:.2f}s (frame {frame_number})")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print()
    
    # Step 1: Select main ROI (entire scoreboard)
    print("=" * 60)
    print("STEP 1: Select MAIN ROI (entire scoreboard)")
    print("=" * 60)
    print("Click and drag to select the entire scoreboard area.")
    print("Press SPACE or ENTER to confirm, ESC to cancel.")
    print()
    
    main_roi = select_roi_interactive(frame.copy(), "Select MAIN ROI (entire scoreboard)")
    if main_roi == (0, 0, 0, 0):
        print("ERROR: Main ROI selection cancelled")
        sys.exit(1)
    
    x_main, y_main, w_main, h_main = main_roi
    print(f"Main ROI: x={x_main}, y={y_main}, w={w_main}, h={h_main}")
    
    # Extract main ROI for sub-selection
    main_roi_img = frame[y_main:y_main+h_main, x_main:x_main+w_main]
    if main_roi_img.size == 0:
        print("ERROR: Invalid main ROI")
        sys.exit(1)
    
    # Save main ROI sample
    main_roi_path = output_dir / "main_roi.png"
    cv2.imwrite(str(main_roi_path), main_roi_img)
    print(f"Saved main ROI sample: {main_roi_path}")
    print()
    
    # Step 2: Select digit sub-ROIs (relative to main ROI)
    print("=" * 60)
    print("STEP 2: Select DIGIT sub-ROIs (relative to main ROI)")
    print("=" * 60)
    print("For each digit, select the ROI in the main scoreboard image.")
    print("Coordinates will be relative to the main ROI.")
    print()
    
    digits = [
        ("HOME TENS", "digit_home_tens"),
        ("HOME UNITS", "digit_home_units"),
        ("AWAY TENS", "digit_away_tens"),
        ("AWAY UNITS", "digit_away_units"),
    ]
    
    digit_rois = {}
    
    for display_name, config_name in digits:
        print(f"Select ROI for {display_name}...")
        roi_rel = select_roi_interactive(
            main_roi_img.copy(),
            f"Select ROI for {display_name} (relative to main ROI)"
        )
        
        if roi_rel == (0, 0, 0, 0):
            print(f"WARNING: {display_name} selection cancelled, skipping")
            continue
        
        x_rel, y_rel, w_rel, h_rel = roi_rel
        digit_rois[config_name] = (x_rel, y_rel, w_rel, h_rel)
        print(f"  {display_name}: x={x_rel}, y={y_rel}, w={w_rel}, h={h_rel}")
        
        # Save digit sample
        digit_img = main_roi_img[y_rel:y_rel+h_rel, x_rel:x_rel+w_rel]
        if digit_img.size > 0:
            digit_path = output_dir / f"{config_name}.png"
            cv2.imwrite(str(digit_path), digit_img)
            print(f"  Saved sample: {digit_path}")
        print()
    
    # Generate configuration code
    print("=" * 60)
    print("CONFIGURATION GENERATED")
    print("=" * 60)
    print()
    print("Use this configuration in your code:")
    print()
    print("from volley_agents.agents.scoreboard_v2 import ScoreboardConfig")
    print()
    print("scoreboard_config = ScoreboardConfig(")
    print(f"    x={x_main},")
    print(f"    y={y_main},")
    print(f"    w={w_main},")
    print(f"    h={h_main},")
    
    for config_name in ["digit_home_tens", "digit_home_units", "digit_away_tens", "digit_away_units"]:
        if config_name in digit_rois:
            x, y, w, h = digit_rois[config_name]
            print(f"    {config_name}=({x}, {y}, {w}, {h}),")
        else:
            print(f"    {config_name}=(0, 0, 0, 0),  # TODO: Calibrate")
    
    print("    history_size=15,")
    print("    min_stable_frames=5,")
    print(")")
    print()
    
    # Create visualization with all ROIs
    vis_frame = frame.copy()
    draw_roi_overlay(vis_frame, main_roi, color=(0, 255, 0), thickness=2)
    
    for config_name, roi_rel in digit_rois.items():
        x_rel, y_rel, w_rel, h_rel = roi_rel
        # Convert relative to absolute coordinates
        x_abs = x_main + x_rel
        y_abs = y_main + y_rel
        draw_roi_overlay(
            vis_frame,
            (x_abs, y_abs, w_rel, h_rel),
            color=(255, 0, 0),
            thickness=1
        )
    
    vis_path = output_dir / "roi_visualization.png"
    cv2.imwrite(str(vis_path), vis_frame)
    print(f"Saved visualization with all ROIs: {vis_path}")
    print()
    print("Calibration complete!")


if __name__ == "__main__":
    main()

