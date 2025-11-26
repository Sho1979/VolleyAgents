"""Test GameStateAgent su segmento video."""
import sys
sys.path.insert(0, ".")

import cv2
from volley_agents.agents.game_state_agent import GameStateAgent, GameStateAgentConfig
from volley_agents.agents.motion_agent import FrameSample

VIDEO_PATH = r"C:\Volley video\Millennium Bienno.mp4"
T_START = 1018  # 16:58
T_END = 1320    # 22:00
FPS = 5

def main():
    print("=== Test GameStateAgent (16:58 - 22:00) ===")
    
    # Config
    config = GameStateAgentConfig(
        window_seconds=3.0,
        stride_seconds=2.0,
        min_confidence=0.6,
        enable_logging=True
    )
    agent = GameStateAgent(config)
    
    # Carica video
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / FPS)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, T_START * 1000)
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > T_END:
            break
        if frame_idx % frame_interval == 0:
            frames.append(FrameSample(frame=frame, time=current_time))
        frame_idx += 1
    
    cap.release()
    print(f"Frame estratti: {len(frames)}")
    
    # Analizza
    events = agent.run(frames)
    
    print(f"\n📊 Totale eventi: {len(events)}")
    
    # Mostra transizioni di stato
    print("\n🎬 Sequenza stati (solo transizioni):")
    prev_state = None
    for e in events:
        state = e.extra["state"]
        if state != prev_state:
            mins = int(e.time // 60)
            secs = e.time % 60
            print(f"  {mins:02d}:{secs:05.2f} -> {state.upper()} ({e.confidence:.0%})")
            prev_state = state

if __name__ == "__main__":
    main()
