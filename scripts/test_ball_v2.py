"""Test rapido BallAgentV2 con modello ONNX."""
import sys
sys.path.insert(0, ".")
from pathlib import Path
import cv2
from volley_agents.agents.ball_agent_v2 import BallAgentV2, BallAgentV2Config
from volley_agents.agents.motion_agent import FrameSample

# Config
VIDEO_PATH = r"C:\Volley video\Millennium Bienno.mp4"
T_START = 1018  # 16:58
T_END = 1050    # ~17:30 (solo 30 secondi per test veloce)
FPS = 5

def main():
    print("=== Test BallAgentV2 ONNX ===")
    
    # Inizializza agent
    config = BallAgentV2Config(
        model_path="volley_agents/models/ball_tracking/VballNetV1b_seq9_grayscale_best.onnx",
        enable_logging=True
    )
    agent = BallAgentV2(config)
    
    # Carica video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERRORE: impossibile aprire {VIDEO_PATH}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")
    
    # Estrai frame
    frames = []
    frame_interval = int(video_fps / FPS)
    
    cap.set(cv2.CAP_PROP_POS_MSEC, T_START * 1000)
    
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
    
    # Esegui detection
    print("\nEseguo detection...")
    events = agent.run(frames)
    
    print(f"\nEventi totali: {len(events)}")
    
    # Statistiche per tipo
    by_type = {}
    for e in events:
        by_type[e.type.name] = by_type.get(e.type.name, 0) + 1
    
    for t, c in by_type.items():
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()