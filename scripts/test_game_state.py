"""Test GameStatusClassifier su video volleyball."""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch

# Config
VIDEO_PATH = r"C:\Volley video\Millennium Bienno.mp4"
MODEL_PATH = "volley_agents/models/game_state"
T_START = 1020  # 17:00 - inizio rally
T_END = 1035    # 17:15 - fine rally

def main():
    print("=== Test GameStatusClassifier ===")
    
    # Carica modello
    print(f"Caricamento modello da {MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print(f"Classi: {model.config.id2label}")
    
    # Estrai frame dal video
    print(f"\nEstrazione frame da {T_START}s a {T_END}s...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_MSEC, T_START * 1000)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time > T_END:
            break
        # Converti BGR -> RGB e resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)
    
    cap.release()
    print(f"Frame estratti: {len(frames)}")
    
    # Seleziona 16 frame uniformemente
    if len(frames) > 16:
        step = len(frames) / 16
        selected = [frames[int(i * step)] for i in range(16)]
    else:
        selected = frames
    
    print(f"Frame selezionati: {len(selected)}")
    
    # Preprocessing
    inputs = processor(selected, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    print("\nInferenza...")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    pred_class = model.config.id2label[pred_id]
    print(f"\n🎯 Predizione: {pred_class} (confidence: {confidence:.2%})")
    print(f"   Tutte le probabilità:")
    for i, label in model.config.id2label.items():
        print(f"     {label}: {probs[0][i].item():.2%}")

if __name__ == "__main__":
    main()


