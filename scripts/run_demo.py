"""
Demo script per testare l'integrazione degli agenti con Timeline e HeadCoach.

Uso:

    python -m scripts.run_demo --video "path/to/video.mp4" --audio "path/to/audio.wav"

Se non passi i parametri, puoi impostare VIDEO_PATH e AUDIO_PATH di default.
"""

from pathlib import Path
from typing import List, Optional
import argparse

import numpy as np

from volley_agents.core.timeline import Timeline
from volley_agents.core.event import EventType
from volley_agents.core.rally import Rally
from volley_agents.agents.audio_agent import AudioAgent, WhistleDetectorConfig
from volley_agents.agents.motion_agent import MotionAgent, OpticalFlowConfig, FrameSample
from volley_agents.agents.scoreboard_agent import ScoreboardAgent  # opzionale, vedi TODO reader
from volley_agents.fusion.head_coach import HeadCoach, HeadCoachConfig

try:
    import cv2
except ImportError:
    cv2 = None


# -----------------------------
# Helpers
# -----------------------------
def load_video_frames(
    video_path: Path,
    fps: float = 25.0,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> List[FrameSample]:
    """
    Carica frame campionati dal video e li converte in FrameSample(time, frame_ndarray).

    - fps: frequenza di campionamento logico per MotionAgent (non deve essere quella originale).
    - t_start / t_end: finestra di interesse in secondi (se None, fino alla fine).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) non disponibile: impossibile caricare i frame.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(native_fps / fps)))  # ogni quanti frame prendere un campione

    samples: List[FrameSample] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if cur_t < t_start:
            frame_idx += 1
            continue
        if t_end is not None and cur_t > t_end:
            break

        if frame_idx % step == 0:
            # frame in BGR (OpenCV); MotionAgent può lavorare in BGR o convertire a suo piacimento
            samples.append(FrameSample(time=cur_t, frame=frame))

        frame_idx += 1

    cap.release()
    return samples


def print_timeline_summary(timeline: Timeline):
    events = timeline.sorted()
    print("\n==== TIMELINE (eventi ordinati) ====")
    for e in events:
        print(e)

    counts = {}
    for e in events:
        counts[e.type] = counts.get(e.type, 0) + 1

    print("\n==== CONTEGGIO EVENTI PER TIPO ====")
    for etype, cnt in counts.items():
        print(f"{etype.value:20s} : {cnt}")


def print_rallies(rallies: List[Rally]):
    print("\n==== RALLY DECISI DAL HEAD COACH ====")
    for i, r in enumerate(rallies, start=1):
        print(f"{i:02d}: {r}")


# -----------------------------
# Main demo
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="VolleyAgents demo (Audio + Motion + HeadCoach)")
    parser.add_argument("--video", type=str, required=False, help="Percorso del file video (mp4, ecc.)")
    parser.add_argument("--audio", type=str, required=False, help="Percorso del file audio WAV estratto")
    parser.add_argument("--t-start", type=float, default=0.0, help="Secondo iniziale da analizzare")
    parser.add_argument("--t-end", type=float, default=None, help="Secondo finale da analizzare (None = fino a fine)")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS logici per MotionAgent")
    parser.add_argument("--score-first", action="store_true", help="Attiva modalità score-first se SCORE_CHANGE è affidabile")

    args = parser.parse_args()

    # TODO: se non passati, puoi mettere qui dei default per test locali
    if args.video is None or args.audio is None:
        print("⚠️  Nessun video/audio passato. Imposta --video e --audio dalla riga di comando.")
        print("    Esempio:")
        print('    python -m scripts.run_demo --video "C:\\path\\clip.mp4" --audio "C:\\path\\audio.wav"')
        return

    video_path = Path(args.video)
    audio_wav = Path(args.audio)

    print("🎯 DEMO VolleyAgents")
    print(f"Video : {video_path}")
    print(f"Audio : {audio_wav}")
    print(f"Finestra: {args.t_start}–{args.t_end or 'fine'} s  |  fps motion = {args.fps}")

    timeline = Timeline()

    # 1) Audio Agent
    print("\n🎵 Analisi audio (whistle)...")
    audio_cfg = WhistleDetectorConfig()  # usa la tua config reale
    audio_agent = AudioAgent(config=audio_cfg)
    try:
        audio_events = audio_agent.load_and_analyze(audio_wav)
        timeline.extend(audio_events)
        print(f"   ✅ Trovati {len(audio_events)} eventi audio (whistle)")
    except FileNotFoundError:
        print(f"   ⚠️ File audio non trovato: {audio_wav} (skip audio)")
    except Exception as e:
        print(f"   ❌ Errore analisi audio: {e}")

    # 2) Motion Agent
    print("\n🎬 Analisi motion (optical flow SX/DX)...")
    motion_cfg = OpticalFlowConfig()  # usa la tua config reale
    motion_agent = MotionAgent(config=motion_cfg)
    try:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) non disponibile per MotionAgent.")
        frames = load_video_frames(video_path, fps=args.fps, t_start=args.t_start, t_end=args.t_end)
        print(f"   Frame campionati per MotionAgent: {len(frames)}")
        motion_events = motion_agent.run(frames, timeline=timeline)
        print(f"   ✅ Trovati {len(motion_events)} eventi motion (hit/gap)")
    except FileNotFoundError:
        print(f"   ⚠️ File video non trovato: {video_path} (skip motion)")
    except Exception as e:
        print(f"   ❌ Errore motion: {e}")

    # 3) Scoreboard Agent (opzionale: per ora solo se hai già un reader concreto)
    # TODO: integra reader OCR + pipeline reale
    # print("\n📊 Analisi tabellone (scoreboard)...")
    # try:
    #     scoreboard_agent = ScoreboardAgent(reader=my_reader_impl, min_confidence=0.6)
    #     score_events = scoreboard_agent.run(video_path, timeline=timeline)
    #     print(f"   ✅ Trovati {len(score_events)} eventi SCORE_CHANGE")
    # except Exception as e:
    #     print(f"   ⚠️ Scoreboard non analizzato: {e}")

    # 4) Stampa timeline
    print_timeline_summary(timeline)

    # 5) Head Coach
    print("\n🧠 Head Coach: costruzione rally...")
    coach_cfg = HeadCoachConfig(
        score_first_mode=args.score_first,   # per ora lo lasciamo False nella pratica
    )
    coach = HeadCoach(cfg=coach_cfg)
    rallies = coach.build_rallies(timeline)

    print_rallies(rallies)
    print("\n✅ Demo completata.")


if __name__ == "__main__":
    main()
