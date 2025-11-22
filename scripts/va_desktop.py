"""
VolleyAgents Desktop GUI (prototipo)

- Permette di selezionare un video e un file audio WAV.
- Permette di scegliere una finestra temporale (t_start, t_end) in secondi.
- Lancia gli agenti Audio + Motion e il HeadCoach per trovare i rally.
- Mostra i rally trovati in una lista.
- Opzionale: esporta i clip dei rally con ffmpeg in una cartella di output.
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from volley_agents.core.timeline import Timeline
from volley_agents.core.rally import Rally
from volley_agents.agents.audio_agent import AudioAgent, WhistleDetectorConfig
from volley_agents.agents.motion_agent import MotionAgent, OpticalFlowConfig, FrameSample
from volley_agents.agents.serve_agent import ServeAgent, ServeAgentConfig
from volley_agents.agents.touch_sequence_agent import TouchSequenceAgent, TouchSequenceConfig
from volley_agents.fusion.head_coach import HeadCoach, HeadCoachConfig
from volley_agents.fusion.master_coach import MasterCoach, MasterCoachConfig

try:
    import cv2
except ImportError:
    cv2 = None


class VolleyAgentsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VolleyAgents Desktop v1 (Multi-Agent Demo)")
        self.geometry("900x600")

        self.video_path = tk.StringVar()
        self.audio_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.t_start = tk.StringVar(value="1170")
        self.t_end = tk.StringVar(value="1200")
        self.fps = tk.StringVar(value="5")
        self.lookback = tk.StringVar(value="8")
        self.post_roll = tk.StringVar(value="1.5")  # coda extra per vedere palla che cade

        self.rallies: List[Rally] = []
        self.roi_left: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.roi_right: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

        self._build_ui()

    # ------------------------
    # UI
    # ------------------------
    def _build_ui(self):
        frm_top = ttk.Frame(self, padding=10)
        frm_top.pack(fill="x")

        # Riga Video
        ttk.Label(frm_top, text="Video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.video_path, width=80).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(frm_top, text="Seleziona...", command=self.on_browse_video).grid(row=0, column=2, padx=5)

        # Riga Audio
        ttk.Label(frm_top, text="Audio WAV:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.audio_path, width=80).grid(row=1, column=1, sticky="we", padx=5)
        ttk.Button(frm_top, text="Seleziona...", command=self.on_browse_audio).grid(row=1, column=2, padx=5)

        # Riga Output
        ttk.Label(frm_top, text="Cartella Output:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_top, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", padx=5)
        ttk.Button(frm_top, text="Seleziona...", command=self.on_browse_output).grid(row=2, column=2, padx=5)

        # Parametri finestra e fps
        frm_params = ttk.LabelFrame(self, text="Parametri analisi", padding=10)
        frm_params.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_params, text="t_start (s):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_params, textvariable=self.t_start, width=8).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(frm_params, text="t_end (s):").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm_params, textvariable=self.t_end, width=8).grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(frm_params, text="fps motion:").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm_params, textvariable=self.fps, width=5).grid(row=0, column=5, sticky="w", padx=5)

        ttk.Label(frm_params, text="lookback (s):").grid(row=0, column=6, sticky="w")
        ttk.Entry(frm_params, textvariable=self.lookback, width=5).grid(row=0, column=7, sticky="w", padx=5)

        ttk.Label(frm_params, text="Post-Roll (s):").grid(row=0, column=8, sticky="w")
        ttk.Entry(frm_params, textvariable=self.post_roll, width=5).grid(row=0, column=9, sticky="w", padx=5)

        # Pulsanti ROI
        frm_roi = ttk.LabelFrame(self, text="ROI Battuta (zona battuta)", padding=10)
        frm_roi.pack(fill="x", padx=10, pady=5)

        ttk.Button(frm_roi, text="ROI SX", command=self.on_select_roi_left).pack(side="left", padx=5)
        ttk.Button(frm_roi, text="ROI DX", command=self.on_select_roi_right).pack(side="left", padx=5)
        self.lbl_roi_status = ttk.Label(frm_roi, text="ROI: SX=❌ DX=❌")
        self.lbl_roi_status.pack(side="left", padx=10)

        # Pulsanti
        frm_buttons = ttk.Frame(self, padding=10)
        frm_buttons.pack(fill="x")

        ttk.Button(frm_buttons, text="Analizza", command=self.on_analyze).pack(side="left", padx=5)
        ttk.Button(frm_buttons, text="Esporta Rally", command=self.on_export_rallies).pack(side="left", padx=5)

        # Log + lista rally
        frm_main = ttk.PanedWindow(self, orient="horizontal")
        frm_main.pack(fill="both", expand=True, padx=10, pady=10)

        # Text log
        frm_log = ttk.Frame(frm_main)
        frm_main.add(frm_log, weight=1)
        ttk.Label(frm_log, text="Log").pack(anchor="w")
        self.txt_log = tk.Text(frm_log, height=20)
        self.txt_log.pack(fill="both", expand=True)

        # Lista rally
        frm_rallies = ttk.Frame(frm_main)
        frm_main.add(frm_rallies, weight=1)
        ttk.Label(frm_rallies, text="Rally trovati").pack(anchor="w")
        self.lst_rallies = tk.Listbox(frm_rallies)
        self.lst_rallies.pack(fill="both", expand=True)

    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    # ------------------------
    # Callbacks UI
    # ------------------------
    def on_browse_video(self):
        path = filedialog.askopenfilename(
            title="Seleziona video",
            filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi;*.m4v"), ("Tutti i file", "*.*")],
        )
        if path:
            self.video_path.set(path)

    def on_browse_audio(self):
        path = filedialog.askopenfilename(
            title="Seleziona audio WAV",
            filetypes=[("Audio WAV", "*.wav"), ("Tutti i file", "*.*")],
        )
        if path:
            self.audio_path.set(path)

    def on_browse_output(self):
        path = filedialog.askdirectory(title="Seleziona cartella output")
        if path:
            self.output_dir.set(path)

    def on_select_roi_left(self):
        """Seleziona ROI per la zona battuta sinistra."""
        video_path = Path(self.video_path.get())
        if not video_path.exists():
            messagebox.showerror("Errore", "Seleziona prima un video.")
            return

        if cv2 is None:
            messagebox.showerror("Errore", "OpenCV non disponibile per la selezione ROI.")
            return

        # Carica un frame dal video per la selezione ROI
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Errore", f"Impossibile aprire il video: {video_path}")
            return

        # Leggi un frame a metà del video
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Errore", "Impossibile leggere un frame dal video.")
            return

        # Seleziona ROI
        roi = cv2.selectROI("Seleziona ROI SX (zona battuta sinistra) - Premi SPACE o ENTER per confermare, ESC per annullare", frame, False)
        cv2.destroyAllWindows()

        if roi[2] > 0 and roi[3] > 0:  # w > 0 e h > 0
            self.roi_left = tuple(map(int, roi))
            self._update_roi_status()

    def on_select_roi_right(self):
        """Seleziona ROI per la zona battuta destra."""
        video_path = Path(self.video_path.get())
        if not video_path.exists():
            messagebox.showerror("Errore", "Seleziona prima un video.")
            return

        if cv2 is None:
            messagebox.showerror("Errore", "OpenCV non disponibile per la selezione ROI.")
            return

        # Carica un frame dal video per la selezione ROI
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            messagebox.showerror("Errore", f"Impossibile aprire il video: {video_path}")
            return

        # Leggi un frame a metà del video
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Errore", "Impossibile leggere un frame dal video.")
            return

        # Seleziona ROI
        roi = cv2.selectROI("Seleziona ROI DX (zona battuta destra) - Premi SPACE o ENTER per confermare, ESC per annullare", frame, False)
        cv2.destroyAllWindows()

        if roi[2] > 0 and roi[3] > 0:  # w > 0 e h > 0
            self.roi_right = tuple(map(int, roi))
            self._update_roi_status()

    def _update_roi_status(self):
        """Aggiorna l'etichetta dello stato ROI."""
        left_status = "✅" if self.roi_left else "❌"
        right_status = "✅" if self.roi_right else "❌"
        self.lbl_roi_status.config(text=f"ROI: SX={left_status} DX={right_status}")

    def on_analyze(self):
        # Eseguiamo in thread separato per non bloccare la GUI
        t = threading.Thread(target=self._analyze_worker, daemon=True)
        t.start()

    def _analyze_worker(self):
        video = Path(self.video_path.get())
        audio = Path(self.audio_path.get())
        outdir = Path(self.output_dir.get()) if self.output_dir.get() else None

        if not video.exists():
            messagebox.showerror("Errore", f"Video non trovato:\n{video}")
            return
        if not audio.exists():
            messagebox.showerror("Errore", f"Audio WAV non trovato:\n{audio}")
            return

        try:
            t_start = float(self.t_start.get())
            t_end = float(self.t_end.get())
        except ValueError:
            messagebox.showerror("Errore", "t_start e t_end devono essere numeri (secondi).")
            return

        try:
            fps = float(self.fps.get())
        except ValueError:
            messagebox.showerror("Errore", "fps deve essere un numero.")
            return

        try:
            lookback = float(self.lookback.get())
        except ValueError:
            lookback = 0.0

        try:
            post_roll = float(self.post_roll.get())
        except ValueError:
            post_roll = 1.5  # default

        # Calcola finestra allargata per l'analisi
        analysis_start = max(0.0, t_start - lookback)
        analysis_end = t_end

        self.log(f"=== Analisi: {video.name} | {audio.name} | {t_start}–{t_end}s @ {fps} fps ===")
        self.log(f"Finestra analisi estesa: {analysis_start:.2f}–{analysis_end:.2f}s (visibile: {t_start:.2f}–{t_end:.2f}s)")

        timeline = Timeline()

        # Audio Agent
        self.log("🎵 AudioAgent: analisi fischi...")
        audio_cfg = WhistleDetectorConfig()
        audio_agent = AudioAgent(config=audio_cfg)
        try:
            audio_events = audio_agent.load_and_analyze(audio)
            # filtro sulla finestra temporale allargata
            audio_events = [e for e in audio_events if analysis_start <= e.time <= analysis_end]
            timeline.extend(audio_events)
            self.log(f"  -> {len(audio_events)} eventi whistle nella finestra allargata.")
        except Exception as e:
            self.log(f"  !! Errore AudioAgent: {e}")

        # Motion Agent
        self.log("🎬 MotionAgent: analisi motion SX/DX...")
        motion_cfg = OpticalFlowConfig()
        motion_agent = MotionAgent(config=motion_cfg)

        if cv2 is None:
            self.log("  !! OpenCV non disponibile: MotionAgent disabilitato.")
        else:
            try:
                frames = load_video_frames(video, fps=fps, t_start=analysis_start, t_end=analysis_end)
                self.log(f"  -> Frame campionati: {len(frames)}")
                motion_events = motion_agent.run(frames, timeline=timeline)
                self.log(f"  -> {len(motion_events)} eventi motion (hit/gap).")

                # ServeAgent: rilevazione battute nella zona battuta (ROI SX/DX)
                if self.roi_left is not None or self.roi_right is not None:
                    self.log("🎯 ServeAgent: analisi battute (serve_start)...")
                    self.log("   Usa percentile motion + calibrazione referee per rilevazione precisa.")
                    try:
                        serve_cfg = ServeAgentConfig(
                            enable_logging=True,
                            log_callback=self.log,  # Reindirizza log alla GUI
                            use_referee_calibration=True,  # Abilita calibrazione referee
                        )
                        serve_agent = ServeAgent(config=serve_cfg)
                        serve_events = serve_agent.run(
                            frames,
                            roi_left=self.roi_left,
                            roi_right=self.roi_right,
                            timeline=timeline,
                        )
                        self.log(f"  -> {len(serve_events)} battute rilevate (con calibrazione referee).")
                    except Exception as e:
                        self.log(f"  !! Errore ServeAgent: {e}")
                else:
                    self.log("🎯 ServeAgent: disattivato (ROI SX/DX non definite).")

                # TouchSequenceAgent: analisi sequenze di tocchi (ricezione -> palleggio -> attacco)
                self.log("🤲 TouchSequenceAgent: analisi sequenze tocchi...")
                try:
                    touch_cfg = TouchSequenceConfig()
                    touch_agent = TouchSequenceAgent(config=touch_cfg)
                    touch_events = touch_agent.run(timeline, timeline_out=timeline)
                    self.log(f"  -> {len(touch_events)} tocchi dettagliati (reception/set/attack).")
                except Exception as e:
                    self.log(f"  !! Errore TouchSequenceAgent: {e}")

            except Exception as e:
                self.log(f"  !! Errore MotionAgent: {e}")

        # MasterCoach: analisi completa con voting multi-agente
        self.log("🧠 MasterCoach: analisi completa (voting multi-agente)...")
        self.log("   Ogni agente vota per inizio/fine rally con confidence, Coach combina i voti.")
        self.log(f"   Post-Roll: {post_roll:.1f}s (coda extra per vedere palla che cade e mini-esultanza).")
        coach_cfg = MasterCoachConfig(
            head_coach_config=HeadCoachConfig(),
            validate_rules=True,
            post_roll=post_roll,  # usa valore da GUI
            tail_min=0.7,  # minimo tempo dopo ultimo hit
        )
        coach = MasterCoach(
            cfg=coach_cfg,
            enable_logging=True,
            log_callback=self.log,  # Reindirizza log di voting alla GUI
        )

        # Timeline locale con finestra allargata per permettere al Coach di vedere eventi precedenti
        local_tl = Timeline()
        for e in timeline.sorted():
            if analysis_start <= e.time <= analysis_end:
                local_tl.events.append(e)

        # Usa MasterCoach per analisi completa con voting
        all_rallies = coach.analyze_game(local_tl)

        # Filtra i rally sulla finestra "vera" [t_start, t_end]
        visible_rallies: List[Rally] = []
        for r in all_rallies:
            # mantieni solo rally che intersecano la finestra visibile
            if r.end < t_start or r.start > t_end:
                continue
            # clampa start/end alla finestra visibile
            start = max(r.start, t_start)
            end = min(r.end, t_end)
            visible_rallies.append(Rally(start=start, end=end, side=r.side))

        self.rallies = visible_rallies

        self.log(f"  -> {len(all_rallies)} rally trovati (finestra allargata), {len(self.rallies)} visibili nella finestra [{t_start:.2f}–{t_end:.2f}s].")
        self._refresh_rally_list()

    def _refresh_rally_list(self):
        self.lst_rallies.delete(0, "end")
        for i, r in enumerate(self.rallies, start=1):
            dur = r.end - r.start
            side_str = r.side or "unknown"
            label = f"{i:02d} | {r.start:.2f}s → {r.end:.2f}s  (dur {dur:.2f}s, side={side_str})"
            self.lst_rallies.insert("end", label)

    def on_export_rallies(self):
        if not self.rallies:
            messagebox.showinfo("Info", "Nessun rally da esportare. Esegui prima l'analisi.")
            return
        outdir = Path(self.output_dir.get()) if self.output_dir.get() else None
        if not outdir:
            messagebox.showerror("Errore", "Seleziona una cartella di output.")
            return
        video = Path(self.video_path.get())
        if not video.exists():
            messagebox.showerror("Errore", f"Video non trovato:\n{video}")
            return

        # Avvia esportazione MP4 in background
        threading.Thread(target=self._export_worker, args=(video, outdir), daemon=True).start()
        
        # Chiedi anche l'esportazione JSON
        self.export_rallies_json()

    def _export_worker(self, video: Path, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        self.log(f"💾 Esportazione rally in: {outdir}")

        for idx, r in enumerate(self.rallies, start=1):
            start = r.start
            end = r.end
            dur = max(0.5, end - start)
            fn = outdir / f"rally_{idx:02d}_{r.side or 'unk'}_{start:.2f}-{end:.2f}.mp4"

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video),
                "-t",
                f"{dur:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                str(fn),
            ]
            self.log(f"  -> Export {fn.name} ({dur:.1f}s)")
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                self.log(f"    !! Errore export {fn.name}: {e}")

        self.log("✅ Esportazione completata.")

    def export_rallies_json(self):
        """Esporta i rally correnti in un file JSON (start/end/side)."""
        rallies = getattr(self, "rallies", None)
        if not rallies:
            messagebox.showwarning("Nessun rally", "Non ci sono rally da esportare. Esegui prima l'analisi.")
            return

        # Finestra di dialogo per scegliere dove salvare
        json_path = filedialog.asksaveasfilename(
            title="Salva rally come JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not json_path:
            return  # utente ha annullato

        data = [
            {
                "start": float(r.start),
                "end": float(r.end),
                "side": getattr(r, "side", "unknown"),
            }
            for r in rallies
        ]

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Esportazione completata", f"Rally salvati in\n{json_path}")
            self.log(f"💾 JSON esportato: {json_path} ({len(data)} rally)")
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante il salvataggio del JSON:\n{e}")
            self.log(f"❌ Errore esportazione JSON: {e}")


# ------------------------------
# Helper per caricare i frame
# ------------------------------
def load_video_frames(
    video_path: Path,
    fps: float = 5.0,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
) -> List[FrameSample]:
    if cv2 is None:
        raise RuntimeError("OpenCV non disponibile (cv2 == None).")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(native_fps / fps)))

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
            samples.append(FrameSample(time=cur_t, frame=frame))

        frame_idx += 1

    cap.release()
    return samples


if __name__ == "__main__":
    app = VolleyAgentsApp()
    app.mainloop()

