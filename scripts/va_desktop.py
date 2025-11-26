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
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Aggiunge la root del progetto a sys.path per evitare problemi di import quando eseguito come script
sys.path.insert(0, str(Path(__file__).parent.parent))

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from tkinter import filedialog, messagebox
import tkinter as tk

from volley_agents.core.timeline import Timeline
from volley_agents.core.rally import Rally
from volley_agents.agents.audio_agent import AudioAgent, WhistleDetectorConfig
from volley_agents.agents.motion_agent import MotionAgent, OpticalFlowConfig, FrameSample
from volley_agents.agents.serve_agent import ServeAgent, ServeAgentConfig
from volley_agents.agents.touch_sequence_agent import TouchSequenceAgent, TouchSequenceConfig
from volley_agents.agents.scoreboard_v3 import (
    ScoreboardAgentV3,
    ScoreboardConfigV3,
)
from volley_agents.fusion.head_coach import HeadCoach, HeadCoachConfig
from volley_agents.fusion.master_coach import MasterCoach, MasterCoachConfig
from volley_agents.io.config import (
    load_scoreboard_roi_config,
    save_scoreboard_roi_config,
)
from volley_agents.calibration.field_auto import FieldAutoCalibrator, FieldAutoConfig
from volley_agents.agents.ball_agent import BallAgent, BallAgentConfig
from volley_agents.agents.ball_agent_v2 import BallAgentV2, BallAgentV2Config
from volley_agents.agents.game_state_agent import GameStateAgent, GameStateAgentConfig

# =============================================================================
# HELPER: Conversione tempo mm:ss <-> secondi
# =============================================================================


def parse_time_input(time_str: str) -> float:
    """
    Converte input tempo in secondi.
    Accetta: "17:30" -> 1050s, "17:30.5" -> 1050.5s, "1050" -> 1050s.
    """
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def format_time_short(seconds: float) -> str:
    """Converte secondi in mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def format_time_precise(seconds: float) -> str:
    """Converte secondi in mm:ss.cc."""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"

try:
    import cv2
except ImportError:
    cv2 = None


class VolleyAgentsApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=15)

        self.video_path = tk.StringVar()
        self.audio_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.t_start = tk.StringVar(value="19:30")
        self.t_end = tk.StringVar(value="20:00")
        self.fps = tk.StringVar(value="5")
        self.lookback = tk.StringVar(value="8")
        self.post_roll = tk.StringVar(value="1.5")  # coda extra per vedere palla che cade

        self.rallies: List[Rally] = []
        self.roi_left: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        self.roi_right: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
        
        # ScoreboardAgentV2 configuration (GUI-based)
        self.var_use_scoreboard = tk.BooleanVar(value=False)
        self.scoreboard_main_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) - ROI principale del tabellone

        # BallAgent configuration (modello custom YOLOv10)
        self.var_use_custom_ball_model = tk.BooleanVar(value=False)
        self.var_use_ball_v2 = tk.BooleanVar(value=True)  # Default: usa ONNX V2
        self.var_use_game_state = tk.BooleanVar(value=True)  # GameStateAgent (VideoMAE)
        self.ball_model_path = tk.StringVar(value="")
        self.ball_class_id = tk.StringVar(value="0")
        self.ball_confidence = tk.StringVar(value="0.20")

        # Progress tracking
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="")

        self._build_ui()
        self.report_data = None  # Dati report dopo analisi

    # ------------------------
    # UI
    # ------------------------
    def _build_ui(self):
        self.pack(fill="both", expand=True)

        # ========== NOTEBOOK (Sistema Tab) ==========
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Tab 1: Analisi
        self.tab_analisi = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_analisi, text="🎬 Analisi")

        # Tab 2: Report
        self.tab_report = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_report, text="📊 Report")

        # Costruisci contenuto tab
        self._build_tab_analisi()
        self._build_tab_report()

    def _build_tab_analisi(self):
        """Costruisce la tab Analisi (interfaccia esistente)."""
        frm_top = ttk.Frame(self.tab_analisi, padding=10)
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
        frm_params = ttk.Labelframe(self.tab_analisi, text="Parametri analisi", padding=10)
        frm_params.pack(fill="x", padx=10, pady=5)

        ttk.Label(frm_params, text="t_start (mm:ss):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_params, textvariable=self.t_start, width=10).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(frm_params, text="t_end (mm:ss):").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm_params, textvariable=self.t_end, width=10).grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(frm_params, text="fps motion:").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm_params, textvariable=self.fps, width=5).grid(row=0, column=5, sticky="w", padx=5)

        ttk.Label(frm_params, text="lookback (s):").grid(row=0, column=6, sticky="w")
        ttk.Entry(frm_params, textvariable=self.lookback, width=5).grid(row=0, column=7, sticky="w", padx=5)

        ttk.Label(frm_params, text="Post-Roll (s):").grid(row=0, column=8, sticky="w")
        ttk.Entry(frm_params, textvariable=self.post_roll, width=5).grid(row=0, column=9, sticky="w", padx=5)

        # Pulsanti ROI
        frm_roi = ttk.Labelframe(self.tab_analisi, text="ROI Battuta (zona battuta)", padding=10)
        frm_roi.pack(fill="x", padx=10, pady=5)

        ttk.Button(frm_roi, text="ROI SX", command=self.on_select_roi_left).pack(side="left", padx=5)
        ttk.Button(frm_roi, text="ROI DX", command=self.on_select_roi_right).pack(side="left", padx=5)
        self.lbl_roi_status = ttk.Label(frm_roi, text="ROI: SX=❌ DX=❌")
        self.lbl_roi_status.pack(side="left", padx=10)

        # Sezione Tabellone (Scoreboard)
        frm_scoreboard = ttk.Labelframe(self.tab_analisi, text="Tabellone (Scoreboard)", padding=10)
        frm_scoreboard.pack(fill="x", padx=10, pady=5)

        # Checkbox per abilitare lettura tabellone
        chk_scoreboard = ttk.Checkbutton(
            frm_scoreboard,
            text="Abilita lettura tabellone",
            variable=self.var_use_scoreboard,
            command=self._on_scoreboard_checkbox_changed
        )
        chk_scoreboard.pack(side="left", padx=5)

        # Pulsante per selezionare ROI tabellone
        self.btn_scoreboard_roi = ttk.Button(
            frm_scoreboard,
            text="Seleziona tabellone",
            command=self.on_select_scoreboard_roi,
            state="disabled"
        )
        self.btn_scoreboard_roi.pack(side="left", padx=5)

        # Label di stato ROI tabellone
        self.lbl_scoreboard_status = ttk.Label(frm_scoreboard, text="ROI tabellone non configurata")
        self.lbl_scoreboard_status.pack(side="left", padx=10)

        # Sezione BallAgent (modello custom)
        frm_ball = ttk.Labelframe(self.tab_analisi, text="BallAgent - Modello Custom YOLOv10", padding=10)
        frm_ball.pack(fill="x", padx=10, pady=5)

        # Checkbox per abilitare modello custom
        chk_custom_ball = ttk.Checkbutton(
            frm_ball,
            text="Usa modello custom (YOLOv10)",
            variable=self.var_use_custom_ball_model,
            command=self._on_custom_ball_checkbox_changed
        )
        chk_custom_ball.grid(row=0, column=0, sticky="w", padx=5)

        chk_ball_v2 = ttk.Checkbutton(
            frm_ball,
            text="Usa BallAgentV2 (ONNX 88%)",
            variable=self.var_use_ball_v2,
        )
        chk_ball_v2.grid(row=0, column=1, sticky="w", padx=5)

        # Path modello
        ttk.Label(frm_ball, text="Path modello (.pt):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.entry_ball_model = ttk.Entry(frm_ball, textvariable=self.ball_model_path, width=60)
        self.entry_ball_model.grid(row=1, column=1, sticky="we", padx=5, pady=2)
        ttk.Button(frm_ball, text="Sfoglia...", command=self.on_browse_ball_model).grid(row=1, column=2, padx=5, pady=2)

        # Ball class ID
        ttk.Label(frm_ball, text="Ball class ID:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.entry_ball_class = ttk.Entry(frm_ball, textvariable=self.ball_class_id, width=10)
        self.entry_ball_class.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_ball, text="(default: 0 per YOLOv10 volleyball)").grid(row=2, column=2, sticky="w", padx=5)

        # Confidence threshold
        ttk.Label(frm_ball, text="Confidence threshold:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.entry_ball_conf = ttk.Entry(frm_ball, textvariable=self.ball_confidence, width=10)
        self.entry_ball_conf.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(frm_ball, text="(default: 0.20)").grid(row=3, column=2, sticky="w", padx=5)

        # Configura colonne per espansione
        frm_ball.columnconfigure(1, weight=1)

        # Disabilita campi inizialmente
        self._update_custom_ball_ui_state()

        # Sezione GameStateAgent
        frm_game_state = ttk.Labelframe(self.tab_analisi, text="GameStateAgent - Classificazione VideoMAE", padding=10)
        frm_game_state.pack(fill="x", padx=10, pady=5)

        chk_game_state = ttk.Checkbutton(
            frm_game_state,
            text="Abilita GameStateAgent (PLAY/NO-PLAY/SERVICE)",
            variable=self.var_use_game_state,
        )
        chk_game_state.pack(anchor="w")

        ttk.Label(
            frm_game_state,
            text="⚡ Classificazione automatica stato partita con VideoMAE (100% accuracy)",
        ).pack(anchor="w")

        # Pulsanti
        frm_buttons = ttk.Frame(self.tab_analisi, padding=10)
        frm_buttons.pack(fill="x")

        ttk.Button(
            frm_buttons,
            text="Analizza",
            command=self.on_analyze,
            bootstyle="success",
        ).pack(side="left", padx=5)
        ttk.Button(
            frm_buttons,
            text="Esporta Rally",
            command=self.on_export_rallies,
            bootstyle="info",
        ).pack(side="left", padx=5)

        # Progress bar section
        frm_progress = ttk.Frame(self.tab_analisi)
        frm_progress.pack(fill="x", padx=10, pady=5)

        self.progress_label = ttk.Label(frm_progress, textvariable=self.progress_text)
        self.progress_label.pack(anchor="w")

        self.progress_bar = ttk.Progressbar(
            frm_progress,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
            length=400,
            bootstyle="success-striped",
        )
        self.progress_bar.pack(fill="x", pady=2)

        # ========== PANNELLO PRINCIPALE LOG + RALLY ==========
        # Frame contenitore che si espande
        frm_main = ttk.Frame(self.tab_analisi)
        frm_main.pack(fill="both", expand=True, padx=10, pady=10)

        # Configura grid con 2 colonne
        frm_main.columnconfigure(0, weight=3)  # Log più largo
        frm_main.columnconfigure(1, weight=2)  # Rally
        frm_main.rowconfigure(0, weight=1)     # Si espande verticalmente

        # ========== PANNELLO LOG (sinistra) ==========
        frm_log = ttk.Labelframe(frm_main, text="📋 Log Analisi", padding=8)
        frm_log.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)

        # Configura espansione interna
        frm_log.columnconfigure(0, weight=1)
        frm_log.rowconfigure(0, weight=1)

        # Frame per Text + Scrollbars
        log_container = ttk.Frame(frm_log)
        log_container.grid(row=0, column=0, sticky="nsew")
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(0, weight=1)

        # Text widget
        self.txt_log = tk.Text(
            log_container,
            wrap="none",
            font=("Consolas", 10),
            bg="#1a1a2e",
            fg="#00ff00",
            insertbackground="white",
        )
        self.txt_log.grid(row=0, column=0, sticky="nsew")

        # Scrollbar verticale
        log_scroll_y = ttk.Scrollbar(log_container, orient="vertical", command=self.txt_log.yview)
        log_scroll_y.grid(row=0, column=1, sticky="ns")
        self.txt_log.config(yscrollcommand=log_scroll_y.set)

        # Scrollbar orizzontale
        log_scroll_x = ttk.Scrollbar(log_container, orient="horizontal", command=self.txt_log.xview)
        log_scroll_x.grid(row=1, column=0, sticky="ew")
        self.txt_log.config(xscrollcommand=log_scroll_x.set)

        # ========== PANNELLO RALLY (destra) ==========
        frm_rallies = ttk.Labelframe(frm_main, text="🏐 Rally Trovati", padding=8)
        frm_rallies.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)

        # Configura espansione interna
        frm_rallies.columnconfigure(0, weight=1)
        frm_rallies.rowconfigure(0, weight=1)

        # Frame per Listbox + Scrollbar
        rally_container = ttk.Frame(frm_rallies)
        rally_container.grid(row=0, column=0, sticky="nsew")
        rally_container.columnconfigure(0, weight=1)
        rally_container.rowconfigure(0, weight=1)

        # Listbox
        self.lst_rallies = tk.Listbox(
            rally_container,
            font=("Segoe UI", 11),
            bg="#16213e",
            fg="#ffffff",
            selectbackground="#0078d4",
            selectforeground="#ffffff",
        )
        self.lst_rallies.grid(row=0, column=0, sticky="nsew")

        # Scrollbar verticale
        rally_scroll = ttk.Scrollbar(rally_container, orient="vertical", command=self.lst_rallies.yview)
        rally_scroll.grid(row=0, column=1, sticky="ns")
        self.lst_rallies.config(yscrollcommand=rally_scroll.set)

        # Doppio click per dettagli
        self.lst_rallies.bind("<Double-Button-1>", self._on_rally_double_click)

        # Configura colori log
        self._setup_log_tags()

    def _build_tab_report(self):
        """Costruisce la tab Report con statistiche."""

        # Frame scrollabile per il report
        scroll_frame = ScrolledFrame(self.tab_report, autohide=True)
        scroll_frame.pack(fill="both", expand=True)

        self.report_container = scroll_frame

        # Header
        header_frame = ttk.Frame(scroll_frame)
        header_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(
            header_frame,
            text="📊 Report Partita",
            font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w")

        self.lbl_report_status = ttk.Label(
            header_frame,
            text="⏳ Esegui un'analisi per vedere il report",
            font=("Segoe UI", 12),
        )
        self.lbl_report_status.pack(anchor="w", pady=5)

        # Container per le statistiche (inizialmente vuoto)
        self.stats_frame = ttk.Frame(scroll_frame)
        self.stats_frame.pack(fill="both", expand=True)

        # Messaggio iniziale
        self.lbl_no_data = ttk.Label(
            self.stats_frame,
            text="Nessun dato disponibile.\n\nVai alla tab 'Analisi' ed esegui un'analisi video.",
            font=("Segoe UI", 14),
            justify="center",
        )
        self.lbl_no_data.pack(expand=True, pady=50)

        # Pulsante esporta
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(anchor="e", pady=5)

        ttk.Button(
            btn_frame,
            text="📄 Esporta HTML",
            command=self._export_html_report,
            bootstyle="info",
        ).pack(side="left", padx=5)

    def _update_report_tab(self):
        """Aggiorna la tab Report con i dati dei rally."""
        if not self.rallies:
            return

        # Pulisci stats_frame
        for widget in self.stats_frame.winfo_children():
            widget.destroy()

        # Calcola statistiche
        durations = [r.end - r.start for r in self.rallies]
        left_count = sum(1 for r in self.rallies if r.side == "left")
        right_count = sum(1 for r in self.rallies if r.side == "right")
        long_rallies = [d for d in durations if d > 15]
        quick_points = [d for d in durations if d < 5]

        # Aggiorna status
        self.lbl_report_status.config(
            text=f"✅ Analisi completata - {len(self.rallies)} rally trovati"
        )

        # ========== SEZIONE RIEPILOGO ==========
        summary_frame = ttk.Labelframe(self.stats_frame, text="📈 Riepilogo", padding=15)
        summary_frame.pack(fill="x", pady=10)

        # Grid per le statistiche principali
        stats_grid = ttk.Frame(summary_frame)
        stats_grid.pack(fill="x")

        stats = [
            ("🏐 Rally Totali", str(len(self.rallies))),
            ("⏱️ Durata Totale", f"{sum(durations):.1f}s ({sum(durations)/60:.1f} min)"),
            ("📊 Durata Media", f"{sum(durations)/len(durations):.1f}s"),
            ("🔥 Rally Più Lungo", f"{max(durations):.1f}s"),
            ("⚡ Rally Più Corto", f"{min(durations):.1f}s"),
        ]

        for i, (label, value) in enumerate(stats):
            col = i % 3
            row = i // 3

            stat_frame = ttk.Frame(stats_grid)
            stat_frame.grid(row=row, column=col, padx=20, pady=10, sticky="w")

            ttk.Label(stat_frame, text=label, font=("Segoe UI", 10)).pack(anchor="w")
            ttk.Label(stat_frame, text=value, font=("Segoe UI", 18, "bold")).pack(anchor="w")

        # ========== SEZIONE BATTUTE ==========
        serve_frame = ttk.Labelframe(self.stats_frame, text="🏐 Statistiche Battuta", padding=15)
        serve_frame.pack(fill="x", pady=10)

        serve_grid = ttk.Frame(serve_frame)
        serve_grid.pack(fill="x")

        # LEFT
        left_frame = ttk.Frame(serve_grid)
        left_frame.pack(side="left", expand=True, padx=20)
        ttk.Label(left_frame, text="⬅️ LEFT", font=("Segoe UI", 12)).pack()
        ttk.Label(left_frame, text=str(left_count), font=("Segoe UI", 36, "bold")).pack()
        ttk.Label(left_frame, text="battute", font=("Segoe UI", 10)).pack()

        # VS
        ttk.Label(serve_grid, text="vs", font=("Segoe UI", 14)).pack(side="left", padx=20)

        # RIGHT
        right_frame = ttk.Frame(serve_grid)
        right_frame.pack(side="left", expand=True, padx=20)
        ttk.Label(right_frame, text="➡️ RIGHT", font=("Segoe UI", 12)).pack()
        ttk.Label(right_frame, text=str(right_count), font=("Segoe UI", 36, "bold")).pack()
        ttk.Label(right_frame, text="battute", font=("Segoe UI", 10)).pack()

        # ========== SEZIONE EVENTI SPECIALI ==========
        events_frame = ttk.Labelframe(self.stats_frame, text="⭐ Eventi Speciali", padding=15)
        events_frame.pack(fill="x", pady=10)

        events_grid = ttk.Frame(events_frame)
        events_grid.pack(fill="x")

        ttk.Label(
            events_grid,
            text=f"🔥 Rally Lunghi (>15s): {len(long_rallies)}",
            font=("Segoe UI", 12),
        ).pack(side="left", padx=20)
        ttk.Label(
            events_grid,
            text=f"⚡ Punti Rapidi (<5s): {len(quick_points)}",
            font=("Segoe UI", 12),
        ).pack(side="left", padx=20)

        # ========== SEZIONE DETTAGLIO RALLY ==========
        detail_frame = ttk.Labelframe(self.stats_frame, text="📋 Dettaglio Rally", padding=15)
        detail_frame.pack(fill="both", expand=True, pady=10)

        # Intestazione tabella
        header = ttk.Frame(detail_frame)
        header.pack(fill="x", pady=(0, 5))

        ttk.Label(header, text="#", width=5, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Side", width=8, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Inizio", width=10, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Fine", width=10, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Durata", width=10, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(header, text="Tag", width=15, font=("Segoe UI", 10, "bold")).pack(side="left")

        ttk.Separator(detail_frame, orient="horizontal").pack(fill="x", pady=5)

        # Righe rally
        for i, r in enumerate(self.rallies, 1):
            dur = r.end - r.start
            side_icon = "⬅️" if r.side == "left" else "➡️" if r.side == "right" else "❓"

            tags = []
            if dur > 15:
                tags.append("🔥")
            if dur < 5:
                tags.append("⚡")

            start_fmt = f"{int(r.start//60)}:{int(r.start%60):02d}"
            end_fmt = f"{int(r.end//60)}:{int(r.end%60):02d}"

            row = ttk.Frame(detail_frame)
            row.pack(fill="x", pady=2)

            ttk.Label(row, text=f"#{i:02d}", width=5).pack(side="left")
            ttk.Label(row, text=side_icon, width=8).pack(side="left")
            ttk.Label(row, text=start_fmt, width=10).pack(side="left")
            ttk.Label(row, text=end_fmt, width=10).pack(side="left")
            ttk.Label(row, text=f"{dur:.1f}s", width=10).pack(side="left")
            ttk.Label(row, text=" ".join(tags) if tags else "-", width=15).pack(side="left")

        # Switch automatico alla tab Report
        self.notebook.select(self.tab_report)

    def _on_rally_double_click(self, event):
        """Mostra dettagli rally al doppio click."""
        selection = self.lst_rallies.curselection()
        if selection and self.rallies:
            idx = selection[0]
            if idx < len(self.rallies):
                r = self.rallies[idx]
                dur = r.end - r.start
                msg = (
                    f"Rally #{idx+1}\n\n"
                    f"Inizio: {format_time_precise(r.start)}\n"
                    f"Fine: {format_time_precise(r.end)}\n"
                    f"Durata: {dur:.2f}s\n"
                    f"Lato: {r.side or 'unknown'}"
                )
                messagebox.showinfo(f"Rally #{idx+1}", msg)

    def log(self, msg: str):
        """Scrive una riga nel log con colorazione per categoria."""
        self.txt_log.insert("end", msg + "\n")

        # Applica tag colori basati sul contenuto della riga appena inserita
        line_start = self.txt_log.index("end-2l linestart")
        line_end = self.txt_log.index("end-1l lineend")

        text_lower = msg.lower()
        if "✅" in msg or "completat" in text_lower:
            self.txt_log.tag_add("success", line_start, line_end)
        elif "❌" in msg or "errore" in text_lower or "!!" in msg:
            self.txt_log.tag_add("error", line_start, line_end)
        elif "⚠️" in msg or "warning" in text_lower:
            self.txt_log.tag_add("warning", line_start, line_end)
        elif "🎯" in msg or "🏐" in msg or "🎵" in msg or "🎬" in msg:
            self.txt_log.tag_add("agent", line_start, line_end)
        elif "rally" in text_lower and ("dur=" in text_lower or "|" in msg):
            self.txt_log.tag_add("rally", line_start, line_end)
        elif "===" in msg:
            self.txt_log.tag_add("header", line_start, line_end)

        self.txt_log.see("end")
        self.update_idletasks()

    def _setup_log_tags(self):
        """Configura i tag colori per il log."""
        self.txt_log.tag_configure("success", foreground="#00ff00")
        self.txt_log.tag_configure("error", foreground="#ff4444")
        self.txt_log.tag_configure("warning", foreground="#ffaa00")
        self.txt_log.tag_configure("agent", foreground="#00ccff")
        self.txt_log.tag_configure("rally", foreground="#ffff00")
        self.txt_log.tag_configure("header", foreground="#ff00ff", font=("Consolas", 10, "bold"))

    def update_progress(self, value: float, text: str = ""):
        """Aggiorna progress bar e testo."""
        self.progress_var.set(value)
        if text:
            self.progress_text.set(text)
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
            # Prova a caricare configurazione tabellone salvata per questo video
            self._load_scoreboard_config()

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

    def _on_scoreboard_checkbox_changed(self):
        """Callback quando la checkbox del tabellone cambia stato."""
        if self.var_use_scoreboard.get():
            self.btn_scoreboard_roi.config(state="normal")
        else:
            self.btn_scoreboard_roi.config(state="disabled")

    def _on_custom_ball_checkbox_changed(self):
        """Callback quando la checkbox del modello custom cambia stato."""
        self._update_custom_ball_ui_state()

    def _update_custom_ball_ui_state(self):
        """Aggiorna lo stato abilitato/disabilitato dei campi modello custom."""
        enabled = self.var_use_custom_ball_model.get()
        state = "normal" if enabled else "disabled"
        self.entry_ball_model.config(state=state)
        self.entry_ball_class.config(state=state)
        self.entry_ball_conf.config(state=state)

    def on_browse_ball_model(self):
        """Apre dialog per selezionare file modello YOLOv10 (.pt)."""
        path = filedialog.askopenfilename(
            title="Seleziona modello YOLOv10",
            filetypes=[("YOLO models", "*.pt"), ("Tutti i file", "*.*")],
        )
        if path:
            self.ball_model_path.set(path)

    def _update_scoreboard_status(self):
        """Aggiorna l'etichetta dello stato ROI tabellone."""
        if self.scoreboard_main_roi is None:
            self.lbl_scoreboard_status.config(text="ROI tabellone non configurata")
        else:
            x, y, w, h = self.scoreboard_main_roi
            self.lbl_scoreboard_status.config(text=f"ROI tabellone: x={x}, y={y}, w={w}, h={h}")

    def on_select_scoreboard_roi(self):
        """Seleziona ROI per il tabellone dal video."""
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

        # Prova a prendere un frame nell'intervallo t_start/t_end, altrimenti frame centrale
        try:
            t_start = float(self.t_start.get())
            t_end = float(self.t_end.get())
            t_target = (t_start + t_end) / 2.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_number = int(t_target * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        except (ValueError, AttributeError):
            # Fallback: frame centrale del video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Errore", "Impossibile leggere un frame dal video.")
            return

        # Mostra istruzioni in italiano
        print("Seleziona il tabellone e premi SPAZIO o ENTER per confermare, ESC per annullare")

        # Seleziona ROI
        roi = cv2.selectROI("Seleziona il tabellone - Premi SPAZIO o ENTER per confermare, ESC per annullare", frame, False)
        cv2.destroyAllWindows()

        if roi[2] > 0 and roi[3] > 0:  # w > 0 e h > 0
            self.scoreboard_main_roi = tuple(map(int, roi))
            self._update_scoreboard_status()
            print(f"ROI tabellone selezionata: {self.scoreboard_main_roi}")
            # Salva la configurazione per riutilizzo futuro
            self._save_scoreboard_config()
        else:
            print("Selezione ROI tabellone annullata")

    def _save_scoreboard_config(self):
        """Salva la configurazione ROI del tabellone in un file JSON."""
        if self.scoreboard_main_roi is None:
            return

        video_path = Path(self.video_path.get())
        if not video_path.exists():
            return

        config_path = video_path.parent / "scoreboard_config.json"

        try:
            save_scoreboard_roi_config(
                config_path,
                self.scoreboard_main_roi,
                video_path=video_path,
            )
            print(f"Configurazione tabellone salvata in: {config_path}")
        except Exception as e:
            print(f"Errore nel salvare configurazione tabellone: {e}")

    def _load_scoreboard_config(self):
        """Carica la configurazione ROI del tabellone da file JSON se disponibile."""
        video_path = Path(self.video_path.get())
        if not video_path.exists():
            return
        
        # Cerca nella stessa cartella del video
        config_path = video_path.parent / "scoreboard_config.json"
        
        if not config_path.exists():
            return

        try:
            roi = load_scoreboard_roi_config(
                config_path,
                expected_video=video_path.name,
            )
            if roi:
                self.scoreboard_main_roi = roi
                self._update_scoreboard_status()
                print(f"Configurazione tabellone caricata da: {config_path}")
        except Exception as e:
            print(f"Errore nel caricare configurazione tabellone: {e}")

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
            t_start = parse_time_input(self.t_start.get())
            t_end = parse_time_input(self.t_end.get())
        except ValueError as exc:
            self.log(f"❌ Errore formato tempo: {exc}")
            self.log("   Usa formato mm:ss (es: 17:30) oppure secondi (es: 1050).")
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

        self.log(
            f"=== Analisi: {video.name} | {audio.name} | "
            f"{format_time_short(t_start)}–{format_time_short(t_end)} @ {fps} fps ==="
        )
        self.log(
            "Finestra analisi estesa: "
            f"{format_time_precise(analysis_start)}–{format_time_precise(analysis_end)} "
            f"(visibile: {format_time_precise(t_start)}–{format_time_precise(t_end)})"
        )
        self.update_progress(0, "🎵 Analisi audio...")

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
        self.update_progress(15, "🎬 Analisi motion...")

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
                self.update_progress(30, "🎯 Analisi serve...")

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
                self.update_progress(45, "🤲 Analisi tocchi...")

                # TouchSequenceAgent: analisi sequenze di tocchi (ricezione -> palleggio -> attacco)
                self.log("🤲 TouchSequenceAgent: analisi sequenze tocchi...")
                try:
                    touch_cfg = TouchSequenceConfig()
                    touch_agent = TouchSequenceAgent(config=touch_cfg)
                    touch_events = touch_agent.run(timeline, timeline_out=timeline)
                    self.log(f"  -> {len(touch_events)} tocchi dettagliati (reception/set/attack).")
                except Exception as e:
                    self.log(f"  !! Errore TouchSequenceAgent: {e}")
                self.update_progress(55, "📊 Analisi scoreboard...")

                # ScoreboardAgentV3: lettura punteggio dal tabellone LED
                if self.var_use_scoreboard.get():
                    scoreboard_main_roi = self.scoreboard_main_roi
                    if scoreboard_main_roi is None:
                        self.log(
                            "📊 ScoreboardAgentV3: disattivato - seleziona prima la ROI del tabellone."
                        )
                    else:
                        self.log("📊 ScoreboardAgentV3: lettura punteggio con OCR...")
                        try:
                            x, y, w, h = scoreboard_main_roi
                            debug_base = (
                                Path(self.output_dir.get())
                                if self.output_dir.get()
                                else Path("tools/scoreboard")
                            )
                            debug_dir = debug_base / "debug_scoreboard"

                            scoreboard_config = ScoreboardConfigV3(
                                x=x,
                                y=y,
                                w=w,
                                h=h,
                                led_color="red",
                                use_adaptive_threshold=True,
                                history_size=10,
                                min_stable_frames=4,
                                save_debug_images=True,
                                debug_dir=str(debug_dir),
                            )

                            scoreboard_agent = ScoreboardAgentV3(
                                cfg=scoreboard_config,
                                enable_logging=True,
                                log_callback=self.log,
                            )

                            scoreboard_events = []
                            for frame_sample in frames:
                                events = scoreboard_agent.process_frame(
                                    frame_sample.frame, frame_sample.time
                                )
                                scoreboard_events.extend(events)
                                timeline.extend(events)

                            self.log(
                                f"  -> {len(scoreboard_events)} eventi SCORE_CHANGE rilevati con OCR"
                            )
                        except Exception as e:
                            self.log(f"  !! Errore ScoreboardAgentV3: {e}")
                            import traceback

                            self.log(traceback.format_exc())
                else:
                    self.log("📊 ScoreboardAgentV3: disattivato (lettura tabellone non abilitata).")
                self.update_progress(65, "📐 Calibrazione campo...")

                # ========================
                # NUOVI AGENTI CV
                # ========================

                # 1. Field Auto-Calibration (primo frame)
                field_calibrator = None
                if len(frames) > 0:
                    self.log("📐 FieldAutoCalibrator: calibrazione campo...")
                    try:
                        field_calibrator = FieldAutoCalibrator()
                        field_events = field_calibrator.calibrate(frames[0].frame)
                        timeline.extend(field_events)

                        if field_calibrator.get_homography() is not None:
                            ppm = field_calibrator.get_pixels_per_meter()
                            self.log(f"  -> Campo calibrato! PPM={ppm:.1f}" if ppm else "  -> Campo calibrato!")
                        else:
                            self.log("  -> ⚠️ Calibrazione campo fallita, uso preset")
                            field_calibrator = None
                    except Exception as e:
                        self.log(f"  !! Errore FieldAutoCalibrator: {e}")
                        import traceback
                        self.log(traceback.format_exc())
                        field_calibrator = None
                self.update_progress(70, "🏐 Ball tracking...")

                # 2. Ball Tracking
                self.log("🏐 BallAgent: tracking palla...")
                try:
                    if self.var_use_ball_v2.get():
                        # Usa BallAgentV2 (ONNX)
                        ball_cfg = BallAgentV2Config(
                            enable_logging=True,
                            log_callback=self.log,
                        )
                        self.log("  -> Usa BallAgentV2 (ONNX - 88% detection)")
                        ball_agent = BallAgentV2(config=ball_cfg)
                        ball_events = ball_agent.run(frames, timeline=timeline)
                    else:
                        # Configurazione modello custom se abilitato
                        if self.var_use_custom_ball_model.get() and self.ball_model_path.get():
                            try:
                                ball_class_id = int(self.ball_class_id.get())
                            except ValueError:
                                ball_class_id = 0
                                self.log(f"  ⚠️ Ball class ID non valido, uso default: {ball_class_id}")
                            
                            try:
                                confidence = float(self.ball_confidence.get())
                            except ValueError:
                                confidence = 0.20
                                self.log(f"  ⚠️ Confidence threshold non valido, uso default: {confidence}")
                            
                            ball_cfg = BallAgentConfig(
                                model_path=self.ball_model_path.get(),
                                ball_class_id=ball_class_id,
                                use_custom_ball_class=True,
                                confidence_threshold=confidence,
                                enable_logging=True,
                                log_callback=self.log,
                            )
                            self.log(f"  -> Usa modello custom: {self.ball_model_path.get()}")
                            self.log(f"     Class ID: {ball_class_id}, Confidence: {confidence}")
                        else:
                            # Configurazione default (COCO class 32)
                            ball_cfg = BallAgentConfig(
                                enable_logging=True,
                                log_callback=self.log,
                            )
                            self.log("  -> Usa modello default (COCO class 32)")
                        
                        ball_agent = BallAgent(config=ball_cfg)
                        ball_events = ball_agent.run(
                            frames,
                            timeline=timeline,
                            field_calibrator=field_calibrator,
                        )
                    self.log(f"  -> {len(ball_events)} eventi palla rilevati.")
                except Exception as e:
                    self.log(f"  !! Errore BallAgent: {e}")
                    import traceback
                    self.log(traceback.format_exc())
                self.update_progress(75, "🎮 GameStateAgent...")

            except Exception as e:
                self.log(f"  !! Errore MotionAgent: {e}")

        # 3. Game State Classification
        game_state_events = []
        if self.var_use_game_state.get():
            self.update_progress(80, "🎮 GameStateAgent...")
            self.log("🎮 GameStateAgent: classificazione stato partita...")
            try:
                game_state_cfg = GameStateAgentConfig(
                    window_seconds=3.0,
                    stride_seconds=2.0,
                    min_confidence=0.6,
                    enable_logging=True,
                    log_callback=self.log,
                )
                game_state_agent = GameStateAgent(config=game_state_cfg)
                game_state_events = game_state_agent.run(frames, timeline=timeline)

                # Conta stati
                play_count = sum(1 for e in game_state_events if e.extra.get("state") == "play")
                no_play_count = sum(1 for e in game_state_events if e.extra.get("state") == "no-play")
                self.log(f"  -> {len(game_state_events)} eventi (PLAY: {play_count}, NO-PLAY: {no_play_count})")
            except Exception as e:
                self.log(f"  !! Errore GameStateAgent: {e}")
                import traceback
                self.log(traceback.format_exc())
        else:
            self.log("🎮 GameStateAgent: disattivato")

        self.update_progress(85, "🧠 MasterCoach analysis...")

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

        self.update_progress(90, "🧠 MasterCoach: post-processing rally...")

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

        # Aggiorna tab Report
        self._update_report_tab()

        self.update_progress(100, "✅ Analisi completata!")

    def _refresh_rally_list(self):
        self.lst_rallies.delete(0, "end")
        for i, r in enumerate(self.rallies, start=1):
            dur = r.end - r.start
            side_str = r.side or "?"
            side_icon = "⬅️" if side_str == "left" else "➡️" if side_str == "right" else "❓"

            # Formato più leggibile con icone e tempi brevi
            label = (
                f"#{i:02d}  {side_icon}  "
                f"{format_time_short(r.start)} → {format_time_short(r.end)}  "
                f"({dur:.1f}s)"
            )
            self.lst_rallies.insert("end", label)

        # Mostra conteggio totale anche nel log
        if self.rallies:
            self.log(f"📊 Totale: {len(self.rallies)} rally trovati")

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

    def _export_html_report(self):
        """Esporta report in HTML."""
        if not self.rallies:
            messagebox.showwarning("Attenzione", "Nessun rally da esportare. Esegui prima un'analisi.")
            return

        from volley_agents.analysis.match_analyzer import MatchReport, RallyAnalysis, generate_html_report
        from datetime import datetime

        # Crea report
        report = MatchReport()
        report.video_file = Path(self.video_path.get()).name if self.video_path.get() else "Video"
        report.video_segment = f"{self.t_start.get()} - {self.t_end.get()}"
        report.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.total_rallies = len(self.rallies)

        durations = []
        for i, r in enumerate(self.rallies):
            dur = r.end - r.start
            durations.append(dur)
            ra = RallyAnalysis(
                rally_id=i + 1,
                start=r.start,
                end=r.end,
                duration=dur,
                serving_side=r.side or "unknown",
            )
            if dur > 15:
                ra.tags.append("long_rally")
            report.rallies.append(ra)

            if r.side == "left":
                report.team_left.serves += 1
            else:
                report.team_right.serves += 1

        if durations:
            report.total_duration = sum(durations)
            report.avg_rally_duration = sum(durations) / len(durations)
            report.longest_rally_duration = max(durations)
            report.shortest_rally_duration = min(durations)

        # Salva file
        html_path = filedialog.asksaveasfilename(
            title="Salva Report HTML",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
        )
        if html_path:
            html = generate_html_report(report)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            messagebox.showinfo("Successo", f"Report salvato in:\n{html_path}")

            # Apri nel browser
            import webbrowser

            webbrowser.open(html_path)


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
    root = ttk.Window(
        title="VolleyAgents Desktop v2",
        themename="superhero",  # Tema scuro moderno (alternative: darkly, cyborg, solar, vapor)
        size=(1400, 900),
        resizable=(True, True),
    )
    # Imposta una dimensione minima per mantenere il layout usabile
    root.minsize(1200, 800)
    app = VolleyAgentsApp(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

