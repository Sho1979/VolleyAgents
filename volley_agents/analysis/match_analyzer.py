"""MatchAnalyzer: Coordinatore centrale analisi partita.

Integra tutti gli agenti:
- AudioAgent: fischi
- MotionAgent: movimento
- ServeAgent: battute
- BallAgentV2: tracking palla
- GameStateAgent: stato gioco
- ActionAgent: azioni (spike, block, set, etc.)
- MasterCoach / RallyDetector: detection rally

Genera MatchReport con:
- Statistiche rally
- Statistiche azioni per squadra
- Sequenze di gioco
- Timeline eventi
- Analisi rotazioni (TODO)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import json


@dataclass
class TeamStats:
    """Statistiche squadra."""

    name: str = "unknown"
    side: str = "unknown"  # left/right

    # Rally stats
    rallies_won: int = 0
    rallies_lost: int = 0

    # Azioni offensive
    spikes: int = 0
    spikes_success: int = 0  # Punti diretti
    spikes_blocked: int = 0
    spikes_error: int = 0

    # Azioni difensive
    blocks: int = 0
    blocks_point: int = 0  # Muri punto
    blocks_touch: int = 0  # Muri tocco

    # Servizio
    serves: int = 0
    aces: int = 0
    serve_errors: int = 0

    # Ricezione
    receptions: int = 0
    receptions_perfect: int = 0  # ++
    receptions_good: int = 0  # +
    receptions_bad: int = 0  # -

    # Palleggio
    sets: int = 0

    # Difesa
    digs: int = 0

    @property
    def spike_efficiency(self) -> float:
        """Efficienza attacco: (punti - errori) / totale."""
        if self.spikes == 0:
            return 0.0
        return float(self.spikes_success - self.spikes_error) / float(self.spikes)

    @property
    def reception_efficiency(self) -> float:
        """Efficienza ricezione: (perfette + buone) / totale."""
        if self.receptions == 0:
            return 0.0
        return float(self.receptions_perfect + self.receptions_good) / float(
            self.receptions
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "side": self.side,
            "rallies_won": self.rallies_won,
            "rallies_lost": self.rallies_lost,
            "attack": {
                "spikes": self.spikes,
                "success": self.spikes_success,
                "blocked": self.spikes_blocked,
                "errors": self.spikes_error,
                "efficiency": f"{self.spike_efficiency:.1%}",
            },
            "block": {
                "total": self.blocks,
                "points": self.blocks_point,
                "touches": self.blocks_touch,
            },
            "serve": {
                "total": self.serves,
                "aces": self.aces,
                "errors": self.serve_errors,
            },
            "reception": {
                "total": self.receptions,
                "perfect": self.receptions_perfect,
                "good": self.receptions_good,
                "bad": self.receptions_bad,
                "efficiency": f"{self.reception_efficiency:.1%}",
            },
            "set": self.sets,
            "dig": self.digs,
        }


@dataclass
class RallyAnalysis:
    """Analisi dettagliata di un singolo rally."""

    rally_id: int
    start: float
    end: float
    duration: float
    serving_side: str
    winning_side: str = "unknown"

    # Azioni nel rally
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Sequenza tocchi
    touch_sequence: List[str] = field(default_factory=list)

    # Pattern rilevato
    pattern: str = "unknown"  # es: "receive-set-spike", "ace", "error"

    # Punteggio dopo rally
    score_left: int = 0
    score_right: int = 0

    # Tag speciali
    tags: List[str] = field(default_factory=list)  # ["ace", "block_point", "long_rally", etc.]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rally_id": self.rally_id,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "serving_side": self.serving_side,
            "winning_side": self.winning_side,
            "actions_count": len(self.actions),
            "touch_sequence": self.touch_sequence,
            "pattern": self.pattern,
            "score": f"{self.score_left}-{self.score_right}",
            "tags": self.tags,
        }


@dataclass
class MatchReport:
    """Report completo della partita."""

    # Metadata
    video_file: str = ""
    analysis_date: str = ""
    analysis_duration_seconds: float = 0.0
    video_segment: str = ""  # es: "16:58 - 22:00"

    # Statistiche generali
    total_rallies: int = 0
    total_duration: float = 0.0
    avg_rally_duration: float = 0.0
    longest_rally_duration: float = 0.0
    shortest_rally_duration: float = 0.0

    # Statistiche squadre
    team_left: TeamStats = field(default_factory=lambda: TeamStats(side="left"))
    team_right: TeamStats = field(default_factory=lambda: TeamStats(side="right"))

    # Analisi rally
    rallies: List[RallyAnalysis] = field(default_factory=list)

    # Eventi speciali
    aces: List[Dict[str, Any]] = field(default_factory=list)
    block_points: List[Dict[str, Any]] = field(default_factory=list)
    long_rallies: List[Dict[str, Any]] = field(default_factory=list)  # > 15s

    # Timeline completa
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Detection quality
    ball_detection_rate: float = 0.0
    whistle_count: int = 0
    serve_count: int = 0
    action_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "video_file": self.video_file,
                "analysis_date": self.analysis_date,
                "analysis_duration_seconds": self.analysis_duration_seconds,
                "video_segment": self.video_segment,
            },
            "summary": {
                "total_rallies": self.total_rallies,
                "total_duration": f"{self.total_duration:.1f}s",
                "avg_rally_duration": f"{self.avg_rally_duration:.1f}s",
                "longest_rally": f"{self.longest_rally_duration:.1f}s",
                "shortest_rally": f"{self.shortest_rally_duration:.1f}s",
            },
            "teams": {
                "left": self.team_left.to_dict(),
                "right": self.team_right.to_dict(),
            },
            "rallies": [r.to_dict() for r in self.rallies],
            "special_events": {
                "aces": len(self.aces),
                "block_points": len(self.block_points),
                "long_rallies": len(self.long_rallies),
            },
            "detection_quality": {
                "ball_detection_rate": f"{self.ball_detection_rate:.1%}",
                "whistle_count": self.whistle_count,
                "serve_count": self.serve_count,
                "action_count": self.action_count,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def generate_summary_text(self) -> str:
        """Genera riassunto testuale."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("📊 MATCH ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"📹 Video: {self.video_file}")
        lines.append(f"⏱️ Segmento: {self.video_segment}")
        lines.append(f"📅 Analisi: {self.analysis_date}")
        lines.append("")
        lines.append("─" * 60)
        lines.append("📈 RIEPILOGO RALLY")
        lines.append("─" * 60)
        lines.append(f"  Rally totali: {self.total_rallies}")
        lines.append(f"  Durata totale: {self.total_duration:.1f}s")
        lines.append(f"  Durata media: {self.avg_rally_duration:.1f}s")
        lines.append(f"  Rally più lungo: {self.longest_rally_duration:.1f}s")
        lines.append(f"  Rally più corto: {self.shortest_rally_duration:.1f}s")
        lines.append("")

        lines.append("─" * 60)
        lines.append("🏐 STATISTICHE SQUADRE")
        lines.append("─" * 60)

        for team in [self.team_left, self.team_right]:
            lines.append(f"\n  {'⬅️ LEFT' if team.side == 'left' else '➡️ RIGHT'} ({team.name})")
            lines.append(f"    Rally vinti: {team.rallies_won}")
            lines.append(f"    Attacchi: {team.spikes} (eff: {team.spike_efficiency:.0%})")
            lines.append(f"    Muri: {team.blocks} (punti: {team.blocks_point})")
            lines.append(f"    Ace: {team.aces}")
            lines.append(
                f"    Ricezioni: {team.receptions} "
                f"(eff: {team.reception_efficiency:.0%})"
            )

        lines.append("")
        lines.append("─" * 60)
        lines.append("⭐ EVENTI SPECIALI")
        lines.append("─" * 60)
        lines.append(f"  Ace: {len(self.aces)}")
        lines.append(f"  Muri punto: {len(self.block_points)}")
        lines.append(f"  Rally lunghi (>15s): {len(self.long_rallies)}")
        lines.append("")

        lines.append("─" * 60)
        lines.append("🔍 QUALITÀ DETECTION")
        lines.append("─" * 60)
        lines.append(f"  Ball detection: {self.ball_detection_rate:.1%}")
        lines.append(f"  Fischi rilevati: {self.whistle_count}")
        lines.append(f"  Battute rilevate: {self.serve_count}")
        lines.append(f"  Azioni rilevate: {self.action_count}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class MatchAnalyzer:
    """
    Coordinatore centrale per analisi partita.

    Orchestro tutti gli agenti e genera MatchReport.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.report = MatchReport()

    def analyze(
        self,
        rallies: List[Any],  # Rally dal MasterCoach o RallyDetector
        whistle_events: Optional[List[Any]] = None,
        serve_events: Optional[List[Any]] = None,
        ball_events: Optional[List[Any]] = None,
        action_events: Optional[List[Any]] = None,
        game_state_events: Optional[List[Any]] = None,  # noqa: ARG002  (per futuro)
        video_file: str = "",
        video_segment: str = "",
        total_frames: int = 0,
    ) -> MatchReport:
        """
        Analizza tutti i dati e genera report.

        Args:
            rallies: Lista rally rilevati
            whistle_events: Eventi fischio
            serve_events: Eventi battuta
            ball_events: Eventi ball tracking
            action_events: Eventi azione (spike, block, etc.)
            game_state_events: Eventi stato gioco
            video_file: Nome file video
            video_segment: Segmento temporale
            total_frames: Numero frame totali
        """
        self.report = MatchReport()
        self.report.video_file = video_file
        self.report.video_segment = video_segment
        self.report.analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Detection quality
        if whistle_events:
            self.report.whistle_count = len(whistle_events)
        if serve_events:
            self.report.serve_count = len(serve_events)
        if action_events:
            self.report.action_count = len(action_events)
        if ball_events and total_frames > 0:
            detected = sum(
                1
                for e in ball_events
                if hasattr(e, "type") and "DETECTED" in str(e.type).upper()
            )
            self.report.ball_detection_rate = detected / float(total_frames)

        # Analizza rally
        self.report.total_rallies = len(rallies)

        if rallies:
            durations: List[float] = []
            for i, rally in enumerate(rallies):
                # Estrai start/end
                if hasattr(rally, "start"):
                    start = float(rally.start)
                    end = float(rally.end)
                    side = getattr(rally, "side", "unknown")
                elif isinstance(rally, dict):
                    start = float(rally.get("start", 0.0))
                    end = float(rally.get("end", 0.0))
                    side = rally.get("side", "unknown")
                else:
                    continue

                duration = end - start
                durations.append(duration)

                # Crea RallyAnalysis
                rally_analysis = RallyAnalysis(
                    rally_id=i + 1,
                    start=start,
                    end=end,
                    duration=duration,
                    serving_side=side,
                )

                # Tag speciali
                if duration > 15.0:
                    rally_analysis.tags.append("long_rally")
                    self.report.long_rallies.append(
                        {
                            "rally_id": i + 1,
                            "duration": duration,
                            "start": start,
                        }
                    )

                # Aggiungi azioni nel rally
                if action_events:
                    rally_actions = self._get_events_in_range(action_events, start, end)
                    for a in rally_actions:
                        action_dict = {
                            "time": getattr(a, "time", 0.0),
                            "type": getattr(a, "action_type", "unknown"),
                            "side": getattr(a, "side", "unknown"),
                        }
                            # NB: action_type potrebbe essere Enum, convertilo a stringa
                        atype = action_dict["type"]
                        if hasattr(atype, "value"):
                            action_dict["type"] = atype.value
                        rally_analysis.actions.append(action_dict)
                        rally_analysis.touch_sequence.append(str(action_dict["type"]))

                # Aggiorna stats squadra (per ora solo serve count basato su side iniziale)
                if side == "left":
                    self.report.team_left.serves += 1
                elif side == "right":
                    self.report.team_right.serves += 1

                self.report.rallies.append(rally_analysis)

            # Calcola statistiche durata
            if durations:
                self.report.total_duration = sum(durations)
                self.report.avg_rally_duration = self.report.total_duration / float(
                    len(durations)
                )
                self.report.longest_rally_duration = max(durations)
                self.report.shortest_rally_duration = min(durations)

        # Analizza azioni per squadra
        if action_events:
            self._analyze_team_actions(action_events)

        # Analizza ace
        if serve_events:
            self._detect_aces(serve_events, whistle_events or [], rallies)

        return self.report

    def _get_events_in_range(
        self,
        events: List[Any],
        start: float,
        end: float,
    ) -> List[Any]:
        """Filtra eventi in un range temporale."""
        result: List[Any] = []
        for e in events:
            t = e.time if hasattr(e, "time") else float(e.get("time", 0.0))
            if start <= t <= end:
                result.append(e)
        return result

    def _analyze_team_actions(self, action_events: List[Any]) -> None:
        """Analizza azioni per squadra."""
        for a in action_events:
            side = getattr(a, "side", "unknown")
            action_type = getattr(a, "action_type", None)

            if action_type is None:
                continue

            team = self.report.team_left if side == "left" else self.report.team_right

            atype = action_type.value if hasattr(action_type, "value") else str(action_type)

            if atype == "spike":
                team.spikes += 1
            elif atype == "block":
                team.blocks += 1
            elif atype == "receive":
                team.receptions += 1
            elif atype == "set":
                team.sets += 1
            elif atype == "dig":
                team.digs += 1
            elif atype == "serve":
                # Servizio già contato altrove
                team.serves += 1

    def _detect_aces(
        self,
        serve_events: List[Any],
        whistle_events: List[Any],
        rallies: List[Any],
    ) -> None:
        """Rileva ace basandosi su serve + whistle veloce."""
        if not whistle_events:
            return

        whistle_times: List[float] = []
        for e in whistle_events:
            t = e.time if hasattr(e, "time") else float(e.get("time", 0.0))
            whistle_times.append(t)
        whistle_times.sort()

        for s in serve_events:
            serve_time = s.time if hasattr(s, "time") else float(s.get("time", 0.0))
            serve_side: Optional[str] = None
            if hasattr(s, "extra") and isinstance(s.extra, dict):
                serve_side = s.extra.get("side")

            # Cerca fischio entro 4s dal serve (potenziale ace)
            for wt in whistle_times:
                if serve_time < wt < serve_time + 4.0:
                    # Potenziale ace - verifica che non ci siano altri serve nel mezzo
                    is_ace = True
                    for s2 in serve_events:
                        s2_time = (
                            s2.time if hasattr(s2, "time") else float(s2.get("time", 0.0))
                        )
                        if serve_time < s2_time < wt:
                            is_ace = False
                            break

                    if is_ace:
                        ace_info = {
                            "serve_time": serve_time,
                            "whistle_time": wt,
                            "duration": wt - serve_time,
                            "side": serve_side,
                        }
                        self.report.aces.append(ace_info)

                        # Aggiorna stats squadra
                        if serve_side == "left":
                            self.report.team_left.aces += 1
                        elif serve_side == "right":
                            self.report.team_right.aces += 1
                    break


def generate_html_report(report: MatchReport) -> str:
    """Genera report HTML interattivo."""

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.7; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }}
        .stat-card:hover {{ transform: translateY(-5px); }}
        .stat-card h3 {{ font-size: 2em; color: #00d9ff; }}
        .stat-card p {{ opacity: 0.7; margin-top: 5px; }}

        .teams-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .team-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
        }}
        .team-card.left {{ border-left: 4px solid #ff6b6b; }}
        .team-card.right {{ border-left: 4px solid #4ecdc4; }}
        .team-card h2 {{ margin-bottom: 20px; }}
        .team-stat {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .team-stat:last-child {{ border-bottom: none; }}

        .rally-list {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
        }}
        .rally-item {{
            display: grid;
            grid-template-columns: 60px 1fr 100px 100px;
            gap: 15px;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            margin-bottom: 10px;
            align-items: center;
        }}
        .rally-id {{
            font-weight: bold;
            font-size: 1.2em;
            color: #00d9ff;
        }}
        .rally-time {{ opacity: 0.7; font-size: 0.9em; }}
        .rally-duration {{ text-align: center; }}
        .tag {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            margin: 2px;
        }}
        .tag.ace {{ background: #ff6b6b; }}
        .tag.long {{ background: #4ecdc4; }}
        .tag.block {{ background: #ffe66d; color: #333; }}

        .footer {{
            text-align: center;
            padding: 20px;
            opacity: 0.5;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏐 Match Analysis Report</h1>
            <p>{report.video_file}</p>
            <p>Segmento: {report.video_segment} | Analisi: {report.analysis_date}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>{report.total_rallies}</h3>
                <p>Rally Totali</p>
            </div>
            <div class="stat-card">
                <h3>{report.avg_rally_duration:.1f}s</h3>
                <p>Durata Media</p>
            </div>
            <div class="stat-card">
                <h3>{len(report.aces)}</h3>
                <p>Ace</p>
            </div>
            <div class="stat-card">
                <h3>{len(report.long_rallies)}</h3>
                <p>Rally Lunghi (&gt;15s)</p>
            </div>
        </div>

        <div class="teams-container">
            <div class="team-card left">
                <h2>⬅️ {report.team_left.name or 'Squadra Sinistra'}</h2>
                <div class="team-stat"><span>Battute</span><span>{report.team_left.serves}</span></div>
                <div class="team-stat"><span>Ace</span><span>{report.team_left.aces}</span></div>
                <div class="team-stat"><span>Attacchi</span><span>{report.team_left.spikes}</span></div>
                <div class="team-stat"><span>Muri</span><span>{report.team_left.blocks}</span></div>
                <div class="team-stat"><span>Ricezioni</span><span>{report.team_left.receptions}</span></div>
            </div>
            <div class="team-card right">
                <h2>➡️ {report.team_right.name or 'Squadra Destra'}</h2>
                <div class="team-stat"><span>Battute</span><span>{report.team_right.serves}</span></div>
                <div class="team-stat"><span>Ace</span><span>{report.team_right.aces}</span></div>
                <div class="team-stat"><span>Attacchi</span><span>{report.team_right.spikes}</span></div>
                <div class="team-stat"><span>Muri</span><span>{report.team_right.blocks}</span></div>
                <div class="team-stat"><span>Ricezioni</span><span>{report.team_right.receptions}</span></div>
            </div>
        </div>

        <div class="rally-list">
            <h2 style="margin-bottom: 20px;">📋 Dettaglio Rally</h2>
            {"".join([f'''
            <div class="rally-item">
                <div class="rally-id">#{r.rally_id}</div>
                <div>
                    <div class="rally-time">{int(r.start//60)}:{int(r.start%60):02d} - {int(r.end//60)}:{int(r.end%60):02d}</div>
                    <div>Serve: {r.serving_side}</div>
                </div>
                <div class="rally-duration">{r.duration:.1f}s</div>
                <div>
                    {"".join([f'<span class="tag {"ace" if "ace" in t else "long" if "long" in t else "block"}'>{t}</span>' for t in r.tags]) or '-'}
                </div>
            </div>
            ''' for r in report.rallies])}
        </div>

        <div class="footer">
            <p>Generato da VolleyAgents - {report.analysis_date}</p>
            <p>Ball Detection: {report.ball_detection_rate:.1%} | Fischi: {report.whistle_count} | Battute: {report.serve_count}</p>
        </div>
    </div>
</body>
</html>"""

    return html


__all__ = ["MatchAnalyzer", "MatchReport", "RallyAnalysis", "TeamStats", "generate_html_report"]


