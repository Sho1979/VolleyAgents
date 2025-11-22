from dataclasses import dataclass
from typing import Optional


@dataclass
class Rally:
    start: float
    end: float
    side: str = "unknown"
    # opzionale: possiamo aggiungere meta, ma lo lasciamo per dopo

    @property
    def duration(self):
        return max(0.0, self.end - self.start)

    def __repr__(self):
        return f"Rally(start={self.start:.2f}, end={self.end:.2f}, dur={self.duration:.2f})"


@dataclass
class RallyCandidate:
    """Intervallo grezzo proposto da un agente (es. scoreboard o audio+motion)."""

    t_start: float
    t_end: float
    source: str                     # es. "audio_motion", "scoreboard"
    meta: Optional[dict] = None


@dataclass
class RallyProposal:
    """
    Proposta di rally per un certo intervallo temporale.

    - interval: (t_start, t_end) grezzo (es. da SCORE, o da audio+motion)
    - start_candidate: tempo suggerito per l'inizio rally
    - end_candidate: tempo suggerito per la fine rally
    - source: chi ha proposto questo intervallo (es. 'score', 'audio_motion')
    - confidence: quanto si fida l'agente di questa proposta
    """

    interval_start: float
    interval_end: float
    start_candidate: Optional[float] = None
    end_candidate: Optional[float] = None
    source: str = "unknown"
    confidence: float = 0.0

