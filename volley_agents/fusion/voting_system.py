"""
VotingSystem: sistema di votazione multi-agente per decidere inizio/fine rally.

Ogni agente vota con confidence per eventi chiave, il Coach combina i voti.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from volley_agents.core.event import Event, EventType


@dataclass
class AgentVote:
    """Voto di un singolo agente."""

    agent_name: str
    event_type: EventType
    time: float
    confidence: float  # 0-1
    extra: Optional[dict] = None
    weight: float = 1.0  # peso del voto (default 1.0)


@dataclass
class VotingResult:
    """Risultato del voting combinato."""

    time: float
    confidence: float  # confidence combinata (0-1)
    votes: List[AgentVote]  # tutti i voti che contribuiscono
    consensus: bool  # True se c'è accordo tra agenti
    dominant_signal: Optional[str] = None  # quale segnale ha "vinto"


class VotingSystem:
    """
    Sistema di votazione per combinare segnali da agenti multipli.

    Ogni agente vota per eventi chiave (inizio rally, fine rally) con confidence.
    Il sistema combina i voti usando pesi e produce un risultato finale.
    """

    def __init__(self, enable_logging: bool = False, log_callback: Optional[Callable[[str], None]] = None):
        # Pesi default per ogni agente (possono essere configurati)
        self.agent_weights: Dict[str, float] = {
            "AudioAgent": 1.2,  # fischio è molto affidabile
            "ServeAgent": 1.5,  # battuta è il segnale più forte per inizio
            "MotionAgent": 0.8,  # motion è più rumoroso
            "RefereeAgent": 1.3,  # arbitro è molto affidabile
            "ScoreboardAgent": 1.4,  # tabellone è affidabile per fine
            "TouchSequenceAgent": 1.0,  # tocchi sono importanti per validazione
        }
        self.enable_logging = enable_logging
        self.log_callback = log_callback  # callback per logging esterno

    def _log(self, message: str):
        """Log interno (può essere reindirizzato a callback esterno)."""
        if self.enable_logging:
            if self.log_callback:
                self.log_callback(message)
            else:
                print(message)

    def vote_rally_start(
        self,
        events: List[Event],
        time_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[VotingResult]:
        """
        Combina voti degli agenti per determinare l'inizio di un rally.

        Segnali rilevanti:
        - WHISTLE_START (AudioAgent) - alto peso
        - SERVE_START (ServeAgent) - peso molto alto
        - REF_SERVE_READY/RELEASE (RefereeAgent) - alto peso
        - Primo HIT dopo fischio (MotionAgent) - peso medio

        Args:
            events: Lista di eventi da analizzare
            time_window: Finestra temporale opzionale (start, end)

        Returns:
            VotingResult con tempo e confidence, o None se nessun segnale
        """
        votes: List[AgentVote] = []

        # Filtra eventi nella finestra temporale
        filtered_events = events
        if time_window:
            t_start, t_end = time_window
            filtered_events = [e for e in events if t_start <= e.time <= t_end]

        # ANALISI 1: Raccogli voti per inizio rally - SOLO SERVE_START
        # Commentati: WHISTLE_START, HIT, REF_SERVE_READY/RELEASE (troppi falsi positivi)
        for event in filtered_events:
            vote = None

            # ServeAgent: SERVE_START (SOLO questo, come Analisi 1)
            # Confidence minima >= 0.60 per evitare falsi positivi
            if event.type == EventType.SERVE_START and event.confidence >= 0.60:
                vote = AgentVote(
                    agent_name="ServeAgent",
                    event_type=event.type,
                    time=event.time,
                    confidence=event.confidence,
                    extra=event.extra,
                    weight=self.agent_weights.get("ServeAgent", 1.5),
                )

            # AudioAgent: WHISTLE_START - DISABILITATO (Analisi 1)
            # if event.type == EventType.WHISTLE_START:
            #     vote = AgentVote(...)

            # RefereeAgent: REF_SERVE_READY/RELEASE - DISABILITATO (Analisi 1)
            # elif event.type in (EventType.REF_SERVE_READY, EventType.REF_SERVE_RELEASE):
            #     vote = AgentVote(...)

            # MotionAgent: HIT - DISABILITATO (Analisi 1)
            # elif event.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT):
            #     vote = AgentVote(...)

            if vote:
                votes.append(vote)

        if not votes:
            return None

        # Combina voti: media pesata
        return self._combine_votes(votes, event_type="rally_start")

    def vote_rally_end(
        self,
        events: List[Event],
        time_window: Optional[Tuple[float, float]] = None,
    ) -> Optional[VotingResult]:
        """
        Combina voti degli agenti per determinare la fine di un rally.

        Segnali rilevanti:
        - WHISTLE_END (AudioAgent) - alto peso
        - REF_POINT_LEFT/RIGHT (RefereeAgent) - peso molto alto
        - SCORE_CHANGE (ScoreboardAgent) - peso molto alto
        - MOTION_GAP lungo (MotionAgent) - peso basso

        Args:
            events: Lista di eventi da analizzare
            time_window: Finestra temporale opzionale (start, end)

        Returns:
            VotingResult con tempo e confidence, o None se nessun segnale
        """
        votes: List[AgentVote] = []

        # Filtra eventi nella finestra temporale
        filtered_events = events
        if time_window:
            t_start, t_end = time_window
            filtered_events = [e for e in events if t_start <= e.time <= t_end]

        # Raccogli voti per fine rally
        for event in filtered_events:
            vote = None

            # AudioAgent: WHISTLE_END
            if event.type == EventType.WHISTLE_END:
                vote = AgentVote(
                    agent_name="AudioAgent",
                    event_type=event.type,
                    time=event.time,
                    confidence=event.confidence,
                    extra=event.extra,
                    weight=self.agent_weights.get("AudioAgent", 1.2),
                )

            # RefereeAgent: REF_POINT_LEFT/RIGHT (segnale molto forte)
            elif event.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT):
                vote = AgentVote(
                    agent_name="RefereeAgent",
                    event_type=event.type,
                    time=event.time,
                    confidence=event.confidence,
                    extra=event.extra,
                    weight=self.agent_weights.get("RefereeAgent", 1.5),  # peso molto alto
                )

            # ScoreboardAgent: SCORE_CHANGE (segnale molto forte)
            elif event.type == EventType.SCORE_CHANGE:
                vote = AgentVote(
                    agent_name="ScoreboardAgent",
                    event_type=event.type,
                    time=event.time,
                    confidence=event.confidence,
                    extra=event.extra,
                    weight=self.agent_weights.get("ScoreboardAgent", 1.4),
                )

            # MotionAgent: MOTION_GAP molto lungo (peso basso)
            # ANALISI 1: Solo MOTION_GAP > 3.0s con priorità più bassa
            elif event.type == EventType.MOTION_GAP:
                gap_duration = event.extra.get("duration", 0) if event.extra else 0
                if gap_duration > 3.0:  # gap > 3 secondi = probabile fine rally
                    vote = AgentVote(
                        agent_name="MotionAgent",
                        event_type=event.type,
                        time=event.time,
                        confidence=event.confidence * 0.6,  # confidence ridotta
                        extra=event.extra,
                        weight=self.agent_weights.get("MotionAgent", 0.7),
                    )

            # TouchSequenceAgent: DISABILITATO (Analisi 1 - troppi falsi positivi)
            # elif event.type in (ATTACK_LEFT, ATTACK_RIGHT, TOUCH_LEFT, TOUCH_RIGHT):
            #     if event.extra and event.extra.get("is_sequence_end", False):
            #         vote = AgentVote(...)

            if vote:
                votes.append(vote)

        if not votes:
            return None

        # Combina voti: media pesata
        return self._combine_votes(votes, event_type="rally_end")

    def _combine_votes(
        self, votes: List[AgentVote], event_type: str
    ) -> VotingResult:
        """
        Combina i voti usando media pesata con normalizzazione temporale e determina consensus.

        Args:
            votes: Lista di voti da combinare
            event_type: Tipo di evento ("rally_start" o "rally_end")

        Returns:
            VotingResult con tempo combinato, confidence e consensus
        """
        if not votes:
            raise ValueError("Nessun voto da combinare")

        # Ordina per tempo (per eventi vicini nel tempo)
        votes_sorted = sorted(votes, key=lambda v: v.time)

        # NORMALIZZAZIONE TEMPORALE: aggiusta i tempi dei voti per evitare anticipazioni
        # Per inizio rally: serve_start potrebbe essere anticipato, normalizza verso il primo voto
        # Per fine rally: score_change potrebbe essere in ritardo, normalizza verso il primo segnale forte
        normalized_votes = self._normalize_vote_times(votes_sorted, event_type)

        # Log dei voti individuali
        if self.enable_logging:
            self._log_votes(normalized_votes, event_type)

        # Calcola weighted average del tempo (usa voti normalizzati)
        total_weight = sum(v.confidence * v.weight for v in normalized_votes)
        if total_weight == 0:
            weighted_time = normalized_votes[0].time
            combined_confidence = normalized_votes[0].confidence
        else:
            weighted_time = sum(v.time * v.confidence * v.weight for v in normalized_votes) / total_weight
            combined_confidence = total_weight / sum(v.weight for v in normalized_votes)
            combined_confidence = min(1.0, combined_confidence)  # clamp a 1.0

        # Determina consensus (voti vicini nel tempo e confidence alta)
        time_span = normalized_votes[-1].time - normalized_votes[0].time
        avg_confidence = sum(v.confidence for v in normalized_votes) / len(normalized_votes)
        
        # Consensus se: voti vicini (< 0.5s) e confidence alta (> 0.7)
        consensus = time_span < 0.5 and avg_confidence > 0.7

        # Determina segnale dominante (agente con peso più alto e confidence alta)
        dominant_vote = max(normalized_votes, key=lambda v: v.confidence * v.weight)
        dominant_signal = f"{dominant_vote.agent_name}:{dominant_vote.event_type.value}"

        # Log risultato finale
        if self.enable_logging:
            total_weighted = sum(v.confidence * v.weight for v in normalized_votes)
            self._log(f"VOTO FINALE {event_type.upper()} = {weighted_time:.2f}s (confidence totale = {total_weighted:.2f}, consensus = {consensus})")

        return VotingResult(
            time=weighted_time,
            confidence=combined_confidence,
            votes=normalized_votes,
            consensus=consensus,
            dominant_signal=dominant_signal,
        )

    def _normalize_vote_times(self, votes: List[AgentVote], event_type: str) -> List[AgentVote]:
        """
        Normalizza i tempi dei voti per evitare anticipazioni/ritardi.

        Per inizio rally: SERVE_START potrebbe essere anticipato, normalizza verso WHISTLE_START
        Per fine rally: SCORE_CHANGE potrebbe essere in ritardo, normalizza verso WHISTLE_END

        Args:
            votes: Lista di voti ordinata per tempo
            event_type: Tipo di evento ("rally_start" o "rally_end")

        Returns:
            Lista di voti con tempi normalizzati
        """
        if not votes:
            return votes

        normalized = []
        base_time = votes[0].time  # Primo voto come riferimento

        if event_type == "rally_start":
            # Per inizio rally: normalizza SERVE_START verso WHISTLE_START se presente
            whistle_time = None
            serve_time = None
            
            for v in votes:
                if v.event_type == EventType.WHISTLE_START:
                    whistle_time = v.time
                elif v.event_type == EventType.SERVE_START:
                    serve_time = v.time

            # Se c'è sia fischio che serve, normalizza serve verso fischio (serve potrebbe essere anticipato)
            if whistle_time is not None and serve_time is not None:
                # Serve di solito avviene 0.2-0.5s DOPO il fischio
                # Se serve è PRIMA del fischio, è un errore: normalizza
                if serve_time < whistle_time:
                    # Serve anticipato: spostalo dopo il fischio
                    correction = 0.3  # 300ms dopo fischio tipico
                    for v in votes:
                        if v.event_type == EventType.SERVE_START:
                            normalized_vote = AgentVote(
                                agent_name=v.agent_name,
                                event_type=v.event_type,
                                time=whistle_time + correction,  # Normalizzato
                                confidence=v.confidence,
                                extra=v.extra,
                                weight=v.weight,
                            )
                            normalized.append(normalized_vote)
                        else:
                            # Crea copia del voto
                            normalized_vote = AgentVote(
                                agent_name=v.agent_name,
                                event_type=v.event_type,
                                time=v.time,
                                confidence=v.confidence,
                                extra=v.extra,
                                weight=v.weight,
                            )
                            normalized.append(normalized_vote)
                    return sorted(normalized, key=lambda v: v.time)

        elif event_type == "rally_end":
            # Per fine rally: normalizza SCORE_CHANGE verso WHISTLE_END se presente
            whistle_time = None
            score_time = None

            for v in votes:
                if v.event_type == EventType.WHISTLE_END:
                    whistle_time = v.time
                elif v.event_type == EventType.SCORE_CHANGE:
                    score_time = v.time

            # Se c'è sia fischio che score, score potrebbe essere in ritardo
            if whistle_time is not None and score_time is not None:
                # Score di solito avviene quasi simultaneo al fischio
                # Se score è molto dopo (>1s), potrebbe essere normalizzato
                if score_time > whistle_time + 1.0:
                    # Score in ritardo: normalizza verso fischio
                    for v in votes:
                        if v.event_type == EventType.SCORE_CHANGE:
                            normalized_vote = AgentVote(
                                agent_name=v.agent_name,
                                event_type=v.event_type,
                                time=whistle_time + 0.2,  # Normalizzato
                                confidence=v.confidence,
                                extra=v.extra,
                                weight=v.weight,
                            )
                            normalized.append(normalized_vote)
                        else:
                            # Crea copia del voto
                            normalized_vote = AgentVote(
                                agent_name=v.agent_name,
                                event_type=v.event_type,
                                time=v.time,
                                confidence=v.confidence,
                                extra=v.extra,
                                weight=v.weight,
                            )
                            normalized.append(normalized_vote)
                    return sorted(normalized, key=lambda v: v.time)

        # Nessuna normalizzazione necessaria
        return votes

    def _log_votes(self, votes: List[AgentVote], event_type: str):
        """Log dei voti individuali per debugging."""
        if not self.enable_logging:
            return

        self._log(f"\n[{event_type.upper()}] Voti degli agenti:")
        for v in votes:
            event_name = v.event_type.value
            if v.event_type == EventType.SERVE_START:
                event_name = f"serve_start={v.time:.2f}"
            elif v.event_type == EventType.WHISTLE_START:
                event_name = f"whistle={v.time:.2f}"
            elif v.event_type == EventType.HIT_LEFT or v.event_type == EventType.HIT_RIGHT:
                event_name = f"{v.event_type.value}={v.time:.2f}"

            self._log(f"  [{v.agent_name}] voto: {event_name} conf={v.confidence:.2f} weight={v.weight:.1f}")


__all__ = [
    "VotingSystem",
    "AgentVote",
    "VotingResult",
]

