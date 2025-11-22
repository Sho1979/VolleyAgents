"""
RallyOrchestrator: sistema intelligente che decide quale agente chiamare
quando manca informazione per chiudere o continuare il rally.

Coordina gli agenti in base alle necessità del momento.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from volley_agents.core.event import Event, EventType
from volley_agents.core.rally import Rally
from volley_agents.core.timeline import Timeline


class RallyState(str, Enum):
    """Stato del rally corrente."""

    IDLE = "idle"  # Nessun rally in corso
    WAITING_START = "waiting_start"  # Aspettando inizio rally
    IN_PROGRESS = "in_progress"  # Rally in corso
    WAITING_END = "waiting_end"  # Aspettando fine rally (WHISTLE_END ma non conferma)
    NEED_CONFIRMATION = "need_confirmation"  # Bisogno conferma fine rally
    CONFIRMED_END = "confirmed_end"  # Fine confermata


class InformationNeed(str, Enum):
    """Cosa manca per prendere una decisione."""

    NONE = "none"  # Nessuna informazione mancante
    RALLY_START = "rally_start"  # Serve inizio rally
    RALLY_END = "rally_end"  # Serve fine rally
    END_CONFIRMATION = "end_confirmation"  # Serve conferma fine (palla a terra)
    SIDE_INFO = "side_info"  # Serve info lato
    SERVE_INFO = "serve_info"  # Serve info battuta


@dataclass
class AgentCapability:
    """Capacità di un agente."""

    agent_name: str
    can_detect_start: bool = False
    can_detect_end: bool = False
    can_confirm_end: bool = False  # Può confermare che palla è a terra
    can_detect_side: bool = False
    can_detect_serve: bool = False
    confidence_weight: float = 1.0  # Peso nella decisione
    response_time: float = 0.0  # Tempo tipico di risposta (secondi)


@dataclass
class OrchestrationDecision:
    """Decisione dell'orchestratore."""

    action: str  # "call_agent", "wait", "close_rally", "extend_rally"
    target_agent: Optional[str] = None  # Quale agente chiamare
    information_needed: InformationNeed = InformationNeed.NONE
    reason: str = ""  # Motivazione della decisione
    timeout: Optional[float] = None  # Timeout per la decisione


class RallyOrchestrator:
    """
    Orchestratore intelligente che decide quale agente chiamare
    quando manca informazione per chiudere o continuare il rally.
    """

    def __init__(self, enable_logging: bool = False, log_callback: Optional[Callable[[str], None]] = None):
        self.enable_logging = enable_logging
        self.log_callback = log_callback

        # Capabilities degli agenti disponibili
        self.agent_capabilities: Dict[str, AgentCapability] = {
            "AudioAgent": AgentCapability(
                agent_name="AudioAgent",
                can_detect_start=True,  # WHISTLE_START
                can_detect_end=True,  # WHISTLE_END
                confidence_weight=1.2,
                response_time=0.1,
            ),
            "ServeAgent": AgentCapability(
                agent_name="ServeAgent",
                can_detect_start=True,  # SERVE_START
                can_detect_serve=True,
                can_detect_side=True,
                confidence_weight=1.5,
                response_time=0.2,
            ),
            "MotionAgent": AgentCapability(
                agent_name="MotionAgent",
                can_detect_start=True,  # HIT dopo whistle
                can_confirm_end=True,  # MOTION_GAP lungo = palla ferma
                can_detect_side=True,  # HIT_LEFT/RIGHT
                confidence_weight=0.8,
                response_time=0.3,
            ),
            "RefereeAgent": AgentCapability(
                agent_name="RefereeAgent",
                can_detect_start=True,  # REF_SERVE_READY/RELEASE
                can_confirm_end=True,  # REF_POINT_LEFT/RIGHT = punto assegnato
                can_detect_serve=True,
                confidence_weight=1.3,
                response_time=0.5,
            ),
            "ScoreboardAgent": AgentCapability(
                agent_name="ScoreboardAgent",
                can_confirm_end=True,  # SCORE_CHANGE = punto assegnato
                confidence_weight=1.4,
                response_time=1.0,  # OCR può essere lento
            ),
            "TouchSequenceAgent": AgentCapability(
                agent_name="TouchSequenceAgent",
                can_detect_side=True,  # Da sequenza tocchi
                confidence_weight=1.0,
                response_time=0.4,
            ),
        }

        # Stato corrente
        self.current_state: RallyState = RallyState.IDLE
        self.current_rally: Optional[Rally] = None
        self.last_event_time: float = 0.0
        self.waiting_since: Optional[float] = None

    def _log(self, message: str):
        """Log interno."""
        if self.enable_logging:
            if self.log_callback:
                self.log_callback(message)
            else:
                print(message)

    def analyze_rally_status(
        self,
        rally: Rally,
        timeline: Timeline,
        current_time: Optional[float] = None,
    ) -> OrchestrationDecision:
        """
        Analizza lo stato del rally e decide quale azione intraprendere.

        Args:
            rally: Rally corrente da analizzare
            timeline: Timeline con tutti gli eventi
            current_time: Tempo corrente (default: ultimo evento timeline)

        Returns:
            OrchestrationDecision con azione da intraprendere
        """
        events = timeline.sorted()
        if not events:
            return OrchestrationDecision(
                action="wait",
                information_needed=InformationNeed.RALLY_START,
                reason="Nessun evento nella timeline",
            )

        if current_time is None:
            current_time = events[-1].time if events else rally.end

        self.current_rally = rally
        self.last_event_time = current_time

        # Determina cosa manca
        info_needed = self._identify_missing_information(rally, events, current_time)

        # Decide quale agente chiamare
        return self._decide_action(rally, events, info_needed, current_time)

    def _identify_missing_information(
        self, rally: Rally, events: List[Event], current_time: float
    ) -> InformationNeed:
        """
        Identifica quale informazione manca per prendere una decisione.

        Args:
            rally: Rally corrente
            events: Lista di eventi nella timeline
            current_time: Tempo corrente

        Returns:
            InformationNeed: Cosa manca
        """
        # Filtra eventi nel rally
        rally_events = [e for e in events if rally.start <= e.time <= current_time]

        # Verifica se abbiamo inizio rally
        has_serve_start = any(e.type == EventType.SERVE_START for e in rally_events)
        has_whistle_start = any(e.type == EventType.WHISTLE_START for e in rally_events)
        if not has_serve_start and not has_whistle_start:
            return InformationNeed.RALLY_START

        # Verifica se abbiamo fine rally confermata
        has_score = any(e.type == EventType.SCORE_CHANGE for e in rally_events)
        has_ref_point = any(
            e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
            for e in rally_events
        )
        has_long_gap = any(
            e.type == EventType.MOTION_GAP
            and e.extra
            and e.extra.get("duration", 0) > 3.0
            for e in rally_events
        )
        has_whistle_end = any(e.type == EventType.WHISTLE_END for e in rally_events)

        # Se abbiamo WHISTLE_END ma non conferma, serve conferma
        if has_whistle_end and not (has_score or has_ref_point or has_long_gap):
            # Verifica se è passato abbastanza tempo da WHISTLE_END
            whistle_end_time = max(
                (e.time for e in rally_events if e.type == EventType.WHISTLE_END),
                default=0.0,
            )
            time_since_whistle = current_time - whistle_end_time
            if time_since_whistle > 1.0:  # Aspettato 1s dopo whistle
                return InformationNeed.END_CONFIRMATION

        # Se non abbiamo fine confermata e siamo oltre durata minima
        duration = current_time - rally.start
        if duration > 3.5 and not (has_score or has_ref_point or has_long_gap):
            if has_whistle_end:
                return InformationNeed.END_CONFIRMATION
            else:
                return InformationNeed.RALLY_END

        return InformationNeed.NONE

    def _decide_action(
        self,
        rally: Rally,
        events: List[Event],
        info_needed: InformationNeed,
        current_time: float,
    ) -> OrchestrationDecision:
        """
        Decide quale azione intraprendere basandosi su cosa manca.

        Args:
            rally: Rally corrente
            events: Lista di eventi
            info_needed: Cosa manca
            current_time: Tempo corrente

        Returns:
            OrchestrationDecision con azione da intraprendere
        """
        if info_needed == InformationNeed.NONE:
            return OrchestrationDecision(
                action="wait",
                information_needed=InformationNeed.NONE,
                reason="Tutte le informazioni disponibili",
            )

        # Filtra eventi nel rally
        rally_events = [e for e in events if rally.start <= e.time <= current_time]

        if info_needed == InformationNeed.RALLY_START:
            # Serve inizio rally: priorità ServeAgent > AudioAgent > MotionAgent
            return OrchestrationDecision(
                action="call_agent",
                target_agent="ServeAgent",
                information_needed=InformationNeed.RALLY_START,
                reason="Serve inizio rally: ServeAgent ha priorità per SERVE_START",
            )

        elif info_needed == InformationNeed.END_CONFIRMATION:
            # Serve conferma fine: priorità ScoreboardAgent > RefereeAgent > MotionAgent
            # Controlla cosa abbiamo già
            has_score = any(e.type == EventType.SCORE_CHANGE for e in rally_events)
            has_ref_point = any(
                e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
                for e in rally_events
            )
            has_long_gap = any(
                e.type == EventType.MOTION_GAP
                and e.extra
                and e.extra.get("duration", 0) > 3.0
                for e in rally_events
            )

            # Se non abbiamo niente, chiama agenti in ordine di priorità
            if not has_score:
                return OrchestrationDecision(
                    action="call_agent",
                    target_agent="ScoreboardAgent",
                    information_needed=InformationNeed.END_CONFIRMATION,
                    reason="Serve conferma fine: ScoreboardAgent può confermare con SCORE_CHANGE",
                    timeout=2.0,
                )
            elif not has_ref_point:
                return OrchestrationDecision(
                    action="call_agent",
                    target_agent="RefereeAgent",
                    information_needed=InformationNeed.END_CONFIRMATION,
                    reason="Serve conferma fine: RefereeAgent può confermare con REF_POINT",
                    timeout=1.5,
                )
            elif not has_long_gap:
                return OrchestrationDecision(
                    action="call_agent",
                    target_agent="MotionAgent",
                    information_needed=InformationNeed.END_CONFIRMATION,
                    reason="Serve conferma fine: MotionAgent può confermare con MOTION_GAP lungo",
                    timeout=1.0,
                )

        elif info_needed == InformationNeed.RALLY_END:
            # Serve fine rally: priorità AudioAgent > RefereeAgent
            return OrchestrationDecision(
                action="call_agent",
                target_agent="AudioAgent",
                information_needed=InformationNeed.RALLY_END,
                reason="Serve fine rally: AudioAgent può rilevare WHISTLE_END",
                timeout=1.0,
            )

        # Default: aspetta
        return OrchestrationDecision(
            action="wait",
            information_needed=info_needed,
            reason="Aspettando informazioni dagli agenti",
        )

    def should_extend_rally(
        self,
        rally: Rally,
        timeline: Timeline,
        window_size: float = 5.0,
    ) -> Tuple[bool, Optional[float]]:
        """
        Determina se il rally dovrebbe essere esteso e fino a quando.

        Args:
            rally: Rally corrente
            timeline: Timeline con tutti gli eventi
            window_size: Finestra temporale dopo rally.end per cercare conferme

        Returns:
            Tuple (should_extend, new_end_time)
        """
        events = timeline.sorted()
        rally_events = [e for e in events if rally.start <= e.time <= rally.end + window_size]

        # Cerca segnali forti di fine dopo rally.end
        after_rally = [e for e in rally_events if e.time > rally.end]

        # Priorità 1: SCORE_CHANGE
        score_events = [e for e in after_rally if e.type == EventType.SCORE_CHANGE]
        if score_events:
            new_end = score_events[0].time
            self._log(
                f"🎯 Rally esteso fino a SCORE_CHANGE: [{rally.start:.2f}-{rally.end:.2f}s] -> "
                f"[{rally.start:.2f}-{new_end:.2f}s]"
            )
            return True, new_end

        # Priorità 2: REF_POINT
        ref_events = [
            e for e in after_rally
            if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
        ]
        if ref_events:
            new_end = ref_events[0].time
            self._log(
                f"🎯 Rally esteso fino a REF_POINT: [{rally.start:.2f}-{rally.end:.2f}s] -> "
                f"[{rally.start:.2f}-{new_end:.2f}s]"
            )
            return True, new_end

        # Priorità 3: MOTION_GAP lungo
        gap_events = [
            e for e in after_rally
            if e.type == EventType.MOTION_GAP
            and e.extra
            and e.extra.get("duration", 0) > 3.0
        ]
        if gap_events:
            new_end = gap_events[0].time
            self._log(
                f"🎯 Rally esteso fino a MOTION_GAP lungo: [{rally.start:.2f}-{rally.end:.2f}s] -> "
                f"[{rally.start:.2f}-{new_end:.2f}s]"
            )
            return True, new_end

        return False, None

    def get_best_agent_for(self, information_need: InformationNeed) -> Optional[str]:
        """
        Restituisce l'agente migliore per soddisfare una necessità informativa.

        Args:
            information_need: Cosa serve

        Returns:
            Nome dell'agente migliore o None
        """
        if information_need == InformationNeed.RALLY_START:
            # Priorità: ServeAgent > AudioAgent > MotionAgent
            if self.agent_capabilities.get("ServeAgent"):
                return "ServeAgent"
            if self.agent_capabilities.get("AudioAgent"):
                return "AudioAgent"
            if self.agent_capabilities.get("MotionAgent"):
                return "MotionAgent"

        elif information_need == InformationNeed.END_CONFIRMATION:
            # Priorità: ScoreboardAgent > RefereeAgent > MotionAgent
            if self.agent_capabilities.get("ScoreboardAgent"):
                return "ScoreboardAgent"
            if self.agent_capabilities.get("RefereeAgent"):
                return "RefereeAgent"
            if self.agent_capabilities.get("MotionAgent"):
                return "MotionAgent"

        elif information_need == InformationNeed.RALLY_END:
            # Priorità: AudioAgent > RefereeAgent
            if self.agent_capabilities.get("AudioAgent"):
                return "AudioAgent"
            if self.agent_capabilities.get("RefereeAgent"):
                return "RefereeAgent"

        return None


__all__ = [
    "RallyOrchestrator",
    "RallyState",
    "InformationNeed",
    "OrchestrationDecision",
    "AgentCapability",
]

