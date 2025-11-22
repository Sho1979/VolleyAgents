"""
TouchSequenceAgent: analizza sequenze di tocchi per validare rally secondo regole FIPAV.

Responsabilità:
- Analizza sequenze di tocchi per ogni lato
- Classifica tocchi (ricezione, palleggio, attacco)
- Valida secondo regole FIPAV (max 3 tocchi, timing, ecc.)
- Pubblica eventi TOUCH/SET/ATTACK dettagliati
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

from volley_agents.core.event import Event, EventType
from volley_agents.fusion.rules_fipav import (
    MAX_TOUCHES_PER_SIDE,
    MIN_TIME_BETWEEN_TOUCHES,
    MAX_TIME_BETWEEN_TOUCHES,
    classify_touch_by_position,
    validate_rally_touches,
    infer_touch_type_from_motion,
)

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline


@dataclass
class TouchSequenceConfig:
    """Configurazione per TouchSequenceAgent."""

    min_time_between_sequences: float = 0.5  # minimo tempo tra sequenze diverse
    max_sequence_duration: float = 3.0  # massimo durata sequenza (3s)
    set_confidence_threshold: float = 0.7  # confidence minima per classificare come set
    attack_confidence_threshold: float = 0.8  # confidence minima per classificare come attack


class TouchSequenceAgent:
    """
    Analizza sequenze di tocchi per validare e classificare secondo regole FIPAV.
    """

    def __init__(self, config: Optional[TouchSequenceConfig] = None):
        self.config = config or TouchSequenceConfig()

    def run(
        self,
        timeline: "Timeline",
        timeline_out: Optional["Timeline"] = None,
    ) -> List[Event]:
        """
        Analizza sequenze di tocchi nella timeline e pubblica eventi dettagliati.

        Args:
            timeline: Timeline con eventi HIT_LEFT/RIGHT esistenti
            timeline_out: Timeline opzionale per aggiungere eventi

        Returns:
            Lista di eventi TOUCH/SET/ATTACK dettagliati
        """
        events = self.analyze(timeline)
        if timeline_out is not None:
            timeline_out.extend(events)
        return events

    def analyze(self, timeline: "Timeline") -> List[Event]:
        """
        Analizza sequenze di tocchi e produce eventi dettagliati.

        Args:
            timeline: Timeline con eventi HIT_LEFT/RIGHT

        Returns:
            Lista di eventi dettagliati
        """
        all_events = timeline.sorted()
        
        # Estrai tutti gli HIT
        hits = [
            (e.time, e.type, e.confidence, e.extra or {})
            for e in all_events
            if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)
        ]

        if not hits:
            return []

        # Raggruppa tocchi in sequenze per lato
        sequences = self._group_touches_into_sequences(hits)

        # Analizza ogni sequenza
        detailed_events: List[Event] = []
        for side, touch_sequence in sequences.items():
            classified = self._classify_touch_sequence(side, touch_sequence, hits)
            detailed_events.extend(classified)

        return detailed_events

    def _group_touches_into_sequences(
        self, hits: List[Tuple[float, EventType, float, dict]]
    ) -> dict:
        """
        Raggruppa tocchi in sequenze per lato.

        Args:
            hits: Lista di (time, type, confidence, extra)

        Returns:
            Dict {side: [(time, type, confidence, extra), ...]}
        """
        sequences: dict = {"left": [], "right": []}

        for time, hit_type, confidence, extra in hits:
            side = "left" if hit_type == EventType.HIT_LEFT else "right"
            sequences[side].append((time, hit_type, confidence, extra))

        # Ordina per tempo
        for side in sequences:
            sequences[side].sort(key=lambda x: x[0])

        return sequences

    def _classify_touch_sequence(
        self, side: str, touch_sequence: List[Tuple], all_hits: List[Tuple]
    ) -> List[Event]:
        """
        Classifica una sequenza di tocchi e produce eventi dettagliati.

        Args:
            side: Lato del campo ("left" o "right")
            touch_sequence: Lista di tocchi del lato specificato
            all_hits: Tutti i tocchi (per contesto)

        Returns:
            Lista di eventi classificati
        """
        if not touch_sequence:
            return []

        cfg = self.config
        events: List[Event] = []

        # Separa in sequenze separate basandosi su gap temporali
        # Usa _split_into_sequences per dividere la sequenza in base a MAX_TIME_BETWEEN_TOUCHES
        separated_sequences = self._split_into_sequences(touch_sequence)

        for seq_idx, sequence in enumerate(separated_sequences):
            if not sequence:
                continue

            # Valida sequenza
            touches_data = [(t, "touch", side) for t, _, _, _ in sequence]
            is_valid, error, touch_count = validate_rally_touches(touches_data)

            if not is_valid and error:
                # Sequenza non valida: crea comunque eventi generici ma con warning
                pass

            # Classifica ogni tocco nella sequenza
            last_touch_time = None
            for pos, (time, hit_type, confidence, extra) in enumerate(sequence, start=1):
                total = len(sequence)
                last_touch_time = time
                
                # Estrai magnitude dal extra se disponibile
                magnitude = extra.get("left_mag", 0.0) if side == "left" else extra.get("right_mag", 0.0)
                if not magnitude:
                    magnitude = extra.get("magnitude", 0.0)

                # Classifica tocco
                touch_type_str = infer_touch_type_from_motion(magnitude, pos, total)

                # Determina EventType
                event_type = self._get_event_type_for_touch(side, touch_type_str, pos, total)

                # Crea evento dettagliato
                new_extra = extra.copy()
                new_extra.update({
                    "touch_position": pos,
                    "total_touches": total,
                    "touch_type": touch_type_str,
                    "sequence_id": seq_idx,
                    "magnitude": magnitude,
                    "is_sequence_end": (pos == total),  # flag per ultimo tocco della sequenza
                })

                event = Event(
                    time=time,
                    type=event_type,
                    confidence=confidence,
                    extra=new_extra,
                )

                events.append(event)
                
            # Dopo l'ultimo tocco della sequenza, aggiungi un evento implicito di fine sequenza
            # Questo può essere usato dal voting system per identificare fine rally
            # Nota: non aggiungiamo un evento esplicito TOUCH_SEQ_END, ma marcare l'ultimo
            # tocco con is_sequence_end=True nel extra è sufficiente per il voting system

        return events

    def _split_into_sequences(self, touches: List[Tuple]) -> List[List[Tuple]]:
        """
        Divide tocchi in sequenze separate basandosi su timing.

        Args:
            touches: Lista di tocchi ordinata per tempo

        Returns:
            Lista di sequenze (ogni sequenza è una lista di tocchi)
        """
        if not touches:
            return []

        sequences: List[List[Tuple]] = []
        current_sequence: List[Tuple] = [touches[0]]

        for i in range(1, len(touches)):
            prev_time = touches[i - 1][0]
            curr_time = touches[i][0]
            dt = curr_time - prev_time

            if dt <= MAX_TIME_BETWEEN_TOUCHES:
                # Stessa sequenza
                current_sequence.append(touches[i])
            else:
                # Nuova sequenza
                sequences.append(current_sequence)
                current_sequence = [touches[i]]

        # Aggiungi ultima sequenza
        if current_sequence:
            sequences.append(current_sequence)

        return sequences

    def _get_event_type_for_touch(self, side: str, touch_type: str, position: int, total: int) -> EventType:
        """
        Determina EventType per un tocco classificato.

        Args:
            side: Lato del campo ("left" o "right")
            touch_type: Tipo di tocco ("reception", "set", "attack", "touch")
            position: Posizione nella sequenza (1-based)
            total: Numero totale di tocchi

        Returns:
            EventType appropriato
        """
        side_suffix = "_LEFT" if side == "left" else "_RIGHT"

        if touch_type == "reception":
            return EventType[f"RECEPTION{side_suffix}"]
        elif touch_type == "set":
            return EventType[f"SET{side_suffix}"]
        elif touch_type == "attack":
            return EventType[f"ATTACK{side_suffix}"]
        elif touch_type == "block":
            return EventType[f"BLOCK{side_suffix}"]
        else:
            # Tocco generico
            return EventType[f"TOUCH{side_suffix}"]


__all__ = [
    "TouchSequenceAgent",
    "TouchSequenceConfig",
]

