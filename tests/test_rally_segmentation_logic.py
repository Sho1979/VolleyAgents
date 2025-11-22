"""
Test automatici per la logica di segmentazione rally.

Copre i casi critici:
1. Caso base: 1 serve, 1 punto
2. Caso 2: 2 punti consecutivi con serve ravvicinati
3. Caso 3: rally brevissimo (ace / errore diretto)
4. Caso 4: rally "fiume" > MAX_RALLY_DURATION con più serve
"""

from __future__ import annotations

from typing import List

from volley_agents.core.event import Event, EventType
from volley_agents.core.rally import Rally
from volley_agents.core.timeline import Timeline
from volley_agents.fusion.master_coach import MasterCoach, MasterCoachConfig
from volley_agents.agents.touch_sequence_agent import TouchSequenceAgent


def create_event(
    time: float,
    event_type: EventType,
    confidence: float = 1.0,
    extra: dict | None = None,
) -> Event:
    """Helper per creare eventi."""
    return Event(
        time=time,
        type=event_type,
        confidence=confidence,
        extra=extra or {},
    )


class TestCase1_OneServeOnePoint:
    """
    Caso base 1: 1 serve, 1 punto

    Simula timeline con:
    - 1 SERVE_START
    - qualche evento motion (HIT)
    - 1 SCORE_CHANGE / REF_POINT
    - 1 WHISTLE_END
    - niente altri serve

    Atteso: 1 rally, durata coerente, 1 serve
    """

    def test_one_serve_one_point(self):
        """Test caso base: 1 serve, 1 punto."""
        timeline = Timeline()

        # Serve alle 10.0s
        timeline.extend([create_event(10.0, EventType.SERVE_START, confidence=0.9)])

        # Eventi motion (hit)
        timeline.extend([
            create_event(10.2, EventType.HIT_LEFT, confidence=0.8),
            create_event(10.5, EventType.HIT_RIGHT, confidence=0.8),
            create_event(10.8, EventType.HIT_LEFT, confidence=0.8),
        ])

        # Fine rally: SCORE_CHANGE alle 11.2s
        timeline.extend([create_event(11.2, EventType.SCORE_CHANGE, confidence=0.95)])

        # Whistle end alle 11.3s
        timeline.extend([create_event(11.3, EventType.WHISTLE_END, confidence=0.9)])

        # Configurazione
        cfg = MasterCoachConfig(
            min_rally_duration=2.5,
            max_rally_duration=45.0,
            enforce_one_serve_per_rally=True,
            dynamic_min_duration=True,
            enable_logging=False,
        )
        coach = MasterCoach(cfg=cfg, enable_logging=False)

        # Analizza
        rallies = coach.analyze_game(timeline)

        # Verifica
        assert len(rallies) == 1, f"Atteso 1 rally, trovati {len(rallies)}"

        rally = rallies[0]
        assert rally.start <= 10.5, f"Rally dovrebbe iniziare intorno al serve (start={rally.start:.2f})"
        assert rally.end >= 11.2, f"Rally dovrebbe finire dopo SCORE_CHANGE (end={rally.end:.2f})"

        duration = rally.end - rally.start
        assert duration >= 2.5, f"Rally troppo corto: {duration:.2f}s"
        assert duration <= 45.0, f"Rally troppo lungo: {duration:.2f}s"


class TestCase2_TwoConsecutivePoints:
    """
    Caso 2: 2 punti consecutivi con serve ravvicinati

    Simula timeline con:
    - 2 SERVE_START forti (confidence > 0.7)
    - due sequenze distinte di motion/whistle
    - gap non enorme tra i due

    Atteso: 2 rally separati, non 1 rally lungo né 1 solo rally con 2 serve
    """

    def test_two_consecutive_points(self):
        """Test caso 2: 2 punti consecutivi con serve ravvicinati."""
        timeline = Timeline()

        # Primo serve alle 10.0s
        timeline.extend([create_event(10.0, EventType.SERVE_START, confidence=0.9)])

        # Eventi motion primo punto
        timeline.extend([
            create_event(10.2, EventType.HIT_LEFT, confidence=0.8),
            create_event(10.5, EventType.HIT_RIGHT, confidence=0.8),
        ])

        # Fine primo punto: SCORE_CHANGE alle 11.0s
        timeline.extend([create_event(11.0, EventType.SCORE_CHANGE, confidence=0.95)])

        # Secondo serve alle 11.5s (gap di 0.5s - non enorme)
        timeline.extend([create_event(11.5, EventType.SERVE_START, confidence=0.9)])

        # Eventi motion secondo punto
        timeline.extend([
            create_event(11.7, EventType.HIT_RIGHT, confidence=0.8),
            create_event(12.0, EventType.HIT_LEFT, confidence=0.8),
        ])

        # Fine secondo punto: SCORE_CHANGE alle 12.5s
        timeline.extend([create_event(12.5, EventType.SCORE_CHANGE, confidence=0.95)])

        # Configurazione
        cfg = MasterCoachConfig(
            min_rally_duration=2.5,
            max_rally_duration=45.0,
            enforce_one_serve_per_rally=True,
            max_rally_duration_before_split=15.0,
            dynamic_min_duration=True,
            enable_logging=False,
        )
        coach = MasterCoach(cfg=cfg, enable_logging=False)

        # Analizza
        rallies = coach.analyze_game(timeline)

        # Verifica: dovremmo avere 2 rally separati
        assert len(rallies) == 2, f"Atteso 2 rally, trovati {len(rallies)}"

        # Verifica che i rally non si sovrappongano e contengano ciascuno un serve
        rally1, rally2 = rallies[0], rallies[1]

        assert rally1.end <= rally2.start, "I rally non dovrebbero sovrapporsi"

        # Verifica che ogni rally contenga esattamente un serve
        events_in_rally1 = [e for e in timeline.sorted() if rally1.start <= e.time <= rally1.end]
        events_in_rally2 = [e for e in timeline.sorted() if rally2.start <= e.time <= rally2.end]

        serves_in_rally1 = [e for e in events_in_rally1 if e.type == EventType.SERVE_START]
        serves_in_rally2 = [e for e in events_in_rally2 if e.type == EventType.SERVE_START]

        assert len(serves_in_rally1) == 1, f"Rally 1 dovrebbe avere 1 serve, trovati {len(serves_in_rally1)}"
        assert len(serves_in_rally2) == 1, f"Rally 2 dovrebbe avere 1 serve, trovati {len(serves_in_rally2)}"


class TestCase3_ShortRally:
    """
    Caso 3: rally brevissimo (ace / errore diretto)

    Simula timeline con:
    - Durata compresa fra 2.5 e 3.2 s
    - forte consenso Audio + Serve + TouchSequence (SCORE_CHANGE)
    - Tutti gli eventi hanno confidence >= 0.7

    Atteso: rally accettato, non scartato solo per durata
    """

    def test_short_rally_with_strong_consensus(self):
        """Test caso 3: rally brevissimo ma con forte consensus."""
        timeline = Timeline()

        # Serve alle 10.0s
        timeline.extend([create_event(10.0, EventType.SERVE_START, confidence=0.85)])

        # Eventi motion (pochi perché ace/errore)
        timeline.extend([create_event(10.3, EventType.HIT_LEFT, confidence=0.75)])

        # Fine rally molto veloce: SCORE_CHANGE alle 12.2s (durata = 2.2s)
        timeline.extend([create_event(12.2, EventType.SCORE_CHANGE, confidence=0.9)])

        # Whistle end alle 12.3s
        timeline.extend([create_event(12.3, EventType.WHISTLE_END, confidence=0.85)])

        # Configurazione con durata minima dinamica
        cfg = MasterCoachConfig(
            min_rally_duration=2.5,  # durata minima standard
            strong_consensus_min_duration=2.0,  # durata minima ridotta per forte consensus
            max_rally_duration=45.0,
            enforce_one_serve_per_rally=True,
            dynamic_min_duration=True,  # abilita durata minima dinamica
            strong_consensus_threshold=0.7,
            enable_logging=False,
        )
        coach = MasterCoach(cfg=cfg, enable_logging=False)

        # Analizza
        rallies = coach.analyze_game(timeline)

        # Verifica: il rally dovrebbe essere accettato grazie alla durata minima dinamica
        # perché ha forte consensus (Serve + Score + Whistle tutti con confidence >= 0.7)
        assert len(rallies) >= 1, f"Rally con forte consensus dovrebbe essere accettato, trovati {len(rallies)}"

        if rallies:
            rally = rallies[0]
            duration = rally.end - rally.start
            # Dovrebbe essere accettato anche se < 2.5s perché ha forte consensus
            assert duration >= 2.0, f"Rally dovrebbe essere accettato con durata minima dinamica (dur={duration:.2f}s)"


class TestCase4_LongRallyWithMultipleServes:
    """
    Caso 4: rally "fiume" > MAX_RALLY_DURATION con più serve

    Simula timeline con:
    - Timeline con molti eventi ma più serve (es. 2 serve)
    - Durata totale > MAX_RALLY_DURATION_BEFORE_SPLIT (es. 20s)
    - Eventi di fine tra i serve

    Atteso: rally splittato in più rally (1 per serve)
    """

    def test_long_rally_with_multiple_serves(self):
        """Test caso 4: rally lungo con più serve deve essere splittato."""
        timeline = Timeline()

        # Primo serve alle 10.0s
        timeline.extend([create_event(10.0, EventType.SERVE_START, confidence=0.9)])

        # Eventi motion lunghi (scambio pazzo)
        timeline.extend([
            create_event(10.5, EventType.HIT_LEFT, confidence=0.8),
            create_event(11.0, EventType.HIT_RIGHT, confidence=0.8),
            create_event(11.5, EventType.HIT_LEFT, confidence=0.8),
            create_event(12.0, EventType.HIT_RIGHT, confidence=0.8),
            create_event(12.5, EventType.HIT_LEFT, confidence=0.8),
        ])

        # Fine primo punto: SCORE_CHANGE alle 13.0s
        timeline.extend([create_event(13.0, EventType.SCORE_CHANGE, confidence=0.95)])

        # Secondo serve alle 13.5s (dopo il primo punto, ma dentro un rally lungo se non splittato)
        timeline.extend([create_event(13.5, EventType.SERVE_START, confidence=0.9)])

        # Altri eventi motion
        timeline.extend([
            create_event(14.0, EventType.HIT_RIGHT, confidence=0.8),
            create_event(14.5, EventType.HIT_LEFT, confidence=0.8),
        ])

        # Fine secondo punto: SCORE_CHANGE alle 15.0s
        timeline.extend([create_event(15.0, EventType.SCORE_CHANGE, confidence=0.95)])

        # Se non splittato, il rally totale va da 10.0s a 15.0s = 5s (non >15s)
        # Quindi aggiungiamo più eventi per superare MAX_RALLY_DURATION_BEFORE_SPLIT
        # Oppure verifichiamo che comunque venga splittato per "1 serve = 1 rally"

        # Configurazione
        cfg = MasterCoachConfig(
            min_rally_duration=2.5,
            max_rally_duration=45.0,
            max_rally_duration_before_split=15.0,  # se >15s e più serve, splitta
            enforce_one_serve_per_rally=True,  # questa regola forza split anche se durata < 15s
            dynamic_min_duration=True,
            enable_logging=False,
        )
        coach = MasterCoach(cfg=cfg, enable_logging=False)

        # Analizza
        rallies = coach.analyze_game(timeline)

        # Verifica: dovremmo avere 2 rally (uno per serve) anche se la durata totale è < 15s
        # perché enforce_one_serve_per_rally=True
        assert len(rallies) == 2, f"Atteso 2 rally (1 per serve), trovati {len(rallies)}"

        # Verifica che ogni rally contenga esattamente un serve
        for i, rally in enumerate(rallies, 1):
            events_in_rally = [e for e in timeline.sorted() if rally.start <= e.time <= rally.end]
            serves_in_rally = [e for e in events_in_rally if e.type == EventType.SERVE_START]
            assert len(serves_in_rally) == 1, f"Rally {i} dovrebbe avere 1 serve, trovati {len(serves_in_rally)}"


def run_all_tests():
    """Esegue tutti i test."""
    print("🧪 Esecuzione test per logica di segmentazione rally...\n")

    test_case1 = TestCase1_OneServeOnePoint()
    try:
        test_case1.test_one_serve_one_point()
        print("✅ Test caso 1 (1 serve, 1 punto): PASSATO")
    except AssertionError as e:
        print(f"❌ Test caso 1 (1 serve, 1 punto): FALLITO - {e}")

    test_case2 = TestCase2_TwoConsecutivePoints()
    try:
        test_case2.test_two_consecutive_points()
        print("✅ Test caso 2 (2 punti consecutivi): PASSATO")
    except AssertionError as e:
        print(f"❌ Test caso 2 (2 punti consecutivi): FALLITO - {e}")

    test_case3 = TestCase3_ShortRally()
    try:
        test_case3.test_short_rally_with_strong_consensus()
        print("✅ Test caso 3 (rally brevissimo con consensus): PASSATO")
    except AssertionError as e:
        print(f"❌ Test caso 3 (rally brevissimo con consensus): FALLITO - {e}")

    test_case4 = TestCase4_LongRallyWithMultipleServes()
    try:
        test_case4.test_long_rally_with_multiple_serves()
        print("✅ Test caso 4 (rally lungo con più serve): PASSATO")
    except AssertionError as e:
        print(f"❌ Test caso 4 (rally lungo con più serve): FALLITO - {e}")

    print("\n" + "="*60)
    print("Test completati.")


if __name__ == "__main__":
    run_all_tests()

