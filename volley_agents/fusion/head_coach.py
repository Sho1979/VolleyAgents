from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from volley_agents.core.event import Event, EventType
from volley_agents.core.timeline import Timeline
from volley_agents.core.rally import Rally, RallyProposal


@dataclass
class HeadCoachConfig:
    min_rally_duration: float = 0.3  # permette ACE/errori servizio (rally molto brevi)
    max_rally_duration: float = 45.0  # massimo 45s per un rally
    min_rally_gap: float = 1.0  # gap minimo per evitare spezzatini troppo vicini
    serve_min_delay: float = 0.2
    serve_max_delay: float = 10.0
    gap_split: float = 7.0

    score_first_mode: bool = True         # abilita il comportamento "score come splitter"
    score_min_events: int = 2            # ridotto per fidarsi di SCORE anche con meno eventi (era 3)
    score_max_interval: float = 40.0      # durata massima plausibile per un singolo rally


class HeadCoach:
    """
    HeadCoach - Sistema di fusione e validazione rally per VolleyAgents
    
    TEST DI REGRESSIONE:
    ====================
    ⚠️ IMPORTANTE: Ogni modifica a HeadCoach deve passare il test di regressione:
    
    python -m tools.video_pipeline.eval_rallies_gt \
      --gt tools/video_pipeline/ground_truth/gt_millennium_16m50_26m54_verified.json \
      --pred tools/video_pipeline/ground_truth/rallies_1010_1600.json \
      --iou 0.5
    
    Risultati attesi: Precision=1.000, Recall=1.000, 19/19 match
    Vedi README_dev.md per dettagli completi e versioni alternative del comando.
    
    REGOLE ATTIVE:
    ==============
    - Serve obbligatoria: ogni rally deve contenere SERVE_START
    - Durata minima: 0.3s (permette ACE/errori servizio)
    - Durata massima: 45.0s
    - Gap minimo: 1.0s tra rally consecutivi
    - Validazione attività palla: rally > 0.5s devono avere azioni palla
    
    ⚠️ Questa configurazione è quella validata dal test di regressione principale (19/19 rally corretti).
    Se modifichi queste regole, esegui sempre il test prima di fare commit.
    """
    def __init__(self, cfg: Optional[HeadCoachConfig] = None):
        self.cfg = cfg or HeadCoachConfig()

    # 1) entrypoint principale
    def build_rallies(self, timeline: Timeline) -> List[Rally]:
        events = timeline.sorted()
        score_events = [e for e in events if e.type == EventType.SCORE_CHANGE]

        # 1) se SCORE è forte: usa lui per definire gli intervalli
        if self._can_use_score_as_splitter(score_events, events):
            intervals = self._build_score_intervals(score_events, events)
            rally_proposals = []
            for (t_start, t_end) in intervals:
                rp = self._propose_rally_in_interval(events, t_start, t_end)
                if rp is not None:
                    rally_proposals.append(rp)
            rallies = [self._finalize_proposal(rp) for rp in rally_proposals if rp is not None]
        else:
            # 2) fallback: logica attuale audio+motion su tutto il periodo
            rallies = self._build_rallies_audio_motion(events)
        
        # 3) Aggiungi rally brevi che potrebbero essere sfuggiti alla detection normale
        short_rallies = self._detect_short_rallies(events)
        
        # Merge: aggiungi rally brevi che non sovrappongono con normali
        for start, end in short_rallies:
            overlaps = any(
                not (end < r.start or start > r.end)
                for r in rallies
            )
            if not overlaps:
                # Determina side dal serve se disponibile
                serve_events = [e for e in events 
                               if e.type == EventType.SERVE_START 
                               and abs(e.time - start) < 0.5]
                side = "unknown"
                if serve_events and serve_events[0].extra and "side" in serve_events[0].extra:
                    side = serve_events[0].extra["side"]
                
                rallies.append(Rally(start=start, end=end, side=side))
        
        # Ordina per tempo di inizio
        rallies.sort(key=lambda r: r.start)
        
        return rallies

    # -------------------------------------------------
    # SCORE-FIRST: verifica se SCORE è affidabile come splitter
    # -------------------------------------------------
    def _can_use_score_as_splitter(self, score_events: List[Event], all_events: List[Event]) -> bool:
        cfg = self.cfg
        if not cfg.score_first_mode:
            return False
        if len(score_events) < cfg.score_min_events:
            return False
        # opzionale: puoi controllare che la durata media tra score_change sia > min_rally_duration
        # e < score_max_interval
        times = [e.time for e in score_events]
        if len(times) < 2:
            return False
        gaps = np.diff(sorted(times))
        if len(gaps) == 0:
            return False
        if np.median(gaps) < cfg.min_rally_duration:
            return False
        if np.median(gaps) > cfg.score_max_interval:
            return False
        return True

    def _build_score_intervals(self, score_events: List[Event], all_events: List[Event]) -> List[Tuple[float, float]]:
        times = sorted([e.time for e in score_events])
        if not times:
            return []

        t_min = all_events[0].time
        t_max = all_events[-1].time

        intervals = []

        # opzionale: se il video inizia prima del primo score_change
        if t_min < times[0]:
            intervals.append((t_min, times[0]))

        for i in range(len(times) - 1):
            intervals.append((times[i], times[i + 1]))

        # ultimo tratto
        if times[-1] < t_max:
            intervals.append((times[-1], t_max))

        return intervals

    def _propose_rally_in_interval(
        self,
        all_events: List[Event],
        t_start: float,
        t_end: float,
    ) -> Optional[RallyProposal]:
        cfg = self.cfg

        # 1) filtra gli eventi nell'intervallo
        window = [e for e in all_events if t_start <= e.time <= t_end]

        hits = [e for e in window if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)]
        ball_events = [e for e in window if e.type in (
            EventType.BALL_TOUCH_GROUND, EventType.BALL_CROSS_NET,
            EventType.BALL_SERVE, EventType.BALL_DETECTED
        )]
        whistles = [e for e in window if e.type == EventType.WHISTLE_END]

        if not hits and not ball_events:
            # se SCORE ha segmentato ma non vediamo hit o eventi palla, non proponiamo nulla
            return None

        # 2) inizio candidati: primo hit "importante" nel blocco
        first_hit = hits[0]
        start_candidate = first_hit.time

        # 3) fine candidati:
        #   - whistle_end nel blocco dopo l'ultimo hit (solo se valido)
        last_hit = hits[-1]
        end_candidate = last_hit.time

        later_whistles = [w for w in whistles if w.time >= last_hit.time]
        if later_whistles:
            # Verifica se il fischio indica davvero fine rally
            for whistle in later_whistles:
                if self._should_end_rally_at_whistle(whistle.time, all_events):
                    end_candidate = whistle.time
                    break

        # 4) controlli minimi e massimi
        dur = end_candidate - start_candidate
        if dur < cfg.min_rally_duration or dur > cfg.max_rally_duration:
            return None

        # 5) stima della "confidence": più lungo = più sicuro (molto semplice per ora)
        dur = end_candidate - start_candidate
        conf = min(1.0, dur / (cfg.score_max_interval or 30.0))

        return RallyProposal(
            interval_start=t_start,
            interval_end=t_end,
            start_candidate=start_candidate,
            end_candidate=end_candidate,
            source="score+agents",
            confidence=conf,
        )

    def _finalize_proposal(self, rp: RallyProposal) -> Rally:
        # per ora usiamo direttamente i candidate; in futuro potremo far votare altri agenti
        start = rp.start_candidate if rp.start_candidate is not None else rp.interval_start
        end = rp.end_candidate if rp.end_candidate is not None else rp.interval_end
        return Rally(start=start, end=end, side="unknown")

    # -------------------------------------------------
    # Helper: eventi in un intervallo
    # -------------------------------------------------
    def _events_in_interval(self, events: List[Event], start: float, end: float) -> List[Event]:
        """Filtra eventi nell'intervallo [start, end]."""
        return [e for e in events if start <= e.time <= end]

    # -------------------------------------------------
    # Helper: validazione fine rally da WHISTLE_END
    # -------------------------------------------------
    def _should_end_rally_at_whistle(self, whistle_time: float, all_events: List[Event]) -> bool:
        """
        Verifica se un fischio indica davvero fine rally.
        Ritorna False se c'è ancora attività dopo il fischio.
        
        Un WHISTLE_END termina il rally SOLO SE:
        1. NON ci sono altri SERVE_START entro i 5 secondi successivi al fischio
        2. OPPURE c'è un gap di attività (nessun HIT) di almeno 3 secondi dopo il fischio
        """
        # Cerca serve nei prossimi 5 secondi
        next_serves = [e for e in all_events 
                       if e.type == EventType.SERVE_START 
                       and whistle_time < e.time < whistle_time + 5.0]
        
        if next_serves:
            # C'è un serve subito dopo → il fischio NON è fine rally
            return False
        
        # Cerca HIT nei prossimi 3 secondi
        next_hits = [e for e in all_events
                     if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)
                     and whistle_time < e.time < whistle_time + 3.0]
        
        if next_hits:
            # C'è ancora attività → il fischio NON è fine rally
            return False
        
        return True  # OK, il fischio indica fine rally

    # -------------------------------------------------
    # Helper: rileva rally brevi (serve → hit → fischio)
    # -------------------------------------------------
    def _detect_short_rallies(self, events: List[Event]) -> List[Tuple[float, float]]:
        """
        Rileva rally brevi (1-2 scambi) che potrebbero sfuggire alla detection normale.
        
        Pattern: serve (confidence >= 0.70) → HIT entro 3s → WHISTLE_END entro 5s dall'ultimo HIT
        """
        short_rallies = []
        serves = [e for e in events if e.type == EventType.SERVE_START and e.confidence >= 0.70]
        whistles = [e for e in events if e.type == EventType.WHISTLE_END]
        hits = [e for e in events if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)]
        
        for serve in serves:
            # Cerca HIT entro 3 secondi dal serve
            serve_hits = [h for h in hits if serve.time < h.time < serve.time + 3.0]
            
            if not serve_hits:
                continue  # Nessuna ricezione → probabilmente ace (gestito altrove)
            
            last_hit = max(serve_hits, key=lambda h: h.time)
            
            # Cerca fischio entro 5 secondi dall'ultimo HIT
            end_whistle = next(
                (w for w in whistles if last_hit.time < w.time < last_hit.time + 5.0),
                None
            )
            
            if end_whistle:
                # Verifica che il fischio sia valido per fine rally
                if not self._should_end_rally_at_whistle(end_whistle.time, events):
                    continue
                
                # Rally breve confermato: serve → hit → fischio
                rally_start = serve.time
                rally_end = end_whistle.time
                duration = rally_end - rally_start
                
                if 1.5 < duration < 8.0:  # Rally breve tipico
                    short_rallies.append((rally_start, rally_end))
                    # Log per debug (commentato se non c'è _log)
                    # self._log(f"🏐 Rally breve rilevato: {rally_start:.2f}s → {rally_end:.2f}s (dur={duration:.1f}s)")
        
        return short_rallies

    # -------------------------------------------------
    # Helper: trova candidati per inizio rally (con promozione serve in sequenza)
    # -------------------------------------------------
    def _find_rally_start_candidates(self, events: List[Event]) -> List[Event]:
        """
        Trova candidati per inizio rally.
        Considera anche serve a bassa confidence se fanno parte di una sequenza.
        
        Serve forte (confidence >= 0.70) → sempre candidato
        Serve debole (0.55-0.70) → candidato SE parte di sequenza (2+ serve entro 6s)
        """
        serves = [e for e in events if e.type == EventType.SERVE_START]
        
        candidates = []
        for i, serve in enumerate(serves):
            # Serve forte → sempre candidato
            if serve.confidence >= 0.70:
                candidates.append(serve)
                continue
            
            # Serve debole (0.55-0.70) → candidato SE parte di sequenza
            if serve.confidence >= 0.55:
                # Conta serve vicini (entro 6 secondi prima/dopo)
                nearby_serves = [s for s in serves 
                               if s != serve 
                               and abs(s.time - serve.time) < 6.0]
                
                if len(nearby_serves) >= 1:  # Almeno un altro serve vicino
                    # La sequenza rafforza la confidence
                    candidates.append(serve)
                    # Log per debug (se disponibile)
                    # self._log(f"🎯 Serve@{serve.time:.2f}s (conf={serve.confidence:.2f}) "
                    #          f"promosso per sequenza ({len(nearby_serves)+1} serve vicini)")
        
        return candidates

    # -------------------------------------------------
    # Helper: validazione "no serve, no rally"
    # -------------------------------------------------
    def _is_valid_rally_segment(self, seg: Rally, events: List[Event]) -> bool:
        """
        Valida un segmento di rally:
        - Deve contenere almeno una battuta (SERVE_START)
        - Durata compresa fra min_rally_duration e max_rally_duration
        - Permette rally molto brevi (<0.3s) se hanno SERVE_START (ACE/errori servizio)
        - Elimina segmenti senza attività palla
        """
        cfg = self.cfg
        evts = self._events_in_interval(events, seg.start, seg.end)
        
        has_serve = any(e.type == EventType.SERVE_START for e in evts)
        if not has_serve:
            # niente battuta → non consideriamo questo segmento un rally vero
            return False
        
        duration = seg.end - seg.start
        
        # Permetti rally molto brevi se c'è SERVE_START (ACE/errori servizio)
        if duration < 0.3:
            # Se c'è una SERVE_START e un WHISTLE_END immediato → ACE o errore diretto
            if has_serve:
                return True
            return False
        
        # usa le soglie della config per rally normali
        if duration < cfg.min_rally_duration:
            return False
        if duration > cfg.max_rally_duration:
            return False
        
        # Elimina segmenti senza attività palla (eccetto ACE/errori già gestiti sopra)
        # IMPORTANTE: Gli eventi BALL_* NON validano un rally - possono solo confermare fine rally
        # o contare scambi, ma MAI validare che un segmento sia un rally valido!
        has_ball_action = any(e.type in (
            EventType.HIT_LEFT, EventType.HIT_RIGHT,
            EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT,
            EventType.RECEPTION_LEFT, EventType.RECEPTION_RIGHT,
            EventType.SET_LEFT, EventType.SET_RIGHT
        ) for e in evts)
        
        if not has_ball_action and duration > 0.5:
            # se non è un ace/errore e non c'è attività → no rally
            return False
        
        # Filtro aggiuntivo: se ha serve ma nessuna azione palla, è probabilmente un segmento vuoto
        # (es. inizio finestra con pezzo morto prima del primo rally vero)
        if has_serve and not has_ball_action and duration > 1.0:
            # serve presente ma nessuna azione palla → segmento vuoto, scarta
            return False
        
        # Filtro rally troppo corti senza attività significativa
        # Un rally valido sotto i 5 secondi DEVE avere almeno 2 HIT
        if duration < 5.0:
            hit_count = sum(1 for e in evts if e.type in (
                EventType.HIT_LEFT, EventType.HIT_RIGHT,
                EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT
            ))
            if hit_count < 2:
                return False  # Rally troppo corto senza abbastanza attività
        
        return True

    # -------------------------------------------------
    # Helper: split su segmenti con più battute
    # -------------------------------------------------
    def _split_on_serves_if_needed(self, seg: Rally, events: List[Event]) -> List[Rally]:
        """
        Per ora non splittiamo più: un macro-rally resta intero.
        Ma agganciamo serve entro 1.0s prima del segmento per evitare spezzettini.
        TODO: reintrodurre split solo se ci sono casi chiari di due punti distinti.
        """
        # se il segmento NON contiene nessuna serve ma ha serve entro 1.0s prima → agganciala
        prev_serves = [e for e in events if e.type == EventType.SERVE_START and seg.start - 1.0 <= e.time <= seg.start]
        if prev_serves:
            # prendi la serve più vicina (ultima prima di seg.start)
            closest_serve = max(prev_serves, key=lambda e: e.time)
            seg.start = closest_serve.time
        
        return [seg]

    # -------------------------------------------------
    # Helper: refine di start/end con serve, whistle e motion gap
    # -------------------------------------------------
    def _refine_segment_boundaries(self, seg: Rally, events: List[Event]) -> Rally:
        """
        Rifinisce i confini di un segmento:
        - Start: aggancia a SERVE_START o WHISTLE_START precedente
        - End: taglia su WHISTLE_END, MOTION_GAP lunghi, o max 3s dopo ultimo HIT
        """
        # Estendi la finestra di ricerca per eventi vicini
        evts = sorted(
            self._events_in_interval(events, seg.start - 3.0, seg.end + 3.0),
            key=lambda e: e.time
        )
        
        # ---- START: aggancia alla battuta / fischio più vicini ----
        # cerca serve subito prima di seg.start (lookback max 3s)
        serves_before = [
            e for e in evts
            if e.type == EventType.SERVE_START and e.time <= seg.start
        ]
        if serves_before:
            last_serve = max(serves_before, key=lambda e: e.time)
            if seg.start - last_serve.time <= 3.0:
                seg.start = max(seg.start - 1.0, last_serve.time)  # margine piccolo
        else:
            # fallback: whistle_start prima dell'inizio
            whistles_before = [
                e for e in evts
                if e.type == EventType.WHISTLE_START and e.time <= seg.start
            ]
            if whistles_before:
                last_wh = max(whistles_before, key=lambda e: e.time)
                if seg.start - last_wh.time <= 3.0:
                    seg.start = max(seg.start - 0.5, last_wh.time)
        
        # ---- END: taglia usando WHISTLE_END / MOTION_GAP / ultimo HIT ----
        candidate_end = seg.end
        
        # 1) whistle_end dentro il segmento (solo se valido)
        whistles_end = [
            e for e in evts
            if e.type == EventType.WHISTLE_END and seg.start <= e.time <= seg.end + 3.0
        ]
        if whistles_end:
            # Verifica quali fischi sono validi per fine rally
            valid_whistles = [w for w in whistles_end 
                             if self._should_end_rally_at_whistle(w.time, events)]
            if valid_whistles:
                last_wh_end = max(valid_whistles, key=lambda e: e.time)
                candidate_end = min(candidate_end, last_wh_end.time + 0.5)  # mezzo secondo dopo il fischio
        
        # 2) motion gap subito dopo un periodo di attività
        LONG_GAP = 2.5
        motion_gaps = [
            e for e in evts
            if e.type == EventType.MOTION_GAP 
            and seg.start <= e.time <= seg.end + 3.0
            and e.extra and e.extra.get("duration", 0) >= LONG_GAP
        ]
        if motion_gaps:
            early_gap = min(
                (e for e in motion_gaps if e.time >= seg.start),
                key=lambda e: e.time,
                default=None,
            )
            if early_gap and early_gap.time - seg.start > 1.0:
                candidate_end = min(candidate_end, early_gap.time)
        
        # 3) ultimo HIT/TOUCH prima della fine
        ball_actions = [
            e for e in evts
            if e.type in (
                EventType.HIT_LEFT, EventType.HIT_RIGHT,
                EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT,
                EventType.RECEPTION_LEFT, EventType.RECEPTION_RIGHT,
                EventType.SET_LEFT, EventType.SET_RIGHT,
                # Eventi palla da BallAgent
                EventType.BALL_TOUCH_GROUND, EventType.BALL_CROSS_NET,
                EventType.BALL_SERVE
            )
        ]
        if ball_actions:
            last_action = max(ball_actions, key=lambda e: e.time)
            # non lasciamo mai più di 3s di coda dopo l'ultima azione
            candidate_end = min(candidate_end, last_action.time + 3.0)
        
        # TODO (opzionale, bassa priorità): Micro-tuning fine rally
        # Se ci sono HIT/ATTACK/SET negli ultimi 1-1.5 secondi prima di seg.end,
        # estendi l'end al timestamp dell'ultima azione + 0.3s, purché non superi:
        # - il prossimo SERVE_START
        # - il t_end della finestra (se disponibile)
        # 
        # Esempio implementazione (commentata):
        # recent_actions = [a for a in ball_actions if seg.end - 1.5 <= a.time <= seg.end]
        # if recent_actions:
        #     last_recent = max(recent_actions, key=lambda e: e.time)
        #     extended_end = min(last_recent.time + 0.3, candidate_end)
        #     # Verifica che non ci sia un serve subito dopo
        #     next_serves = [e for e in evts if e.type == EventType.SERVE_START and e.time > seg.end]
        #     if not next_serves or extended_end < next_serves[0].time:
        #         candidate_end = extended_end
        
        # applica candidate_end se non rende il rally ridicolo
        if candidate_end - seg.start >= 1.5:
            seg.end = candidate_end
        
        return seg

    # -------------------------------------------------
    # Fallback: Audio + Motion → rally grezzi (macro-based, non per serve)
    # -------------------------------------------------
    def _build_rallies_audio_motion(self, events: List[Event]) -> List[Rally]:
        """
        Costruisce macro-rallies basati su attività continua e segnali di fine forti.
        Un macro-rally può contenere più serve (2-4 serve per rally è accettabile).
        
        Logica:
        - Activity events (SERVE_START, HIT_LEFT/RIGHT, WHISTLE_START) iniziano/continuano segmenti
        - End events (SCORE_CHANGE, REF_POINT, WHISTLE_END, MOTION_GAP > 3s) chiudono segmenti
        - Gap di inattività > gap_split o durata > max_rally_duration chiudono segmenti
        """
        cfg = self.cfg
        
        if not events:
            return []
        
        # Estrai tutti gli eventi rilevanti
        # Usa _find_rally_start_candidates() per includere serve a bassa confidence in sequenza
        serve_candidates = self._find_rally_start_candidates(events)
        serves = serve_candidates  # Usa solo i candidati validi (forti o in sequenza)
        
        scores = [e for e in events if e.type == EventType.SCORE_CHANGE]
        referee_points = [e for e in events if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)]
        whistles_end = [e for e in events if e.type == EventType.WHISTLE_END]
        whistles_start = [e for e in events if e.type == EventType.WHISTLE_START]
        gaps = [e for e in events if e.type == EventType.MOTION_GAP]
        hits = [e for e in events if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)]
        ball_events = [e for e in events if e.type in (
            EventType.BALL_TOUCH_GROUND, EventType.BALL_CROSS_NET,
            EventType.BALL_SERVE, EventType.BALL_DETECTED
        )]

        # Crea set di end events per lookup veloce (solo fischi validi)
        end_events_set = set()
        for e in scores + referee_points:
            end_events_set.add(e.time)
        # Aggiungi solo fischi validi per fine rally
        for w in whistles_end:
            if self._should_end_rally_at_whistle(w.time, events):
                end_events_set.add(w.time)
        for g in gaps:
            if g.extra and g.extra.get("duration", 0) > 3.0:
                end_events_set.add(g.time)

        # Se non ci sono serve, prova comunque con hits, ball events e whistles
        if not serves and not hits and not ball_events:
            return []

        # Crea lista di tutti gli eventi rilevanti per attività, ordinati per tempo
        activity_events = []
        activity_events.extend(serves)
        activity_events.extend(hits)
        activity_events.extend(ball_events)  # eventi palla da BallAgent
        activity_events.extend(whistles_start)
        activity_events.extend(whistles_end)  # anche whistles possono indicare attività
        activity_events = sorted(activity_events, key=lambda e: e.time)

        if not activity_events:
            return []

        rallies: List[Rally] = []
        
        # Stato del segmento corrente
        current_start: Optional[float] = None
        last_activity_time: Optional[float] = None
        strong_end_candidate: Optional[float] = None
        rally_side: str = "unknown"

        # Soglia per gap di inattività (usa gap_split o un default)
        gap_threshold = getattr(cfg, 'gap_split', 7.0)

        def close_current_segment():
            """Chiude il segmento corrente e aggiunge un rally se valido."""
            nonlocal current_start, last_activity_time, strong_end_candidate
            
            if current_start is None:
                return
            
            # Determina la fine del segmento
            segment_end: Optional[float] = None
            
            # Priorità: strong_end_candidate > last_activity_time
            if strong_end_candidate is not None:
                segment_end = strong_end_candidate
            elif last_activity_time is not None:
                segment_end = last_activity_time
            else:
                # Non dovrebbe succedere, ma fallback
                current_start = None
                return
            
            # Verifica durata valida
            dur = segment_end - current_start
            if cfg.min_rally_duration <= dur <= cfg.max_rally_duration:
                rallies.append(Rally(start=current_start, end=segment_end, side=rally_side))
            
            # Reset per il prossimo segmento
            current_start = None
            last_activity_time = None
            strong_end_candidate = None

        # Iterazione principale: passa attraverso tutti gli eventi ordinati per tempo
        all_relevant_events = sorted(
            serves + hits + whistles_start + whistles_end + scores + referee_points + gaps,
            key=lambda e: e.time
        )

        for event in all_relevant_events:
            event_time = event.time
            
            # Verifica se è un evento di attività
            is_activity = (
                event.type == EventType.SERVE_START or
                event.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT) or
                event.type in (
                    EventType.BALL_TOUCH_GROUND, EventType.BALL_CROSS_NET,
                    EventType.BALL_SERVE, EventType.BALL_DETECTED
                ) or
                event.type == EventType.WHISTLE_START or
                event.type == EventType.WHISTLE_END
            )
            
            # Verifica se è un evento di fine forte
            # WHISTLE_END è forte solo se valido (verificato con _should_end_rally_at_whistle)
            is_strong_end = (
                event.type == EventType.SCORE_CHANGE or
                event.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT) or
                (event.type == EventType.WHISTLE_END and 
                 self._should_end_rally_at_whistle(event.time, events)) or
                (event.type == EventType.MOTION_GAP and 
                 event.extra and event.extra.get("duration", 0) > 3.0)
            )

            # Gestisci attività
            if is_activity:
                if current_start is None:
                    # Inizia un nuovo segmento
                    current_start = event_time
                    # Determina side se possibile
                    if event.type == EventType.SERVE_START and event.extra and "side" in event.extra:
                        rally_side = event.extra["side"]
                    elif event.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT):
                        rally_side = "left" if event.type == EventType.HIT_LEFT else "right"
                    elif event.type in (EventType.BALL_CROSS_NET, EventType.BALL_SERVE) and event.extra:
                        # Prova a estrarre side da eventi palla (se presente in extra)
                        if "direction" in event.extra:
                            direction = event.extra["direction"]
                            if "left" in direction.lower():
                                rally_side = "left"
                            elif "right" in direction.lower():
                                rally_side = "right"
                
                last_activity_time = event_time
                
                # Se WHISTLE_END valido, lo trattiamo anche come strong end
                if event.type == EventType.WHISTLE_END:
                    if self._should_end_rally_at_whistle(event_time, events):
                        strong_end_candidate = event_time

            # Gestisci eventi di fine forti
            if is_strong_end and current_start is not None:
                # Aggiorna strong_end_candidate se questo è più forte o più recente
                if strong_end_candidate is None or event_time > strong_end_candidate:
                    strong_end_candidate = event_time
                    # Chiudi immediatamente il segmento per eventi di fine forti
                    if event.type in (EventType.SCORE_CHANGE, EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT):
                        close_current_segment()
                        continue
                    # Chiudi anche per MOTION_GAP lungo (>3s)
                    elif event.type == EventType.MOTION_GAP and event.extra and event.extra.get("duration", 0) > 3.0:
                        close_current_segment()
                        continue

            # Verifica gap di inattività
            if current_start is not None and last_activity_time is not None:
                gap_since_activity = event_time - last_activity_time
                segment_duration = event_time - current_start
                
                # Chiudi se gap troppo grande o durata massima superata
                if gap_since_activity > gap_threshold or segment_duration > cfg.max_rally_duration:
                    close_current_segment()

        # Chiudi l'ultimo segmento se ancora aperto
        if current_start is not None:
            close_current_segment()

        # Applica i nuovi helper per migliorare la qualità dei rally
        refined_rallies: List[Rally] = []
        
        for macro_rally in rallies:
            # 1) Split sui segmenti con più battute
            pieces = self._split_on_serves_if_needed(macro_rally, events)
            
            for seg in pieces:
                # 2) Refine start/end con serve/whistle/motion gap
                seg = self._refine_segment_boundaries(seg, events)
                
                # 3) Validazione "no serve no rally" + durata
                if not self._is_valid_rally_segment(seg, events):
                    continue
                
                refined_rallies.append(seg)
        
        # Filtra duplicati e sovrapposizioni
        return self._filter_overlapping_rallies(refined_rallies)

    def _filter_overlapping_rallies(self, rallies: List[Rally]) -> List[Rally]:
        """Filtra rally che si sovrappongono troppo, mantenendo solo quello più lungo."""
        if not rallies:
            return []

        # Ordina per tempo di inizio
        rallies_sorted = sorted(rallies, key=lambda r: r.start)
        filtered: List[Rally] = []

        for rally in rallies_sorted:
            # Verifica sovrapposizione con rally esistenti
            overlaps = False
            for existing_rally in filtered:
                # Calcola sovrapposizione
                overlap_start = max(rally.start, existing_rally.start)
                overlap_end = min(rally.end, existing_rally.end)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    # Se sovrapposizione > 50% della durata del rally più corto, è un duplicato (era 30%)
                    min_duration = min(
                        rally.end - rally.start,
                        existing_rally.end - existing_rally.start
                    )
                    if overlap_duration > min_duration * 0.5:
                        overlaps = True
                        # Se questo rally è più lungo, sostituisci quello esistente
                        if (rally.end - rally.start) > (existing_rally.end - existing_rally.start):
                            filtered.remove(existing_rally)
                            overlaps = False
                        break

            if not overlaps:
                filtered.append(rally)

        # Verifica distanza minima tra rally consecutivi
        final_filtered: List[Rally] = []
        for rally in filtered:
            if not final_filtered:
                final_filtered.append(rally)
                continue

            last_rally = final_filtered[-1]
            gap = rally.start - last_rally.end

            # Gap deve essere >= 0 (no sovrapposizioni) e >= min_rally_gap
            if gap >= self.cfg.min_rally_gap:
                final_filtered.append(rally)
            elif gap >= 0:
                # Gap positivo ma troppo piccolo: scegli il rally più lungo
                if (rally.end - rally.start) > (last_rally.end - last_rally.start):
                    final_filtered.pop()
                    final_filtered.append(rally)

        return final_filtered

        # TODO: split interno con GAP (MOTION_GAP) se vuoi portare dentro anche EventType.MOTION_GAP

    def _filter_and_deduplicate_rallies(self, rallies: List[Rally]) -> List[Rally]:
        """
        Filtra e deduplica rally rimuovendo:
        - Rally troppo corti o troppo lunghi
        - Rally troppo vicini nel tempo
        - Rally che si sovrappongono troppo
        """
        if not rallies:
            return []

        cfg = self.cfg
        filtered: List[Rally] = []

        # Ordina per tempo di inizio
        rallies_sorted = sorted(rallies, key=lambda r: r.start)

        for rally in rallies_sorted:
            duration = rally.end - rally.start

            # Filtro 1: durata minima e massima
            if duration < cfg.min_rally_duration:
                continue
            if duration > cfg.max_rally_duration:
                continue

            # Filtro 2: verifica distanza minima dal rally precedente
            if filtered:
                last_rally = filtered[-1]
                gap = rally.start - last_rally.end
                if gap < cfg.min_rally_gap:
                    # Rally troppo vicino: scegli quello con durata maggiore
                    if duration > (last_rally.end - last_rally.start):
                        # Il nuovo rally è più lungo: rimuovi quello precedente
                        filtered.pop()
                    else:
                        # Il precedente è più lungo: salta questo
                        continue

            # Filtro 3: verifica sovrapposizione con rally esistenti
            overlaps = False
            for existing_rally in filtered:
                # Calcola sovrapposizione
                overlap_start = max(rally.start, existing_rally.start)
                overlap_end = min(rally.end, existing_rally.end)
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    # Se sovrapposizione > 50% della durata del rally più corto, è un duplicato
                    min_duration = min(duration, existing_rally.end - existing_rally.start)
                    if overlap_duration > min_duration * 0.5:
                        overlaps = True
                        # Se questo rally è più lungo, sostituisci quello esistente
                        if duration > (existing_rally.end - existing_rally.start):
                            filtered.remove(existing_rally)
                            overlaps = False
                        break

            if not overlaps:
                filtered.append(rally)

        return filtered



__all__ = [
    "HeadCoach",
    "HeadCoachConfig",
]

