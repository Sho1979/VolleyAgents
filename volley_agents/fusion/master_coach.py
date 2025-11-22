"""
MasterCoach: agente finale che combina tutti gli agenti per analisi completa della partita.

Estende HeadCoach aggiungendo:
- Integrazione con GameState
- Arricchimento rally con informazioni da tutti gli agenti
- Validazione secondo regole FIPAV
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from volley_agents.core.event import Event, EventType
from volley_agents.core.game_state import GameState
from volley_agents.core.rally import Rally
from volley_agents.core.timeline import Timeline
from volley_agents.fusion.head_coach import HeadCoach, HeadCoachConfig
# RallyOrchestrator disabilitato - ritorno a logica Analisi 1
# from volley_agents.fusion.rally_orchestrator import RallyOrchestrator, OrchestrationDecision, InformationNeed
from volley_agents.fusion.voting_system import VotingSystem, VotingResult
from volley_agents.fusion.rules_fipav import (
    is_valid_rally_duration,
    is_valid_serve_timing,
    update_rotation_after_point,
    validate_rally_touches,
    MAX_TOUCHES_PER_SIDE,
)


@dataclass
class MasterCoachConfig:
    """Configurazione per MasterCoach."""

    # Configurazione HeadCoach
    head_coach_config: Optional[HeadCoachConfig] = None

    # Validazione regole FIPAV
    validate_rules: bool = True
    min_rally_duration: float = 2.5  # ridotto per catturare rally più brevi (era 3.5)
    max_rally_duration: float = 45.0  # durata massima assoluta (secondi)
    # Durata massima per un singolo rally prima di considerare split automatico
    max_rally_duration_before_split: float = 15.0  # se >15s e ci sono più serve, splitta
    min_rally_gap: float = 0.8  # ridotto per permettere rally più vicini (era 1.5)
    
    # Parametri per estensione fine rally
    tail_min: float = 0.7  # minimo tempo dopo ultimo hit (secondi)
    post_roll: float = 1.5  # coda extra per vedere palla che cade e mini-esultanza (secondi)
    
    # Parametri per regola "1 serve = 1 rally"
    enforce_one_serve_per_rally: bool = True  # se True, splitta rally con più serve
    serve_confidence_threshold: float = 0.7  # confidence minima per considerare serve "forte"
    serve_max_distance_from_start: float = 0.7  # massima distanza serve da start (secondi)
    
    # Parametri per durata minima dinamica (rally brevi ma con forte consensus)
    dynamic_min_duration: bool = True  # se True, riduce min_dur per rally con forte consensus
    strong_consensus_min_duration: float = 2.0  # min_dur ridotto per rally con forte consensus
    # Consensus considerato "forte" se AudioAgent + ServeAgent + (ScoreboardAgent o TouchSequenceAgent)
    # hanno tutti confidence > strong_consensus_threshold
    strong_consensus_threshold: float = 0.7
    
    # Parametri per split e filtro finale
    min_dur_split: float = 2.0  # durata minima per accettare un sotto-rally dopo split
    min_dur_final: float = 2.8  # durata minima per rally definitivo nella lista finale
    # Filtro finale obbligatorio: serve_count == 1 e serve_time entro serve_max_distance_from_start da start
    enforce_exactly_one_serve: bool = True  # se True, scarta rally con serve != 1
    reject_rallies_without_serve: bool = True  # se True, scarta rally con serve_count == 0
    
    # Feature flag per Level B/C splitting (disabilitato di default)
    enable_serve_splitting: bool = False  # se True, abilita Level B serve→punto splitting
    
    # Parametri per pipeline serve→punto (Level B) - usati solo se enable_serve_splitting=True
    pre_roll_s: float = 0.4  # padding prima del serve (secondi)
    post_roll_s: float = 1.3  # padding dopo fine punto (secondi)
    min_dur_hard_s: float = 2.0  # durata minima assoluta (secondi)
    min_dur_soft_s: float = 2.5  # durata minima "morbida" (se >= 2.0 ma < 2.5, richiede forte consensus)
    serve_conf_min: float = 0.7  # confidence minima per considerare serve valido
    serve_to_start_max_offset_s: float = 0.7  # massimo offset serve da start (secondi)
    serve_window_min_s: float = 1.6  # minimo tempo dopo serve prima di cercare fine (secondi)
    serve_window_max_s: float = 12.0  # massimo tempo dopo serve per cercare fine (secondi)
    
    # Parametri per post-processing voting-based rallies
    vote_cluster_threshold: float = 0.3  # distanza per clustering voti (secondi)
    min_rally_duration_voted: float = 2.0  # durata minima per rally votato (secondi)
    serve_snap_window: float = 1.0  # finestra per allineare start al serve (secondi)
    pre_roll_serve: float = 0.3  # padding prima del serve quando si allinea (secondi)


class MasterCoach:
    """
    Agente finale che combina HeadCoach con altri agenti per analisi completa.

    Responsabilità:
    - Usa HeadCoach per segmentare i rally
    - Arricchisce i rally con informazioni da altri agenti
    - Aggiorna GameState
    - Valida secondo regole FIPAV
    """

    def __init__(
        self,
        cfg: Optional[MasterCoachConfig] = None,
        initial_state: Optional[GameState] = None,
        enable_logging: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = cfg or MasterCoachConfig()
        self.head_coach = HeadCoach(cfg=self.cfg.head_coach_config or HeadCoachConfig())
        self.game_state = initial_state or GameState()
        self.voting_system = VotingSystem(enable_logging=enable_logging, log_callback=log_callback)
        # RallyOrchestrator disabilitato - ritorno a logica Analisi 1
        # self.rally_orchestrator = RallyOrchestrator(enable_logging=enable_logging, log_callback=log_callback)
        self.enable_logging = enable_logging
        self.log_callback = log_callback

    def _log(self, message: str):
        """Log interno."""
        if self.enable_logging:
            if self.log_callback:
                self.log_callback(message)
            else:
                print(message)

    def analyze_game(self, timeline: Timeline) -> List[Rally]:
        """
        Analizza la partita completa usando Timeline e produce rally basati su HeadCoach.
        
        Design originale (Analisi 1):
        1. Usa SOLO HeadCoach per costruire i rally grezzi (score-first o audio+motion)
        2. Applica filtro di deduplica/distanza per rimuovere overlap
        3. Arricchisci i rally con informazioni da altri agenti
        4. Aggiorna GameState

        Args:
            timeline: Timeline con tutti gli eventi

        Returns:
            Lista di Rally (HeadCoach-based, come Analisi 1)
        """
        # Feature flag: se serve splitting è abilitato, usa la vecchia logica Level B/C
        if self.cfg.enable_serve_splitting:
            return self._analyze_game_with_serve_splitting(timeline)

        # APPROCCIO ANALISI 1: HeadCoach-based (design originale)
        
        # 1) Usa SOLO HeadCoach per costruire i rally grezzi
        # HeadCoach usa score-first se SCORE_CHANGE è affidabile, altrimenti audio+motion
        raw_rallies = self.head_coach.build_rallies(timeline)
        
        if self.enable_logging:
            self._log(f"📊 HeadCoach: {len(raw_rallies)} rally grezzi generati")

        # 2) Applica filtro di deduplica/distanza (rimuove overlap e rally troppo vicini)
        cleaned_rallies = self._filter_and_deduplicate_rallies(raw_rallies, timeline)
        
        if self.enable_logging:
            self._log(f"🧹 Dopo deduplica: {len(cleaned_rallies)} rally")

        # 3) Arricchisci i rally con informazioni da altri agenti
        enriched_rallies = []
        for rally in cleaned_rallies:
            enriched = self._enrich_rally(rally, timeline)
            if enriched is not None:
                enriched_rallies.append(enriched)
            else:
                enriched_rallies.append(rally)

        # 4) Log dettagliato
        if self.enable_logging:
            self._log_rally_details(enriched_rallies, timeline)

        # 5) Aggiorna GameState
        self._update_game_state(enriched_rallies, timeline)

        if self.enable_logging:
            self._log(f"\n✅ MasterCoach: {len(enriched_rallies)} rally validati (HeadCoach-based)")

        return enriched_rallies

    def _analyze_game_with_serve_splitting(self, timeline: Timeline) -> List[Rally]:
        """
        Vecchia logica Level B/C con serve splitting (usata solo se enable_serve_splitting=True).
        
        DEPRECATED: Questo metodo mantiene la vecchia logica per retrocompatibilità.
        """
        # Implementazione legacy Level B/C (mantenuta per retrocompatibilità)
        # ... [codice esistente Level B/C] ...
        # Per ora ritorna lista vuota se non implementato
        if self.enable_logging:
            self._log("⚠️ Serve splitting abilitato ma non implementato completamente")
        return []

    def _extract_voted_starts_and_ends(self, timeline: Timeline) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Estrae start/end clusters usando VotingSystem (logica Analisi 1 - quasi perfetto).
        
        START detection (ANALISI 1):
        - SOLO SERVE_START (confidence >= 0.60)
        - WHISTLE_START, HIT, REF_SERVE_READY/RELEASE DISABILITATI (troppi falsi positivi)
        - MIN_START_GAP_S = 2.0s (rilassato per U14/U16 con punti veloci)
        - Clustering start events entro 1.2s (mantiene primo tempo)
        
        END detection (ANALISI 1):
        - Priorità esatta: SCORE_CHANGE > REF_POINT > WHISTLE_END > MOTION_GAP (long > 3s)
        - TouchSequenceAgent DISABILITATO (troppi falsi positivi)
        - Clustering end events entro 1.6s (mantiene ultimo tempo)
        
        Args:
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Tuple (start_clusters, end_clusters): liste di (first_time, last_time) per cluster
        """
        events = timeline.sorted()
        
        # ANALISI 1: START detection - SOLO SERVE_START con MIN_START_GAP_S
        start_votes: List[Tuple[float, VotingResult]] = []
        window_size = 1.0
        MIN_START_GAP_S = 2.0  # gap minimo tra start (rilassato da 3.0s per U14/U16 con punti veloci)
        last_start_time: Optional[float] = None
        
        for event in events:
            # ANALISI 1: Solo SERVE_START (WHISTLE_START disabilitato in VotingSystem)
            if event.type != EventType.SERVE_START:
                continue
            
            # Confidence minima >= 0.60 (già controllato in VotingSystem, ma doppio check)
            if event.confidence < 0.60:
                continue
            
            # ENFORCE MIN_START_GAP_S: ignora start troppo vicini
            if last_start_time is not None:
                gap_from_last = event.time - last_start_time
                if gap_from_last < MIN_START_GAP_S:
                    # Ignora questo start (troppo vicino al precedente)
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Start ignorato: troppo vicino ({gap_from_last:.2f}s < {MIN_START_GAP_S:.1f}s) "
                            f"dal precedente SERVE_START@{last_start_time:.2f}s"
                        )
                    continue
            
            t_start_window = max(0, event.time - window_size)
            t_end_window = event.time + window_size
            
            vote_start = self.voting_system.vote_rally_start(
                events,
                time_window=(t_start_window, t_end_window),
            )
            if vote_start:
                start_votes.append((vote_start.time, vote_start))
                last_start_time = event.time  # Aggiorna ultimo start accettato
        
        # Clustering start: entro 1.2s (mantiene primo tempo)
        start_clusters = self._cluster_votes(start_votes, threshold=1.2, use_first=True)
        
        # ANALISI 1: END detection - priorità precisa (SCORE_CHANGE > REF_POINT > WHISTLE_END > MOTION_GAP)
        end_votes: List[Tuple[float, VotingResult]] = []
        
        for event in events:
            # ANALISI 1: Priorità esatta come richiesto
            # 1. SCORE_CHANGE (priorità massima)
            # 2. REF_POINT_LEFT / REF_POINT_RIGHT
            # 3. WHISTLE_END
            # 4. MOTION_GAP lungo (> 3s) - priorità minima
            
            if event.type == EventType.SCORE_CHANGE:
                pass  # valido - priorità 1
            elif event.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT):
                pass  # valido - priorità 2
            elif event.type == EventType.WHISTLE_END:
                pass  # valido - priorità 3
            elif event.type == EventType.MOTION_GAP:
                # Solo MOTION_GAP lungo (> 3s) - priorità 4
                if not (event.extra and event.extra.get("duration", 0) > 3.0):
                    continue
            else:
                continue  # Ignora altri eventi
            
            t_start_window = max(0, event.time - window_size)
            t_end_window = event.time + window_size
            
            vote_end = self.voting_system.vote_rally_end(
                events,
                time_window=(t_start_window, t_end_window),
            )
            if vote_end:
                end_votes.append((vote_end.time, vote_end))
        
        # Clustering end: entro 1.6s (mantiene ultimo tempo)
        end_clusters = self._cluster_votes(end_votes, threshold=1.6, use_first=False)
        
        return start_clusters, end_clusters

    def _cluster_votes(self, votes: List[Tuple[float, VotingResult]], threshold: float, use_first: bool = True) -> List[Tuple[float, float]]:
        """
        Clustering dei voti vicini nel tempo.
        
        Se due voti sono a distanza < threshold, vengono clusterizzati:
        - Per start (use_first=True): mantieni (primo_tempo, ultimo_tempo)
        - Per end (use_first=False): mantieni (primo_tempo, ultimo_tempo)
        
        Args:
            votes: Lista di (time, VotingResult)
            threshold: Soglia di distanza per clustering (secondi)
            use_first: Se True, usa primo tempo per rappresentare cluster (start), altrimenti ultimo (end)
            
        Returns:
            Lista di (first_time, last_time) per ogni cluster
        """
        if not votes:
            return []
        
        # Ordina per tempo
        sorted_votes = sorted(votes, key=lambda x: x[0])
        clusters: List[Tuple[float, float]] = []
        
        current_cluster = [sorted_votes[0]]
        
        for vote_time, vote_result in sorted_votes[1:]:
            last_time = current_cluster[-1][0]
            gap = vote_time - last_time
            
            if gap < threshold:
                # Aggiungi al cluster corrente
                current_cluster.append((vote_time, vote_result))
            else:
                # Finalizza cluster precedente
                cluster_first = current_cluster[0][0]
                cluster_last = current_cluster[-1][0]
                clusters.append((cluster_first, cluster_last))
                current_cluster = [(vote_time, vote_result)]
        
        # Finalizza ultimo cluster
        if current_cluster:
            cluster_first = current_cluster[0][0]
            cluster_last = current_cluster[-1][0]
            clusters.append((cluster_first, cluster_last))
        
        return clusters

    def _pair_starts_and_ends(self, start_clusters: List[Tuple[float, float]], end_clusters: List[Tuple[float, float]], timeline: Timeline) -> List[Rally]:
        """
        Abbina start/end clusters in Rally objects.
        
        Per ogni start cluster:
        - Prendi il primo tempo (earliest) del cluster
        - Trova il primo end cluster dopo quello start
        - Prendi l'ultimo tempo (latest) di quell'end cluster
        
        Args:
            start_clusters: Lista di (first_time, last_time) per start clusters
            end_clusters: Lista di (first_time, last_time) per end clusters
            timeline: Timeline per contesto
            
        Returns:
            Lista di Rally abbinati
        """
        rallies: List[Rally] = []
        
        # Ordina clusters per primo tempo
        start_clusters_sorted = sorted(start_clusters, key=lambda x: x[0])
        end_clusters_sorted = sorted(end_clusters, key=lambda x: x[0])
        
        if not start_clusters_sorted or not end_clusters_sorted:
            return []
        
        # ANALISI 1: Pairing semplice - start[i] con primo end > start[i]
        # Stop qui - NON cercare di fare 1:1 tutti gli start/end
        end_idx = 0
        for start_first, start_last in start_clusters_sorted:
            # Usa il primo tempo dello start cluster
            start_time = start_first
            
            # Verifica durata minima START -> END (almeno 1.5s come richiesto)
            MIN_RALLY_DURATION_FOR_PAIRING = 1.5
            
            # Trova il primo end cluster dopo questo start
            while end_idx < len(end_clusters_sorted):
                end_first, end_last = end_clusters_sorted[end_idx]
                # Verifica se end cluster inizia dopo start (con margine minimo)
                if end_first > start_time + MIN_RALLY_DURATION_FOR_PAIRING:
                    # Usa l'ultimo tempo dell'end cluster
                    end_time = end_last
                    
                    # Verifica durata minima (dopo pairing)
                    duration = end_time - start_time
                    if duration >= self.cfg.min_rally_duration:
                        rallies.append(Rally(
                            start=start_time,
                            end=end_time,
                            side="unknown"
                        ))
                    end_idx += 1  # Usa ogni end solo una volta
                    break  # Stop qui - passa al prossimo start
                end_idx += 1
        
        return rallies

    def _filter_and_deduplicate_rallies(self, rallies: List[Rally], timeline: Timeline) -> List[Rally]:
        """
        Filtra e deduplica rally rimuovendo overlap e rally troppo vicini.
        
        Post-processing leggero dopo HeadCoach:
        - Rimuove rally troppo corti (< min_rally_duration)
        - Rimuove rally troppo lunghi (> max_rally_duration)
        - Rimuove rally troppo vicini (< min_rally_gap)
        - Rimuove rally che si sovrappongono troppo (> 50% overlap)
        
        Args:
            rallies: Lista di rally grezzi da HeadCoach
            timeline: Timeline con tutti gli eventi (opzionale, per contesto)
            
        Returns:
            Lista di rally filtrati e deduplicati
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
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato: durata troppo corta "
                        f"({duration:.2f}s < {cfg.min_rally_duration:.2f}s) [{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            if duration > cfg.max_rally_duration:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato: durata troppo lunga "
                        f"({duration:.2f}s > {cfg.max_rally_duration:.2f}s) [{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            # Filtro 2: verifica distanza minima dal rally precedente
            if filtered:
                last_rally = filtered[-1]
                gap = rally.start - last_rally.end
                
                # Se gap negativo (sovrapposizione) o troppo piccolo
                if gap < 0 or (gap >= 0 and gap < getattr(cfg, 'min_rally_gap', 0.5)):
                    # Scegli quello con durata maggiore
                    if duration > (last_rally.end - last_rally.start):
                        # Il nuovo rally è più lungo: rimuovi quello precedente
                        filtered.pop()
                        if self.enable_logging:
                            self._log(
                                f"🔄 Rally sostituito: gap={gap:.2f}s - "
                                f"mantenuto rally più lungo [{rally.start:.2f}-{rally.end:.2f}s]"
                            )
                    else:
                        # Il precedente è più lungo: salta questo
                        if self.enable_logging:
                            self._log(
                                f"⚠️ Rally rimosso: gap={gap:.2f}s - "
                                f"mantenuto rally precedente [{last_rally.start:.2f}-{last_rally.end:.2f}s]"
                            )
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
                            if self.enable_logging:
                                self._log(
                                    f"🔄 Rally sostituito: sovrapposizione {overlap_duration:.2f}s - "
                                    f"mantenuto rally più lungo [{rally.start:.2f}-{rally.end:.2f}s]"
                                )
                        else:
                            if self.enable_logging:
                                self._log(
                                    f"⚠️ Rally rimosso: sovrapposizione {overlap_duration:.2f}s - "
                                    f"mantenuto rally esistente [{existing_rally.start:.2f}-{existing_rally.end:.2f}s]"
                                )
                        break
            
            if not overlaps:
                filtered.append(rally)
        
        return filtered

    def _filter_rallies_like_analisi1(self, rallies: List[Rally], timeline: Timeline) -> List[Rally]:
        """
        Filtri storici di Analisi 1 (sistema quasi perfetto).
        
        Applica esattamente i filtri che producevano 18 rally finali di alta qualità:
        1. Filtro "serve all'inizio": scarta rally senza SERVE_START all'inizio (finestra ~0.7-1.0s)
        2. Filtro "fine non confermata": scarta senza SCORE_CHANGE/REF_POINT/long MOTION_GAP vicino alla fine
        3. Estensione fine con MOTION_GAP: estende fino al primo motion_gap lungo dopo fischio/attacco
        4. Filtro "too close / overlap": rimuove rally troppo vicini (gap negativo o < 0.5s)
        5. Filtro "durata minima": scarta rally < min_dur_final (~2.8s)
        
        Args:
            rallies: Lista di rally candidati da filtrare
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally filtrati (come Analisi 1)
        """
        if not rallies:
            return []
        
        events = timeline.sorted()
        cfg = self.cfg
        filtered: List[Rally] = []
        
        # Estrai eventi chiave
        serves = [e for e in events if e.type == EventType.SERVE_START]
        scores = [e for e in events if e.type == EventType.SCORE_CHANGE]
        ref_points = [e for e in events if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)]
        whistles_end = [e for e in events if e.type == EventType.WHISTLE_END]
        long_gaps = [
            e for e in events
            if e.type == EventType.MOTION_GAP
            and e.extra
            and e.extra.get("duration", 0) > 3.0
        ]
        
        # ANALISI 1: Durata minima per accettare WHISTLE_END da solo
        MIN_DUR_WHISTLE_ONLY = 4.0  # se rally >= 4.0s, accetta WHISTLE_END anche senza SCORE/REF/GAP
        
        for rally in rallies:
            # FILTRO 1: Serve all'inizio (finestra -0.7s a +1.0s dallo start)
            # Il serve può essere leggermente prima dello start del rally
            has_serve_at_start = any(
                rally.start - 0.7 <= s.time <= rally.start + 1.0
                for s in serves
            )
            if not has_serve_at_start:
                if self.enable_logging:
                    self._log(
                        f"⚠️ Rally scartato: nessun SERVE_START all'inizio "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            # FILTRO 2: Fine non confermata (ANALISI 1 - rilassato con WHISTLE_END)
            # Cerca SCORE_CHANGE, REF_POINT, long MOTION_GAP vicino alla fine (finestra 3s prima e 2s dopo)
            end_window_start = rally.end - 3.0
            end_window_end = rally.end + 2.0
            
            # Priorità: SCORE_CHANGE > REF_POINT > long MOTION_GAP
            score_near_end = any(
                end_window_start <= s.time <= end_window_end
                for s in scores
            )
            ref_near_end = any(
                end_window_start <= r.time <= end_window_end
                for r in ref_points
            )
            gap_near_end = any(
                end_window_start <= g.time <= end_window_end
                for g in long_gaps
            )
            whistle_near_end = any(
                end_window_start <= w.time <= end_window_end
                for w in whistles_end
            )
            
            # NUOVA LOGICA "ANALISI 1 style"
            has_end_confirmation = False
            if score_near_end or ref_near_end or gap_near_end:
                # Segnali forti: SCORE_CHANGE, REF_POINT, o MOTION_GAP lungo
                has_end_confirmation = True
            elif whistle_near_end:
                # Fallback: accetta WHISTLE_END da solo se rally ha durata sufficiente
                duration = rally.end - rally.start
                if duration >= MIN_DUR_WHISTLE_ONLY:
                    has_end_confirmation = True
            
            if not has_end_confirmation:
                if self.enable_logging:
                    self._log(
                        f"⚠️ Rally scartato: nessuna conferma fine trovata "
                        f"(WHISTLE_END/SCORE_CHANGE/REF_POINT/MOTION_GAP) [{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            # FILTRO 3: Estensione fine con MOTION_GAP lungo
            # Estendi fino al primo motion_gap lungo dopo la fine originale (finestra 10s)
            original_end = rally.end
            motion_gaps_after = [
                g for g in long_gaps
                if rally.end < g.time <= rally.end + 10.0
            ]
            if motion_gaps_after:
                # Prendi il primo MOTION_GAP lungo dopo la fine
                first_gap = min(motion_gaps_after, key=lambda g: g.time)
                extended_end = first_gap.time
                rally.end = extended_end
                if self.enable_logging:
                    self._log(
                        f"✅ Rally esteso fino a MOTION_GAP lungo: "
                        f"[{rally.start:.2f}-{original_end:.2f}s] -> [{rally.start:.2f}-{extended_end:.2f}s]"
                    )
            
            # FILTRO 4: Durata minima (dopo estensione)
            duration = rally.end - rally.start
            min_dur_final = cfg.min_dur_final if hasattr(cfg, 'min_dur_final') else 2.8
            if duration < min_dur_final:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato (filtro finale): durata troppo corta "
                        f"({duration:.2f}s < {min_dur_final:.2f}s) [{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            # Rally passato tutti i filtri base
            filtered.append(rally)
        
        # FILTRO 5: "Too close / overlap" - rimuove rally troppo vicini o sovrapposti
        # Ordina per tempo di inizio
        filtered_sorted = sorted(filtered, key=lambda r: r.start)
        final: List[Rally] = []
        gap_min = 0.5  # gap minimo tra rally (s)
        
        for rally in filtered_sorted:
            if not final:
                final.append(rally)
            else:
                prev = final[-1]
                gap = rally.start - prev.end
                
                # Se gap negativo (sovrapposizione) o troppo piccolo, rimuovi questo rally
                if gap < gap_min:
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Rally rimosso: troppo vicino ({gap:.2f}s gap) - "
                            f"mantenuto rally precedente [{prev.start:.2f}-{prev.end:.2f}s]"
                        )
                    continue
                
                final.append(rally)
        
        return final

    def _postprocess_minimally(self, rallies: List[Rally], timeline: Timeline) -> List[Rally]:
        """
        Post-processing minimale (logica Analisi 1).
        
        Applica solo:
        1. Drop rallies < 3s
        2. Extend end usando SCORE_CHANGE / REF_POINT se disponibile (senza Orchestrator)
        
        NO splitting by serve.
        NO enforce serve_count==1.
        
        Args:
            rallies: Lista di rally da pulire
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally puliti
        """
        if not rallies:
            return []
        
        events = timeline.sorted()
        cleaned: List[Rally] = []
        
        for rally in rallies:
            # 1. Drop rallies < 3s
            duration = rally.end - rally.start
            if duration < 3.0:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato: durata {duration:.2f}s < 3.0s "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            # 2. Extend end usando SCORE_CHANGE / REF_POINT se disponibile (senza Orchestrator)
            # Cerca eventi di fine dopo rally.end entro 3s
            after_rally = [
                e for e in events
                if rally.end < e.time <= rally.end + 3.0
            ]
            
            # Priorità: SCORE_CHANGE > REF_POINT
            score_after = next(
                (e for e in after_rally if e.type == EventType.SCORE_CHANGE),
                None
            )
            if score_after:
                rally.end = score_after.time
                if self.enable_logging:
                    self._log(
                        f"🎯 End esteso a SCORE_CHANGE: end={score_after.time:.2f}s"
                    )
            else:
                ref_after = next(
                    (e for e in after_rally 
                     if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)),
                    None
                )
                if ref_after:
                    rally.end = ref_after.time
                    if self.enable_logging:
                        self._log(
                            f"🎯 End esteso a REF_POINT: end={ref_after.time:.2f}s"
                        )
            
            cleaned.append(rally)
        
        return cleaned

    def _build_voted_rallies_from_timeline(self, timeline: Timeline) -> List[Rally]:
        """
        Costruisce rally usando VotingSystem + HeadCoach.
        
        Combina i risultati di VotingSystem (voti per start/end) con HeadCoach
        per produrre una lista di rally candidati. Clustering dei voti per evitare
        duplicati vicini nel tempo.
        
        Args:
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di Rally candidati da VotingSystem/HeadCoach
        """
        events = timeline.sorted()
        
        # Usa VotingSystem per votare start/end
        rally_candidates = self._identify_rallies_with_voting(events)
        voting_rallies = []
        if rally_candidates:
            voting_rallies = self._rallies_from_voting(rally_candidates, events)
        
        # Usa HeadCoach come complemento/fallback
        hc_rallies = self.head_coach.build_rallies(timeline)
        
        # Combina voting + HeadCoach (dando priorità a voting se c'è consensus)
        if voting_rallies:
            # Se voting ha molti rally con consensus, preferisci voting
            high_quality_voting = [
                r for r in rally_candidates
                if r[0] and r[0].consensus and r[0].confidence > 0.7
            ]
            if len(high_quality_voting) >= 2:
                raw_rallies = self._merge_rally_lists(voting_rallies, hc_rallies)
            else:
                # HeadCoach ha priorità se voting non ha consensus
                raw_rallies = self._merge_rally_lists(hc_rallies, voting_rallies)
        else:
            raw_rallies = hc_rallies
        
        # Clustering: rimuovi rally molto vicini (< 0.3s) mantenendo solo il primo start e ultimo end
        clustered = self._cluster_close_rallies(raw_rallies, threshold=self.cfg.vote_cluster_threshold)
        
        return clustered

    def _cluster_close_rallies(self, rallies: List[Rally], threshold: float = 0.3) -> List[Rally]:
        """
        Clustering dei rally molto vicini nel tempo.
        
        Se due rally hanno start/end a distanza < threshold, vengono fusi:
        - start = primo start
        - end = ultimo end
        
        Args:
            rallies: Lista di rally da clusterizzare
            threshold: Soglia di distanza per clustering (secondi)
            
        Returns:
            Lista di rally clusterizzati
        """
        if not rallies:
            return []
        
        # Ordina per start
        sorted_rallies = sorted(rallies, key=lambda r: r.start)
        clustered: List[Rally] = []
        
        current_cluster = [sorted_rallies[0]]
        
        for rally in sorted_rallies[1:]:
            # Verifica se questo rally è vicino al cluster corrente
            last_in_cluster = current_cluster[-1]
            gap_to_start = rally.start - last_in_cluster.end
            gap_to_cluster_start = rally.start - current_cluster[0].start
            
            # Se è molto vicino (gap < threshold), aggiungi al cluster
            if gap_to_start < threshold or gap_to_cluster_start < threshold:
                current_cluster.append(rally)
            else:
                # Nuovo cluster: finalizza il precedente
                if len(current_cluster) == 1:
                    clustered.append(current_cluster[0])
                else:
                    # Fondo cluster: primo start, ultimo end
                    cluster_start = min(r.start for r in current_cluster)
                    cluster_end = max(r.end for r in current_cluster)
                    clustered.append(Rally(
                        start=cluster_start,
                        end=cluster_end,
                        side=current_cluster[0].side
                    ))
                current_cluster = [rally]
        
        # Finalizza ultimo cluster
        if current_cluster:
            if len(current_cluster) == 1:
                clustered.append(current_cluster[0])
            else:
                cluster_start = min(r.start for r in current_cluster)
                cluster_end = max(r.end for r in current_cluster)
                clustered.append(Rally(
                    start=cluster_start,
                    end=cluster_end,
                    side=current_cluster[0].side
                ))
        
        return clustered

    def _postprocess_voted_rallies(self, rallies: List[Rally], timeline: Timeline) -> List[Rally]:
        """
        Post-processing minimale sui rally votati.
        
        Applica solo:
        1. Drop fragments < 2.0s senza scoring evidence
        2. Snap start al serve più vicino (opzionale, allineamento)
        3. Extend end usando SCORE_CHANGE / REF_POINT (senza Orchestrator)
        
        Args:
            rallies: Lista di rally votati da pulire
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally puliti
        """
        if not rallies:
            return []
        
        cfg = self.cfg
        events = timeline.sorted()
        cleaned: List[Rally] = []
        
        for rally in rallies:
            # 1. Drop fragments < min_rally_duration_voted senza scoring evidence
            duration = rally.end - rally.start
            if duration < cfg.min_rally_duration_voted:
                # Verifica se c'è scoring evidence
                has_score, _ = self._has_scoring_evidence(timeline, rally.start, rally.end)
                
                if not has_score:
                    # Fragment troppo corto senza evidenza: scarta
                    if self.enable_logging:
                        self._log(
                            f"🚫 Fragment scartato: durata {duration:.2f}s < {cfg.min_rally_duration_voted:.2f}s "
                            f"senza scoring evidence [{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue
            
            # 2. Snap start al serve più vicino (opzionale, allineamento)
            serves = [
                e for e in events
                if e.type == EventType.SERVE_START
                and rally.start - cfg.serve_snap_window <= e.time <= rally.start + cfg.serve_snap_window
            ]
            
            if serves:
                # Trova serve più forte (confidence + vicinanza)
                best_serve = max(serves, key=lambda s: s.confidence - abs(s.time - rally.start) / 10.0)
                
                # Allinea start al serve (con pre_roll)
                new_start = best_serve.time - cfg.pre_roll_serve
                if new_start < rally.start:
                    rally.start = new_start
                    if self.enable_logging:
                        self._log(
                            f"📍 Start allineato al serve {best_serve.time:.2f}s: "
                            f"start={new_start:.2f}s"
                        )
            
            # 3. Extend end usando SCORE_CHANGE / REF_POINT (senza Orchestrator)
            # Cerca eventi di fine dopo rally.end entro 3s
            after_rally = [
                e for e in events
                if rally.end < e.time <= rally.end + 3.0
            ]
            
            # Priorità: SCORE_CHANGE > REF_POINT
            score_after = next(
                (e for e in after_rally if e.type == EventType.SCORE_CHANGE),
                None
            )
            if score_after:
                rally.end = score_after.time
                if self.enable_logging:
                    self._log(
                        f"🎯 End esteso a SCORE_CHANGE: end={score_after.time:.2f}s"
                    )
            else:
                ref_after = next(
                    (e for e in after_rally 
                     if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)),
                    None
                )
                if ref_after:
                    rally.end = ref_after.time
                    if self.enable_logging:
                        self._log(
                            f"🎯 End esteso a REF_POINT: end={ref_after.time:.2f}s"
                        )
            
            # Verifica durata finale dopo post-processing
            final_duration = rally.end - rally.start
            if final_duration >= cfg.min_rally_duration:
                cleaned.append(rally)
            else:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato dopo post-processing: durata {final_duration:.2f}s "
                        f"< {cfg.min_rally_duration:.2f}s"
                    )
        
        return cleaned

    def _build_macro_rallies(self, timeline: Timeline) -> List[Rally]:
        """
        LEVEL A: Costruisci macro-rally come container temporali.
        
        Questi macro-rally NON vengono esportati come rally finali.
        Sono solo container temporali usati per cercare serve→punto.
        
        Args:
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di macro-rally (container, non esportati)
        """
        events = timeline.sorted()

        # Usa la logica esistente per costruire macro-rally (HeadCoach + Voting)
        # Ma questi NON vengono esportati, sono solo container

        # Prova voting system
        rally_candidates = self._identify_rallies_with_voting(events)
        voting_rallies = []
        if rally_candidates and len(rally_candidates) >= 1:
            voting_rallies = self._rallies_from_voting(rally_candidates, events)

        # Usa HeadCoach come fallback o complemento
        hc_rallies = self.head_coach.build_rallies(timeline)

        # Combina voting + HeadCoach
        if voting_rallies:
            high_quality_voting = [
                r for r in rally_candidates
                if r[0] and r[0].consensus and r[0].confidence > 0.7
            ]
            if len(high_quality_voting) >= 2:
                raw_rallies = self._merge_rally_lists(voting_rallies, hc_rallies)
            else:
                raw_rallies = self._merge_rally_lists(hc_rallies, voting_rallies)
        else:
            raw_rallies = hc_rallies

        # Estendi macro-rally fino a conferma definitiva
        extended_rallies = self._orchestrate_rally_extensions_rigorous(raw_rallies, timeline)

        # Rimuovi duplicati e sovrapposizioni tra macro-rally
        macro_rallies = self._filter_final_rallies(extended_rallies)

        return macro_rallies

    def _extract_serves_in_macro(self, macro_rally: Rally, timeline: Timeline) -> List[Event]:
        """
        Estrae serve dentro un macro-rally.
        
        Args:
            macro_rally: Macro-rally container
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di SERVE_START eventi dentro il macro, ordinati per tempo
        """
        cfg = self.cfg
        events = timeline.sorted()
        
        # Estrai SERVE_START dentro il macro con confidence >= threshold
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_conf_min
            and macro_rally.start <= e.time <= macro_rally.end
        ]
        
        # Ordina per tempo
        serves.sort(key=lambda e: e.time)
        
        return serves

    def _build_serve_to_point_rally(
        self, serve: Event, macro_rally: Rally, timeline: Timeline
    ) -> Optional[Rally]:
        """
        LEVEL B: Costruisci un rally serve→punto per un singolo serve.
        
        Args:
            serve: SERVE_START event
            macro_rally: Macro-rally container che contiene questo serve
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Rally serve→punto o None se non valido
        """
        cfg = self.cfg
        events = timeline.sorted()
        
        serve_time = serve.time
        
        # Finestra per cercare fine punto
        t_min = serve_time + cfg.serve_window_min_s  # minimo 1.6s dopo serve
        t_max = min(macro_rally.end, serve_time + cfg.serve_window_max_s)  # massimo 12s o fine macro
        
        # Cerca eventi di fine punto dentro [t_min, t_max]
        # Priorità: SCORE_CHANGE > last attack + whistle > whistle_end > motion_gap
        
        # Priorità 1: SCORE_CHANGE / REF_POINT
        scores = [
            e for e in events
            if e.type == EventType.SCORE_CHANGE
            and t_min <= e.time <= t_max
        ]
        ref_points = [
            e for e in events
            if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
            and t_min <= e.time <= t_max
        ]
        
        end_candidate = None
        end_signal_type = None
        
        if scores:
            end_candidate = scores[0].time
            end_signal_type = "SCORE_CHANGE"
        elif ref_points:
            end_candidate = ref_points[0].time
            end_signal_type = "REF_POINT"
        else:
            # Priorità 2: last attack + whistle
            attacks = [
                e for e in events
                if e.type in (EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT)
                and serve_time <= e.time <= t_max
                and e.extra
                and e.extra.get("is_sequence_end", False)
            ]
            
            if attacks:
                last_attack = max(attacks, key=lambda e: e.time)
                # Cerca whistle_end poco dopo l'attacco (entro 2s)
                whistles_after = [
                    e for e in events
                    if e.type == EventType.WHISTLE_END
                    and last_attack.time < e.time <= last_attack.time + 2.0
                    and e.time <= t_max
                ]
                if whistles_after:
                    end_candidate = whistles_after[0].time
                    end_signal_type = "ATTACK+WHISTLE"
            
            # Priorità 3: whistle_end (se non trovato attack+whistle)
            if end_candidate is None:
                whistles = [
                    e for e in events
                    if e.type == EventType.WHISTLE_END
                    and t_min <= e.time <= t_max
                ]
                if whistles:
                    end_candidate = whistles[0].time
                    end_signal_type = "WHISTLE_END"
            
            # Priorità 4: motion_gap lungo (fallback)
            if end_candidate is None:
                gaps = [
                    e for e in events
                    if e.type == EventType.MOTION_GAP
                    and t_min <= e.time <= t_max
                    and e.extra
                    and e.extra.get("duration", 0) > 3.0
                ]
                if gaps:
                    end_candidate = gaps[0].time
                    end_signal_type = "MOTION_GAP"
        
        # Se non trovato fine punto, scarta questo serve
        if end_candidate is None:
            return None
        
        # Calcola start_clip e end_clip con padding
        start_clip = max(macro_rally.start, serve_time - cfg.pre_roll_s)
        end_clip = min(macro_rally.end, end_candidate + cfg.post_roll_s)
        
        # Crea rally candidato
        # Nota: Rally non ha campo extra, ma possiamo passare end_signal_type via logging
        rally = Rally(
            start=start_clip,
            end=end_clip,
            side=macro_rally.side,
        )
        
        # Verifica durata minima duale (hard/soft)
        duration = rally.end - rally.start
        
        if duration < cfg.min_dur_hard_s:
            # Durata troppo corta: scarta
            if self.enable_logging:
                self._log(
                    f"      ⚠️ Durata troppo corta ({duration:.2f}s < {cfg.min_dur_hard_s:.2f}s), scartato"
                )
            return None
        
        if cfg.min_dur_hard_s <= duration < cfg.min_dur_soft_s:
            # Durata in zona "soft": richiede forte consensus
            has_strong_consensus = self._check_strong_consensus(serve, end_candidate, end_signal_type, timeline)
            if not has_strong_consensus:
                # Non c'è forte consensus: scarta
                if self.enable_logging:
                    self._log(
                        f"      ⚠️ Durata soft ({duration:.2f}s) senza forte consensus, scartato"
                    )
                return None
        
        # Durata >= min_dur_soft_s: accetta
        # Log dettagli per debug (end_signal_type non può essere salvato in Rally, ma possiamo loggarlo)
        if self.enable_logging:
            self._log(
                f"      ✅ Rally valido: dur={duration:.2f}s, end_signal={end_signal_type}"
            )
        
        return rally

    def _check_strong_consensus(
        self, serve: Event, end_time: float, end_signal_type: str, timeline: Timeline
    ) -> bool:
        """
        Verifica se c'è forte consensus per un rally breve (2.0-2.5s).
        
        Forte consensus richiede:
        - Serve forte (confidence >= threshold)
        - Fine punto forte (SCORE_CHANGE, REF_POINT, o attack + whistle)
        - Whistle_end se non già incluso
        
        Args:
            serve: SERVE_START event
            end_time: Tempo fine punto
            end_signal_type: Tipo di segnale fine ("SCORE_CHANGE", "ATTACK+WHISTLE", ecc.)
            timeline: Timeline con tutti gli eventi
            
        Returns:
            True se c'è forte consensus
        """
        cfg = self.cfg
        
        # Verifica serve forte
        if serve.confidence < cfg.serve_conf_min:
            return False
        
        # Verifica fine punto forte
        if end_signal_type in ("SCORE_CHANGE", "REF_POINT", "ATTACK+WHISTLE"):
            return True
        
        # Per altri segnali, verifica se c'è whistle_end vicino
        events = timeline.sorted()
        whistles_near = [
            e for e in events
            if e.type == EventType.WHISTLE_END
            and abs(e.time - end_time) <= 1.0
            and e.confidence >= cfg.strong_consensus_threshold
        ]
        
        return len(whistles_near) > 0

    def _filter_serve_based_rallies(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Filtra rally serve-based: esattamente 1 serve, durata minima duale.
        
        Per ogni rally:
        - Conta serve dentro [start, end]
        - Se serve_count == 0: scarta
        - Se serve_count == 1: verifica durata e serve vicino a start
        - Se serve_count > 1: trim end_clip prima del secondo serve
        
        Args:
            rallies: Lista di rally serve-based da filtrare
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally filtrati con esattamente 1 serve
        """
        if not rallies:
            return []

        cfg = self.cfg
        events = timeline.sorted()

        # Estrai tutti i SERVE_START
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_conf_min
        ]
        serves.sort(key=lambda e: e.time)

        filtered: List[Rally] = []

        for rally in rallies:
            # Trova serve dentro questo rally
            serves_in_rally = [
                s for s in serves
                if rally.start <= s.time <= rally.end
            ]

            num_serves = len(serves_in_rally)

            # Scarta rally con serve_count == 0
            if num_serves == 0:
                if cfg.reject_rallies_without_serve:
                    if self.enable_logging:
                        self._log(
                            f"🚫 Rally scartato: serve_count=0 "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

            # Se serve_count > 1, trim end_clip prima del secondo serve
            if num_serves > 1:
                second_serve = serves_in_rally[1]
                # Trim end_clip a 0.3s prima del secondo serve
                new_end = second_serve.time - 0.3
                if new_end > rally.start:
                    rally.end = new_end
                    # Riconta serve dopo trim
                    serves_in_rally = [
                        s for s in serves
                        if rally.start <= s.time <= rally.end
                    ]
                    num_serves = len(serves_in_rally)
                    
                    if self.enable_logging:
                        self._log(
                            f"✂️ Rally trimato per rimuovere serve multipli: "
                            f"end={new_end:.2f}s (serve_count={num_serves})"
                        )

            # Verifica serve_count == 1 dopo eventuale trim
            if cfg.enforce_exactly_one_serve:
                if num_serves != 1:
                    if self.enable_logging:
                        self._log(
                            f"🚫 Rally scartato: serve_count={num_serves} != 1 "
                            f"(dopo trim) [{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

                # Verifica serve vicino a start
                serve = serves_in_rally[0]
                serve_distance = abs(serve.time - rally.start)
                if serve_distance > cfg.serve_to_start_max_offset_s:
                    # Serve troppo lontano: sposta start verso serve
                    rally.start = max(rally.start, serve.time - cfg.pre_roll_s)
                    serve_distance = abs(serve.time - rally.start)
                    
                    if serve_distance > cfg.serve_to_start_max_offset_s:
                        # Ancora troppo lontano: scarta
                        if self.enable_logging:
                            self._log(
                                f"🚫 Rally scartato: serve troppo distante da start "
                                f"({serve_distance:.2f}s > {cfg.serve_to_start_max_offset_s:.2f}s) "
                                f"[{rally.start:.2f}-{rally.end:.2f}s]"
                            )
                        continue

            # Verifica durata dopo eventuale trim/spostamento
            duration = rally.end - rally.start
            
            if duration < cfg.min_dur_hard_s:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato: durata troppo corta dopo trim "
                        f"({duration:.2f}s < {cfg.min_dur_hard_s:.2f}s) "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue

            # Rally valido: aggiungi alla lista filtrata
            filtered.append(rally)

        return filtered

    def _has_scoring_evidence(self, timeline: Timeline, start: float, end: float) -> Tuple[bool, List[Event]]:
        """
        Verifica se c'è evidenza di punteggio in un intervallo temporale.
        
        Evidenza di punteggio (scoring evidence):
        - SCORE_CHANGE (ScoreboardAgent) - evidenza più forte
        - REF_POINT_LEFT, REF_POINT_RIGHT (RefereeAgent) - evidenza molto forte
        - ATTACK_LEFT/RIGHT con is_sequence_end=True seguito da WHISTLE_END (pattern forte)
        
        Args:
            timeline: Timeline con tutti gli eventi
            start: Inizio intervallo (secondi)
            end: Fine intervallo incluso post_roll (secondi)
            
        Returns:
            Tuple (has_evidence, scoring_events):
            - has_evidence: True se c'è almeno un evento di punteggio
            - scoring_events: Lista di eventi di punteggio trovati
        """
        cfg = self.cfg
        events = timeline.sorted()
        
        # Estendi fine intervallo con post_roll per catturare eventi poco dopo la fine
        search_end = end  # già include post_roll_s dal chiamante
        
        # Eventi nell'intervallo [start, search_end]
        window_events = [
            e for e in events
            if start <= e.time <= search_end
        ]
        
        scoring_events: List[Event] = []
        
        # Priorità 1: SCORE_CHANGE (ScoreboardAgent)
        # Evento più forte: cambio punteggio rilevato dal tabellone
        score_changes = [
            e for e in window_events
            if e.type == EventType.SCORE_CHANGE
        ]
        if score_changes:
            scoring_events.extend(score_changes)
            # Se c'è SCORE_CHANGE, è evidenza sufficiente
            return (True, scoring_events)
        
        # Priorità 2: REF_POINT_LEFT, REF_POINT_RIGHT (RefereeAgent)
        # Evidenza molto forte: arbitro assegna punto
        ref_points = [
            e for e in window_events
            if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
        ]
        if ref_points:
            scoring_events.extend(ref_points)
            # Se c'è REF_POINT, è evidenza sufficiente
            return (True, scoring_events)
        
        # Priorità 3: Pattern "last attack + whistle_end"
        # TouchSequenceAgent marca l'ultimo attacco con is_sequence_end=True,
        # seguito da WHISTLE_END di AudioAgent
        attacks_with_end = [
            e for e in window_events
            if e.type in (EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT)
            and e.extra
            and e.extra.get("is_sequence_end", False)
        ]
        
        if attacks_with_end:
            # Cerca WHISTLE_END dopo l'ultimo attacco (entro 2s)
            last_attack = max(attacks_with_end, key=lambda e: e.time)
            whistles_after = [
                e for e in window_events
                if e.type == EventType.WHISTLE_END
                and last_attack.time < e.time <= last_attack.time + 2.0
            ]
            
            if whistles_after:
                # Pattern completo: last attack + whistle_end
                scoring_events.append(last_attack)
                scoring_events.extend(whistles_after)
                return (True, scoring_events)
        
        # Nessuna evidenza di punteggio trovata
        return (False, [])

    def _filter_scoring_rallies(
        self,
        rallies: List[Rally],
        timeline: Timeline,
    ) -> List[Rally]:
        """
        LEVEL C: Filtra rally per mantenere solo quelli con evidenza di punteggio.
        
        Per ogni rally serve-based:
        - Verifica se c'è almeno un evento di punteggio nell'intervallo [start, end + post_roll_s]
        - Mantiene solo rally con evidenza di cambio punteggio
        
        Evidenza di punteggio (scoring evidence):
        - SCORE_CHANGE (ScoreboardAgent)
        - REF_POINT_LEFT, REF_POINT_RIGHT (RefereeAgent)
        - ATTACK_LEFT/RIGHT con is_sequence_end=True seguito da WHISTLE_END (pattern)
        
        Args:
            rallies: Lista di rally serve-based (serve_count == 1)
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally con evidenza di punteggio
        """
        if not rallies:
            return []
        
        cfg = self.cfg
        filtered: List[Rally] = []
        
        for rally in rallies:
            # Verifica evidenza di punteggio nell'intervallo [start, end + post_roll_s]
            # end già include post_roll_s da _build_serve_to_point_rally, ma estendiamo
            # ulteriormente per sicurezza
            search_end = rally.end  # già include post_roll_s
            
            has_evidence, scoring_events = self._has_scoring_evidence(
                timeline, rally.start, search_end
            )
            
            if has_evidence:
                # Rally ha evidenza di punteggio: mantieni
                filtered.append(rally)
                if self.enable_logging:
                    scoring_types = [e.type.value for e in scoring_events]
                    self._log(
                        f"✅ Rally con evidenza punteggio: "
                        f"[{rally.start:.2f}-{rally.end:.2f}s] "
                        f"score_events={len(scoring_events)} ({', '.join(scoring_types[:2])})"
                    )
            else:
                # Nessuna evidenza di punteggio: scarta
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato (no evidenza punteggio): "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
        
        return filtered

    def _resolve_overlaps_serve_based(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Risolve overlap tra rally serve-based.
        
        Se due rally si sovrappongono:
        - Se provengono da serve diversi: trim end_i prima di serve_j
        - Se provengono dallo stesso serve: mantieni il migliore (score/whistle > motion_gap)
        
        Args:
            rallies: Lista di rally serve-based
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally senza overlap significativi
        """
        if not rallies:
            return []

        cfg = self.cfg
        events = timeline.sorted()

        # Estrai serve per identificare quale serve ha generato ogni rally
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_conf_min
        ]
        serves.sort(key=lambda e: e.time)

        # Ordina rally per start
        rallies_sorted = sorted(rallies, key=lambda r: r.start)
        resolved: List[Rally] = []

        for rally in rallies_sorted:
            # Trova serve che ha generato questo rally
            serves_in_rally = [
                s for s in serves
                if rally.start <= s.time <= rally.end
            ]
            rally_serve = serves_in_rally[0] if serves_in_rally else None

            overlaps = False
            for existing_rally in resolved:
                # Calcola sovrapposizione
                overlap_start = max(rally.start, existing_rally.start)
                overlap_end = min(rally.end, existing_rally.end)
                overlap_duration = overlap_end - overlap_start if overlap_end > overlap_start else 0

                # Se sovrapposizione > 1s, è un conflitto
                if overlap_duration > 1.0:
                    overlaps = True

                    # Trova serve che ha generato existing_rally
                    existing_serves = [
                        s for s in serves
                        if existing_rally.start <= s.time <= existing_rally.end
                    ]
                    existing_serve = existing_serves[0] if existing_serves else None

                    # Se serve diversi: trim end_i prima di serve_j
                    if rally_serve and existing_serve and rally_serve.time != existing_serve.time:
                        # Trim end di existing_rally prima del serve di rally
                        new_end = rally_serve.time - 0.3
                        if new_end > existing_rally.start:
                            existing_rally.end = new_end
                            # Verifica durata dopo trim
                            if existing_rally.end - existing_rally.start < cfg.min_dur_hard_s:
                                # Durata troppo corta: rimuovi existing_rally
                                resolved.remove(existing_rally)
                                resolved.append(rally)
                                overlaps = False
                                if self.enable_logging:
                                    self._log(
                                        f"🔄 Rally sostituito per overlap: "
                                        f"trim ha reso existing troppo corto, mantenuto nuovo "
                                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                                    )
                            else:
                                resolved.append(rally)
                                overlaps = False
                                if self.enable_logging:
                                    self._log(
                                        f"✂️ Rally trimato per overlap: "
                                        f"existing end={new_end:.2f}s, aggiunto nuovo "
                                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                                    )
                        else:
                            # Trim non possibile: mantieni existing
                            overlaps = True
                    else:
                        # Stesso serve o serve non identificati: mantieni il migliore
                        # Heuristica: preferisci rally con fine più vicina al serve
                        if rally_serve:
                            rally_distance = abs(rally.end - rally_serve.time)
                            existing_distance = abs(existing_rally.end - rally_serve.time) if existing_serve else float('inf')
                            
                            if rally_distance < existing_distance:
                                resolved.remove(existing_rally)
                                resolved.append(rally)
                                overlaps = False
                                if self.enable_logging:
                                    self._log(
                                        f"🔄 Rally sostituito per overlap (stesso serve): "
                                        f"mantenuto migliore [{rally.start:.2f}-{rally.end:.2f}s]"
                                    )
                            else:
                                overlaps = True
                    break

            if not overlaps:
                resolved.append(rally)

        return resolved

    def _enforce_exactly_one_serve_final(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Filtro di sicurezza finale: enforce serve_count == 1 su tutti i rally.
        
        Questo garantisce che anche se c'è un bug upstream, i rally finali
        hanno esattamente 1 serve. NON modifica i rally, solo filtra.
        
        Args:
            rallies: Lista di rally da verificare
            timeline: Timeline con tutti gli eventi
            
        Returns:
            Lista di rally con serve_count == 1
        """
        if not rallies:
            return []

        cfg = self.cfg
        events = timeline.sorted()

        # Estrai tutti i SERVE_START
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_conf_min
        ]
        serves.sort(key=lambda e: e.time)

        filtered: List[Rally] = []

        for rally in rallies:
            # Conta serve dentro questo rally
            serves_in_rally = [
                s for s in serves
                if rally.start <= s.time <= rally.end
            ]
            
            num_serves = len(serves_in_rally)
            
            # ENFORCE: serve_count == 1
            if num_serves != 1:
                if self.enable_logging:
                    self._log(
                        f"🚫 FILTRO SICUREZZA: Rally scartato con serve_count={num_serves} != 1 "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue
            
            filtered.append(rally)

        return filtered

    def _verify_serve_count_invariants(
        self, rallies: List[Rally], timeline: Timeline
    ) -> None:
        """
        Verifica invarianti: tutti i rally finali devono avere serve_count == 1.
        
        Args:
            rallies: Lista di rally da verificare
            timeline: Timeline con tutti gli eventi
        """
        cfg = self.cfg
        events = timeline.sorted()

        # Estrai tutti i SERVE_START
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_conf_min
        ]
        serves.sort(key=lambda e: e.time)

        violations = []
        for i, rally in enumerate(rallies, start=1):
            serves_in_rally = [
                s for s in serves
                if rally.start <= s.time <= rally.end
            ]
            num_serves = len(serves_in_rally)
            
            if num_serves != 1:
                violations.append((i, rally, num_serves))

        if violations:
            self._log("\n" + "="*80)
            self._log("❌ ERRORE: Violazioni dell'invariante serve_count == 1:")
            for i, rally, num_serves in violations:
                self._log(
                    f"   Rally {i}: [{rally.start:.2f}-{rally.end:.2f}s] "
                    f"serve_count={num_serves} != 1"
                )
            self._log("="*80 + "\n")
        else:
            self._log("\n✅ Verifica invarianti: tutti i rally finali hanno serve_count == 1\n")

    def _orchestrate_rally_extensions_rigorous(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Metodo MIGLIORATO per estendere i rally:
        - Preferisce rally che iniziano con SERVE_START
        - Accetta fine con SCORE_CHANGE, REF_POINT, WHISTLE_END o MOTION_GAP lungo (>3s)
        - Se non trova conferma, mantiene il rally originale se ha durata valida
        """
        if not rallies:
            return []

        events = timeline.sorted()
        extended: List[Rally] = []

        # Estrai eventi chiave
        serves = [e for e in events if e.type == EventType.SERVE_START]
        scores = [e for e in events if e.type == EventType.SCORE_CHANGE]
        ref_points = [e for e in events if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)]
        whistles_end = [e for e in events if e.type == EventType.WHISTLE_END]
        long_gaps = [
            e for e in events
            if e.type == EventType.MOTION_GAP
            and e.extra
            and e.extra.get("duration", 0) > 3.0
        ]

        for rally in rallies:
            # Verifica che il rally inizi con SERVE_START (tolleranza aumentata a 1.0s)
            serve_at_start = any(
                s.time == rally.start or abs(s.time - rally.start) < 1.0
                for s in serves
            )

            # Se non c'è SERVE_START ma il rally ha durata valida, accettalo comunque
            if not serve_at_start:
                duration = rally.end - rally.start
                if duration >= self.cfg.min_rally_duration and duration <= self.cfg.max_rally_duration:
                    # Rally valido anche senza SERVE_START esplicito
                    extended.append(rally)
                    if self.enable_logging:
                        self._log(
                            f"✅ Rally accettato senza SERVE_START esplicito (durata valida): "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue
                else:
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Rally scartato: nessun SERVE_START e durata non valida "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

            # Cerca fine confermata dopo il serve
            serve_time = rally.start
            confirmed_end = None

            # Priorità 1: SCORE_CHANGE dopo serve (finestra 45s)
            score_after_serve = next(
                (s for s in scores if serve_time < s.time < serve_time + 45.0),
                None
            )
            if score_after_serve:
                confirmed_end = score_after_serve.time
                extended.append(
                    Rally(start=rally.start, end=confirmed_end, side=rally.side)
                )
                if self.enable_logging:
                    self._log(
                        f"✅ Rally esteso fino a SCORE_CHANGE: "
                        f"[{rally.start:.2f}-{rally.end:.2f}s] -> [{rally.start:.2f}-{confirmed_end:.2f}s]"
                    )
                continue

            # Priorità 2: REF_POINT dopo serve (finestra 45s)
            ref_after_serve = next(
                (r for r in ref_points if serve_time < r.time < serve_time + 45.0),
                None
            )
            if ref_after_serve:
                confirmed_end = ref_after_serve.time
                extended.append(
                    Rally(start=rally.start, end=confirmed_end, side=rally.side)
                )
                if self.enable_logging:
                    self._log(
                        f"✅ Rally esteso fino a REF_POINT: "
                        f"[{rally.start:.2f}-{rally.end:.2f}s] -> [{rally.start:.2f}-{confirmed_end:.2f}s]"
                    )
                continue

            # Priorità 3: WHISTLE_END dopo serve (finestra 45s) - NUOVO
            whistle_after_serve = next(
                (w for w in whistles_end if serve_time < w.time < serve_time + 45.0),
                None
            )
            if whistle_after_serve:
                confirmed_end = whistle_after_serve.time
                if confirmed_end > rally.end:
                    extended.append(
                        Rally(start=rally.start, end=confirmed_end, side=rally.side)
                    )
                    if self.enable_logging:
                        self._log(
                            f"✅ Rally esteso fino a WHISTLE_END: "
                            f"[{rally.start:.2f}-{rally.end:.2f}s] -> [{rally.start:.2f}-{confirmed_end:.2f}s]"
                        )
                    continue

            # Priorità 4: MOTION_GAP lungo dopo serve (finestra 45s)
            gap_after_serve = next(
                (g for g in long_gaps if serve_time < g.time < serve_time + 45.0),
                None
            )
            if gap_after_serve:
                confirmed_end = gap_after_serve.time
                if confirmed_end > rally.end:
                    extended.append(
                        Rally(start=rally.start, end=confirmed_end, side=rally.side)
                    )
                    if self.enable_logging:
                        self._log(
                            f"✅ Rally esteso fino a MOTION_GAP lungo: "
                            f"[{rally.start:.2f}-{rally.end:.2f}s] -> [{rally.start:.2f}-{confirmed_end:.2f}s]"
                        )
                    continue

            # Fallback: se il rally ha durata valida, mantienilo anche senza conferma esplicita
            duration = rally.end - rally.start
            if duration >= self.cfg.min_rally_duration and duration <= self.cfg.max_rally_duration:
                extended.append(rally)
                if self.enable_logging:
                    self._log(
                        f"✅ Rally mantenuto senza conferma esplicita (durata valida): "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
            else:
                if self.enable_logging:
                    self._log(
                        f"⚠️ Rally scartato: durata non valida "
                        f"[{rally.start:.2f}-{rally.end:.2f}s] (dur={duration:.2f}s)"
                    )

        return extended

    def _split_rallies_with_multiple_serves(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Applica regola "1 serve = 1 rally" e split per rally troppo lunghi.

        Se un rally contiene più di un SERVE_START forte (confidence >= threshold),
        lo splitta in più rally, ciascuno iniziando con un serve e finendo prima del serve successivo.

        Se un rally è troppo lungo (>max_rally_duration_before_split) e contiene più serve,
        lo splitta automaticamente.

        Args:
            rallies: Lista di rally da controllare
            timeline: Timeline con tutti gli eventi

        Returns:
            Lista di rally eventualmente splittati
        """
        if not rallies:
            return []

        cfg = self.cfg
        events = timeline.sorted()
        split_rallies: List[Rally] = []

        # Estrai tutti i SERVE_START
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_confidence_threshold
        ]
        serves.sort(key=lambda e: e.time)

        # Estrai eventi di fine rally (per determinare fine dei sotto-rally)
        scores = [e for e in events if e.type == EventType.SCORE_CHANGE]
        ref_points = [e for e in events if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)]
        whistles_end = [e for e in events if e.type == EventType.WHISTLE_END]
        long_gaps = [
            e for e in events
            if e.type == EventType.MOTION_GAP
            and e.extra
            and e.extra.get("duration", 0) > 3.0
        ]

        for rally in rallies:
            duration = rally.end - rally.start

            # Trova tutti i serve dentro questo rally
            serves_in_rally = [
                s for s in serves
                if rally.start <= s.time <= rally.end
            ]

            # Determina durata minima da usare (dinamica o statica)
            effective_min_dur = self._get_effective_min_duration(rally, timeline)

            # Casi di split:
            # 1) Regola "1 serve = 1 rally": se ci sono più serve e enforce_one_serve_per_rally
            # 2) Rally troppo lungo: se duration > max_rally_duration_before_split e ci sono più serve

            should_split = False
            if cfg.enforce_one_serve_per_rally and len(serves_in_rally) > 1:
                should_split = True
                if self.enable_logging:
                    self._log(
                        f"🔄 Rally con {len(serves_in_rally)} serve: splittando in {len(serves_in_rally)} rally "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
            elif duration > cfg.max_rally_duration_before_split and len(serves_in_rally) > 1:
                should_split = True
                if self.enable_logging:
                    self._log(
                        f"🔄 Rally troppo lungo ({duration:.2f}s > {cfg.max_rally_duration_before_split:.2f}s) "
                        f"con {len(serves_in_rally)} serve: splittando "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )

            if should_split and len(serves_in_rally) > 1:
                # Split rally in base ai serve
                valid_sub_rallies = []
                for i, serve in enumerate(serves_in_rally):
                    serve_start = serve.time

                    # Fine del sotto-rally:
                    # - Se c'è un serve successivo, finisci poco prima di quello
                    # - Altrimenti, usa la fine del rally originale o un evento di fine confermato

                    if i < len(serves_in_rally) - 1:
                        # C'è un serve successivo: finisci poco prima di quello
                        next_serve_time = serves_in_rally[i + 1].time
                        # Cerca un evento di fine tra questo serve e il prossimo
                        sub_rally_end = self._find_rally_end(
                            serve_start,
                            next_serve_time - 0.5,  # poco prima del serve successivo
                            scores, ref_points, whistles_end, long_gaps,
                        )
                        if sub_rally_end is None:
                            # Nessun evento di fine trovato: usa tempo intermedio
                            sub_rally_end = serve_start + (next_serve_time - serve_start) * 0.95
                    else:
                        # Ultimo serve: usa fine originale o cerca evento di fine dopo
                        sub_rally_end = self._find_rally_end(
                            serve_start,
                            rally.end + 5.0,  # cerca fino a 5s dopo fine originale
                            scores, ref_points, whistles_end, long_gaps,
                        )
                        if sub_rally_end is None:
                            sub_rally_end = rally.end

                    # Verifica durata minima per split (min_dur_split, più permissivo)
                    sub_duration = sub_rally_end - serve_start
                    if sub_duration >= cfg.min_dur_split:
                        sub_rally = Rally(
                            start=serve_start,
                            end=sub_rally_end,
                            side=rally.side,
                        )
                        valid_sub_rallies.append(sub_rally)
                        split_rallies.append(sub_rally)
                        if self.enable_logging:
                            self._log(
                                f"  ✅ Sotto-rally {i+1}/{len(serves_in_rally)}: "
                                f"[{serve_start:.2f}-{sub_rally_end:.2f}s] "
                                f"(dur={sub_duration:.2f}s, min_dur_split={cfg.min_dur_split:.2f}s)"
                            )
                    else:
                        if self.enable_logging:
                            self._log(
                                f"  ⚠️ Sotto-rally {i+1}/{len(serves_in_rally)} scartato: "
                                f"durata troppo corta ({sub_duration:.2f}s < {cfg.min_dur_split:.2f}s)"
                            )
                
                # IMPORTANTE: se il rally è stato splittato, NON aggiungere il macro-rally originale
                # Usa solo i sotto-rally validi trovati sopra
                if self.enable_logging:
                    num_valid_sub_rallies = len(valid_sub_rallies)
                    if num_valid_sub_rallies >= 2:
                        self._log(
                            f"✅ Macro-rally SOSTITUITO con {num_valid_sub_rallies} sotto-rally validi "
                            f"[{rally.start:.2f}-{rally.end:.2f}s] → sotto-rally: "
                            + ", ".join(f"[{r.start:.2f}-{r.end:.2f}s]" for r in valid_sub_rallies)
                        )
                    elif num_valid_sub_rallies == 1:
                        self._log(
                            f"⚠️ Macro-rally splittato ma solo 1 sotto-rally valido: "
                            f"mantenuto solo quello [{valid_sub_rallies[0].start:.2f}-{valid_sub_rallies[0].end:.2f}s]"
                        )
                    else:
                        self._log(
                            f"⚠️ Macro-rally splittato ma nessun sotto-rally valido: "
                            f"scartato completamente [{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                continue  # NON aggiungere il macro-rally, usa solo i sotto-rally
            
            else:
                # Nessun split necessario: verifica serve singolo o assenza
                num_serves = len(serves_in_rally)
                
                # Scarta rally con serve_count == 0 se reject_rallies_without_serve è True
                if num_serves == 0 and cfg.reject_rallies_without_serve:
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Rally scartato: serve_count=0 (reject_rallies_without_serve=True) "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue
                
                # Verifica serve singolo
                if num_serves == 1:
                    serve = serves_in_rally[0]
                    # Verifica che il serve sia vicino all'inizio
                    serve_distance = abs(serve.time - rally.start)
                    if serve_distance > cfg.serve_max_distance_from_start:
                        # Serve troppo lontano dall'inizio: sposta start verso serve
                        if self.enable_logging:
                            self._log(
                                f"📍 Serve distante {serve_distance:.2f}s da start: "
                                f"spostando start da {rally.start:.2f}s a {serve.time:.2f}s"
                            )
                        rally.start = serve.time
                        duration = rally.end - rally.start  # Ricalcola durata dopo spostamento

                # Verifica durata con min_dur dinamico (ma qui usiamo min_dur_split come minimo)
                # Il filtro finale userà min_dur_final
                if duration >= effective_min_dur:
                    split_rallies.append(rally)
                    if self.enable_logging:
                        self._log(
                            f"✅ Rally mantenuto: [{rally.start:.2f}-{rally.end:.2f}s] "
                            f"(dur={duration:.2f}s, serve={num_serves}, min_dur={effective_min_dur:.2f}s)"
                        )
                else:
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Rally scartato: durata troppo corta "
                            f"({duration:.2f}s < {effective_min_dur:.2f}s) "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )

        return split_rallies

    def _filter_rallies_exactly_one_serve(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Filtro finale: mantiene solo rally con esattamente 1 serve.

        Scarta:
        - Rally con serve_count == 0 (se reject_rallies_without_serve è True)
        - Rally con serve_count > 1 (split fallito)
        - Rally con durata < min_dur_final
        - Rally con serve_time non entro serve_max_distance_from_start da start

        Gestisce anche overlap: se due rally si sovrappongono >1s, mantiene solo uno.

        Args:
            rallies: Lista di rally da filtrare
            timeline: Timeline con tutti gli eventi

        Returns:
            Lista di rally filtrati con esattamente 1 serve
        """
        if not rallies:
            return []

        cfg = self.cfg
        events = timeline.sorted()

        # Estrai tutti i SERVE_START
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
            and e.confidence >= cfg.serve_confidence_threshold
        ]
        serves.sort(key=lambda e: e.time)

        filtered: List[Rally] = []

        for rally in rallies:
            duration = rally.end - rally.start

            # Trova serve dentro questo rally
            serves_in_rally = [
                s for s in serves
                if rally.start - 0.7 <= s.time <= rally.end + 0.7  # epsilon ~0.7s
            ]

            num_serves = len(serves_in_rally)

            # Filtro 1: scarta rally con serve_count == 0
            if num_serves == 0:
                if cfg.reject_rallies_without_serve:
                    if self.enable_logging:
                        self._log(
                            f"🚫 Rally scartato (filtro finale): serve_count=0 "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

            # Filtro 2: mantieni solo rally con serve_count == 1 (se enforce_exactly_one_serve è True)
            if cfg.enforce_exactly_one_serve:
                if num_serves != 1:
                    if self.enable_logging:
                        self._log(
                            f"🚫 Rally scartato (filtro finale): serve_count={num_serves} != 1 "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

                # Verifica che il serve sia entro serve_max_distance_from_start da start
                serve = serves_in_rally[0]
                serve_distance = abs(serve.time - rally.start)
                if serve_distance > cfg.serve_max_distance_from_start:
                    if self.enable_logging:
                        self._log(
                            f"🚫 Rally scartato (filtro finale): serve troppo distante "
                            f"({serve_distance:.2f}s > {cfg.serve_max_distance_from_start:.2f}s) "
                            f"[{rally.start:.2f}-{rally.end:.2f}s]"
                        )
                    continue

            # Filtro 3: durata minima finale (min_dur_final, più rigido di min_dur_split)
            if duration < cfg.min_dur_final:
                if self.enable_logging:
                    self._log(
                        f"🚫 Rally scartato (filtro finale): durata troppo corta "
                        f"({duration:.2f}s < {cfg.min_dur_final:.2f}s) "
                        f"[{rally.start:.2f}-{rally.end:.2f}s]"
                    )
                continue

            # Aggiungi alla lista filtrata
            filtered.append(rally)

        # Filtro 4: gestisci overlap (se due rally si sovrappongono >1s, mantieni solo uno)
        filtered = self._filter_overlapping_rallies_final(filtered, timeline)

        return filtered

    def _filter_overlapping_rallies_final(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Filtra rally sovrapposti: se due rally si sovrappongono >1s, mantiene solo uno.

        Heuristica: mantiene il rally con:
        - Serve più chiaro (confidence maggiore)
        - Durata più ragionevole (3-10s)
        - Serve più vicino all'inizio

        Args:
            rallies: Lista di rally da filtrare
            timeline: Timeline con tutti gli eventi

        Returns:
            Lista di rally senza overlap significativi
        """
        if not rallies:
            return []

        events = timeline.sorted()
        serves = [
            e for e in events
            if e.type == EventType.SERVE_START
        ]

        # Ordina per tempo di inizio
        rallies_sorted = sorted(rallies, key=lambda r: r.start)
        filtered: List[Rally] = []

        for rally in rallies_sorted:
            # Trova serve nel rally per valutare confidence
            serves_in_rally = [
                s for s in serves
                if rally.start - 0.7 <= s.time <= rally.end + 0.7
            ]
            serve_conf = serves_in_rally[0].confidence if serves_in_rally else 0.0

            overlaps = False
            for existing_rally in filtered:
                # Calcola sovrapposizione
                overlap_start = max(rally.start, existing_rally.start)
                overlap_end = min(rally.end, existing_rally.end)
                overlap_duration = overlap_end - overlap_start if overlap_end > overlap_start else 0

                # Se sovrapposizione > 1s, è un conflitto
                if overlap_duration > 1.0:
                    overlaps = True

                    # Heuristica: mantieni il rally migliore
                    existing_serves = [
                        s for s in serves
                        if existing_rally.start - 0.7 <= s.time <= existing_rally.end + 0.7
                    ]
                    existing_serve_conf = existing_serves[0].confidence if existing_serves else 0.0

                    existing_duration = existing_rally.end - existing_rally.start
                    rally_duration = rally.end - rally.start

                    # Preferisci rally con serve più chiaro
                    if serve_conf > existing_serve_conf:
                        # Questo rally è migliore: sostituisci quello esistente
                        filtered.remove(existing_rally)
                        filtered.append(rally)
                        overlaps = False
                        if self.enable_logging:
                            self._log(
                                f"🔄 Rally sostituito per overlap ({overlap_duration:.2f}s): "
                                f"mantenuto [{rally.start:.2f}-{rally.end:.2f}s] "
                                f"(serve_conf={serve_conf:.2f} > {existing_serve_conf:.2f})"
                            )
                    elif serve_conf < existing_serve_conf:
                        # Quello esistente è migliore: scarta questo
                        if self.enable_logging:
                            self._log(
                                f"🔄 Rally scartato per overlap ({overlap_duration:.2f}s): "
                                f"mantenuto [{existing_rally.start:.2f}-{existing_rally.end:.2f}s] "
                                f"(serve_conf={existing_serve_conf:.2f} > {serve_conf:.2f})"
                            )
                    else:
                        # Stessa confidence: preferisci durata ragionevole (3-10s)
                        if 3.0 <= rally_duration <= 10.0 and not (3.0 <= existing_duration <= 10.0):
                            filtered.remove(existing_rally)
                            filtered.append(rally)
                            overlaps = False
                            if self.enable_logging:
                                self._log(
                                    f"🔄 Rally sostituito per overlap ({overlap_duration:.2f}s): "
                                    f"mantenuto [{rally.start:.2f}-{rally.end:.2f}s] "
                                    f"(durata ragionevole: {rally_duration:.2f}s vs {existing_duration:.2f}s)"
                                )
                        else:
                            if self.enable_logging:
                                self._log(
                                    f"🔄 Rally scartato per overlap ({overlap_duration:.2f}s): "
                                    f"mantenuto [{existing_rally.start:.2f}-{existing_rally.end:.2f}s]"
                                )
                    break

            if not overlaps:
                filtered.append(rally)

        return filtered

    def _find_rally_end(
        self,
        start_time: float,
        max_search_time: float,
        scores: List[Event],
        ref_points: List[Event],
        whistles_end: List[Event],
        long_gaps: List[Event],
    ) -> Optional[float]:
        """
        Trova l'evento di fine rally più vicino dopo start_time.

        Priorità: SCORE_CHANGE > REF_POINT > WHISTLE_END > MOTION_GAP lungo

        Returns:
            Tempo dell'evento di fine o None se non trovato
        """
        # Priorità 1: SCORE_CHANGE
        score_after = next(
            (s.time for s in scores if start_time < s.time <= max_search_time),
            None
        )
        if score_after:
            return score_after

        # Priorità 2: REF_POINT
        ref_after = next(
            (r.time for r in ref_points if start_time < r.time <= max_search_time),
            None
        )
        if ref_after:
            return ref_after

        # Priorità 3: WHISTLE_END
        whistle_after = next(
            (w.time for w in whistles_end if start_time < w.time <= max_search_time),
            None
        )
        if whistle_after:
            return whistle_after

        # Priorità 4: MOTION_GAP lungo
        gap_after = next(
            (g.time for g in long_gaps if start_time < g.time <= max_search_time),
            None
        )
        if gap_after:
            return gap_after

        return None

    def _get_effective_min_duration(self, rally: Rally, timeline: Timeline) -> float:
        """
        Calcola durata minima effettiva per un rally.

        Se dynamic_min_duration è attivo e il rally ha forte consensus tra agenti,
        usa strong_consensus_min_duration invece di min_rally_duration.

        Args:
            rally: Rally da analizzare
            timeline: Timeline con tutti gli eventi

        Returns:
            Durata minima effettiva (secondi)
        """
        cfg = self.cfg
        if not cfg.dynamic_min_duration:
            return cfg.min_rally_duration

        # Verifica se c'è forte consensus
        events_in_rally = [
            e for e in timeline.sorted()
            if rally.start <= e.time <= rally.end
        ]

        # Cerca eventi da AudioAgent, ServeAgent, ScoreboardAgent, TouchSequenceAgent
        has_whistle = any(
            e.type == EventType.WHISTLE_END
            and e.confidence >= cfg.strong_consensus_threshold
            for e in events_in_rally
        )
        has_serve = any(
            e.type == EventType.SERVE_START
            and e.confidence >= cfg.strong_consensus_threshold
            for e in events_in_rally
        )
        has_score = any(
            e.type == EventType.SCORE_CHANGE
            and e.confidence >= cfg.strong_consensus_threshold
            for e in events_in_rally
        )
        # TouchSequenceAgent: verifica se c'è un ultimo tocco marcato come fine sequenza
        has_touch_seq_end = any(
            e.type in (EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT, EventType.TOUCH_LEFT, EventType.TOUCH_RIGHT)
            and e.extra
            and e.extra.get("is_sequence_end", False)
            and e.confidence >= cfg.strong_consensus_threshold
            for e in events_in_rally
        )

        # Consenso forte se: (Audio + Serve) E (Score OR TouchSequence)
        strong_consensus = (has_whistle and has_serve) and (has_score or has_touch_seq_end)

        if strong_consensus:
            return cfg.strong_consensus_min_duration
        else:
            return cfg.min_rally_duration

    def _orchestrate_rally_extensions(
        self, rallies: List[Rally], timeline: Timeline
    ) -> List[Rally]:
        """
        Usa RallyOrchestrator per estendere i rally fino a conferma definitiva.

        L'orchestratore analizza ogni rally, identifica cosa manca e decide
        quale agente chiamare per ottenere le informazioni necessarie.

        Args:
            rallies: Lista di rally da estendere
            timeline: Timeline con tutti gli eventi

        Returns:
            Lista di rally estesi fino a conferma definitiva
        """
        if not rallies:
            return []

        extended: List[Rally] = []

        for rally in rallies:
            # RallyOrchestrator disabilitato - ritorno a logica Analisi 1
            # Usa estensione semplice basata su SCORE_CHANGE / REF_POINT
            events = timeline.sorted()
            after_rally = [
                e for e in events
                if rally.end < e.time <= rally.end + 3.0
            ]
            
            # Priorità: SCORE_CHANGE > REF_POINT
            score_after = next(
                (e for e in after_rally if e.type == EventType.SCORE_CHANGE),
                None
            )
            if score_after:
                extended.append(
                    Rally(start=rally.start, end=score_after.time, side=rally.side)
                )
                continue
            
            ref_after = next(
                (e for e in after_rally 
                 if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)),
                None
            )
            if ref_after:
                extended.append(
                    Rally(start=rally.start, end=ref_after.time, side=rally.side)
                )
                continue

            # Nessun segnale forte trovato: usa il rally originale
            extended.append(rally)

        return extended

    def _filter_final_rallies(self, rallies: List[Rally]) -> List[Rally]:
        """
        Filtro finale per rimuovere duplicati e sovrapposizioni tra rally.

        Args:
            rallies: Lista di rally da filtrare

        Returns:
            Lista di rally filtrati e deduplicati
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

            # Filtro 2: verifica distanza minima dal rally precedente (più tollerante)
            if filtered:
                last_rally = filtered[-1]
                gap = rally.start - last_rally.end
                # Se gap negativo (sovrapposizione), gestisci sovrapposizione
                if gap < 0:
                    # Sovrapposizione: gestita nel filtro 3
                    pass
                elif gap < cfg.min_rally_gap:
                    # Rally troppo vicino: scegli quello con durata maggiore
                    if duration > (last_rally.end - last_rally.start):
                        # Il nuovo rally è più lungo: rimuovi quello precedente
                        filtered.pop()
                        if self.enable_logging:
                            self._log(
                                f"⚠️ Rally rimosso: troppo vicino ({gap:.2f}s gap) - "
                                f"mantenuto rally più lungo [{rally.start:.2f}-{rally.end:.2f}s]"
                            )
                    else:
                        # Il precedente è più lungo: salta questo
                        if self.enable_logging:
                            self._log(
                                f"⚠️ Rally rimosso: troppo vicino ({gap:.2f}s gap) - "
                                f"mantenuto rally precedente [{last_rally.start:.2f}-{last_rally.end:.2f}s]"
                            )
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
                            if self.enable_logging:
                                self._log(
                                    f"⚠️ Rally sostituito: sovrapposizione {overlap_duration:.2f}s - "
                                    f"mantenuto rally più lungo [{rally.start:.2f}-{rally.end:.2f}s]"
                                )
                        else:
                            if self.enable_logging:
                                self._log(
                                    f"⚠️ Rally rimosso: sovrapposizione {overlap_duration:.2f}s - "
                                    f"mantenuto rally esistente [{existing_rally.start:.2f}-{existing_rally.end:.2f}s]"
                                )
                        break

            if not overlaps:
                filtered.append(rally)

        return filtered

    def _identify_rallies_with_voting(self, events: List[Event]) -> List[Tuple[Optional[VotingResult], Optional[VotingResult]]]:
        """
        Identifica inizio/fine rally usando voting system multi-agente.

        Ogni agente vota per inizio/fine con confidence, il Coach combina i voti.

        Args:
            events: Lista di eventi ordinata

        Returns:
            Lista di (start_vote, end_vote) per ogni rally candidato
        """
        if not events:
            return []

        # Raccogli voti per inizio rally (usa finestre temporali per raggruppare)
        # Cerca cluster di eventi che potrebbero indicare inizio rally
        start_votes = []
        end_votes = []

        # Finestra temporale per raggruppare eventi vicini (1 secondo)
        window_size = 1.0

        # Scorri eventi e cerca cluster di segnali per inizio
        i = 0
        while i < len(events):
            event = events[i]
            # Finestra temporale attorno all'evento
            t_start_window = max(0, event.time - window_size)
            t_end_window = event.time + window_size

            # Vota inizio rally in questa finestra
            vote_start = self.voting_system.vote_rally_start(
                events,
                time_window=(t_start_window, t_end_window),
            )
            if vote_start and vote_start not in [v[1] for v in start_votes]:
                start_votes.append((vote_start.time, vote_start))

            # Vota fine rally in questa finestra
            vote_end = self.voting_system.vote_rally_end(
                events,
                time_window=(t_start_window, t_end_window),
            )
            if vote_end and vote_end not in [v[1] for v in end_votes]:
                end_votes.append((vote_end.time, vote_end))

            i += 1

        # Rimuovi duplicati (voti molto vicini nel tempo)
        start_votes = self._deduplicate_votes(start_votes)
        end_votes = self._deduplicate_votes(end_votes)

        # Abbina inizi e fine per formare rally
        rallies = []
        for start_time, start_vote in start_votes:
            # Cerca fine più vicina dopo l'inizio
            end_vote = None
            for end_time, vote in end_votes:
                if end_time > start_time and (end_vote is None or end_time < end_vote.time):
                    end_vote = vote

            if end_vote:
                rallies.append((start_vote, end_vote))
            else:
                # Inizio senza fine (rally incompleto) - usa fallback
                pass

        return rallies

    def _deduplicate_votes(self, votes: List[Tuple[float, VotingResult]]) -> List[Tuple[float, VotingResult]]:
        """Rimuove voti duplicati (molto vicini nel tempo)."""
        if not votes:
            return []

        votes_sorted = sorted(votes, key=lambda x: x[0])
        deduplicated = [votes_sorted[0]]

        for time, vote in votes_sorted[1:]:
            # Se il voto è a più di 0.5s dal precedente, è un nuovo voto
            if time - deduplicated[-1][0] > 0.5:
                deduplicated.append((time, vote))
            else:
                # Prendi il voto con confidence più alta
                if vote.confidence > deduplicated[-1][1].confidence:
                    deduplicated[-1] = (time, vote)

        return deduplicated

    def _rallies_from_voting(
        self,
        rally_candidates: List[Tuple[Optional[VotingResult], Optional[VotingResult]]],
        events: List[Event],
    ) -> List[Rally]:
        """
        Converte voting results in Rally con estensione intelligente della fine.

        Usa ultimo HIT + tail_min + post_roll per allungare i clip e mostrare
        la palla che cade e la mini-esultanza.

        Args:
            rally_candidates: Lista di (start_vote, end_vote)
            events: Lista di eventi per contesto

        Returns:
            Lista di Rally con fine estesa
        """
        cfg = self.cfg
        rallies: List[Rally] = []

        # Determina finestra temporale globale per clamp
        if events:
            t_min = min(e.time for e in events)
            t_max = max(e.time for e in events)
        else:
            t_min = 0.0
            t_max = float('inf')

        for start_vote, end_vote in rally_candidates:
            if start_vote is None:
                continue

            start_t = start_vote.time
            raw_end_t = end_vote.time if end_vote is not None else start_t + 10.0

            # Trova eventi nella finestra del rally
            window_events = [
                e for e in events
                if start_t <= e.time <= raw_end_t + 5.0  # estendi ricerca a 5s dopo raw_end
            ]

            # Trova l'ultimo HIT nella finestra dopo l'inizio rally
            hits_in_window = [
                e for e in window_events
                if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT) and e.time >= start_t
            ]

            last_hit_time = max(
                (e.time for e in hits_in_window),
                default=start_t  # fallback: usa start_t se non ci sono HIT
            )

            # Calcola fine base: massimo tra raw_end_t e last_hit + tail_min
            tail_min = cfg.tail_min
            base_end = max(raw_end_t, last_hit_time + tail_min)

            # Aggiungi post_roll per vedere palla che cade e mini-esultanza
            post_roll = cfg.post_roll
            end_t = base_end + post_roll

            # Clamp: non superare i limiti della finestra
            end_t = min(end_t, t_max)

            # Determina side da start_vote
            side = "unknown"
            if start_vote.dominant_signal:
                # Estrai side da segnale dominante
                if "LEFT" in start_vote.dominant_signal or "left" in start_vote.dominant_signal.lower():
                    side = "left"
                elif "RIGHT" in start_vote.dominant_signal or "right" in start_vote.dominant_signal.lower():
                    side = "right"
            elif hits_in_window:
                # Fallback: usa side dal primo HIT
                first_hit = hits_in_window[0]
                side = "left" if first_hit.type == EventType.HIT_LEFT else "right"

            # Verifica durata minima (dopo estensione)
            duration = end_t - start_t
            if duration < cfg.min_rally_duration:
                # Durata troppo corta dopo estensione: scarta
                if self.enable_logging:
                    self._log(
                        f"⚠️ Rally scartato: durata troppo corta ({duration:.2f}s < {cfg.min_rally_duration:.2f}s) "
                        f"dopo estensione [{start_t:.2f}-{end_t:.2f}s]"
                    )
                continue

            # Verifica sovrapposizione con rally precedenti
            overlaps_previous = False
            for existing_rally in rallies:
                # Verifica sovrapposizione
                if not (end_t < existing_rally.start or start_t > existing_rally.end):
                    # C'è sovrapposizione: salta questo rally
                    overlaps_previous = True
                    if self.enable_logging:
                        self._log(
                            f"⚠️ Rally scartato: sovrappone con rally precedente "
                            f"[{existing_rally.start:.2f}-{existing_rally.end:.2f}s] vs "
                            f"[{start_t:.2f}-{end_t:.2f}s]"
                        )
                    break

            if not overlaps_previous:
                rallies.append(
                    Rally(
                        start=start_t,
                        end=end_t,
                        side=side,
                    )
                )
                if self.enable_logging:
                    self._log(
                        f"✅ Rally creato: [{start_t:.2f}-{end_t:.2f}s] "
                        f"(dur={duration:.2f}s, last_hit={last_hit_time:.2f}s, "
                        f"tail={tail_min:.1f}s, post_roll={post_roll:.1f}s)"
                    )

        return rallies

    def _merge_rally_lists(self, voting_rallies: List[Rally], hc_rallies: List[Rally]) -> List[Rally]:
        """
        Combina rally da voting e HeadCoach, dando priorità a voting.

        Args:
            voting_rallies: Rally da voting system
            hc_rallies: Rally da HeadCoach

        Returns:
            Lista combinata di rally
        """
        if not voting_rallies:
            return hc_rallies

        merged = list(voting_rallies)

        # Aggiungi rally di HeadCoach che non si sovrappongono con voting
        for hc_rally in hc_rallies:
            overlaps = False
            for v_rally in voting_rallies:
                # Verifica sovrapposizione
                if not (hc_rally.end < v_rally.start or hc_rally.start > v_rally.end):
                    overlaps = True
                    break
            if not overlaps:
                merged.append(hc_rally)

        # Ordina per tempo
        merged.sort(key=lambda r: r.start)
        return merged

    def _enrich_rally(self, rally: Rally, timeline: Timeline) -> Optional[Rally]:
        """
        Arricchisce un rally con informazioni da tutti gli agenti.

        Args:
            rally: Rally base da HeadCoach
            timeline: Timeline con tutti gli eventi

        Returns:
            Rally arricchito o None se non valido
        """
        # Filtra eventi nel rally
        events_in_rally = [
            e
            for e in timeline.sorted()
            if rally.start <= e.time <= rally.end
        ]

        # NOTA: NON validare durata qui con min_rally_duration
        # Level B già ha applicato min_dur_hard_s e min_dur_soft_s con consensus
        # _enrich_rally() deve solo aggiungere metadati, non scartare rally validi
        # Se validate_rules è attivo, usa solo per validazione FIPAV (tocchi, ecc.), non durata

        # Estrai informazioni da eventi
        serve_events = [e for e in events_in_rally if e.type == EventType.SERVE_START]
        score_events = [e for e in events_in_rally if e.type == EventType.SCORE_CHANGE]
        hit_events = [e for e in events_in_rally if e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT)]
        
        # Estrai tocchi dettagliati (da TouchSequenceAgent se disponibile)
        touch_events = [
            e for e in events_in_rally
            if e.type in (
                EventType.TOUCH_LEFT, EventType.TOUCH_RIGHT,
                EventType.SET_LEFT, EventType.SET_RIGHT,
                EventType.ATTACK_LEFT, EventType.ATTACK_RIGHT,
                EventType.RECEPTION_LEFT, EventType.RECEPTION_RIGHT,
            )
        ]
        
        jump_events = [e for e in events_in_rally if e.type == EventType.JUMP_EVENT]

        # Valida sequenze di tocchi secondo regole FIPAV
        if self.cfg.validate_rules and touch_events:
            # Costruisci lista tocchi per validazione
            touches_data = []
            for e in touch_events:
                side = "left" if "LEFT" in e.type.value else "right"
                touches_data.append((e.time, e.type.value, side))
            
            touches_data.sort(key=lambda x: x[0])  # Ordina per tempo
            
            is_valid, error, touch_count = validate_rally_touches(touches_data)
            
            if not is_valid:
                # Rally non valido secondo regole FIPAV (troppi tocchi, ecc.)
                # Per ora lo accettiamo ma potrebbe essere filtrato in futuro
                pass

        # Determina chi serve (da SERVE_START se disponibile)
        serving_player = None
        serving_role = None
        if serve_events:
            serve = serve_events[0]
            if serve.extra and "side" in serve.extra:
                rally.side = serve.extra["side"]
            if serve.extra and "player_id" in serve.extra:
                serving_player = serve.extra["player_id"]
            if serve.extra and "role" in serve.extra:
                serving_role = serve.extra["role"]

        # Determina sequenza di tocchi (ricezione -> palleggio -> attacco)
        touch_sequence = []
        if touch_events:
            touch_sequence = sorted(touch_events, key=lambda e: e.time)
            
            # Estrai informazioni sulla sequenza
            reception_count = sum(1 for e in touch_sequence if "RECEPTION" in e.type.value)
            set_count = sum(1 for e in touch_sequence if "SET" in e.type.value)
            attack_count = sum(1 for e in touch_sequence if "ATTACK" in e.type.value)

        # Determina altezza salto principale (da JUMP_EVENT se disponibile)
        max_jump_height = None
        if jump_events:
            jump_heights = [
                e.extra.get("jump_height_m")
                for e in jump_events
                if e.extra and "jump_height_m" in e.extra
            ]
            if jump_heights:
                max_jump_height = max(jump_heights)

        # Aggiorna GameState per il punteggio
        if score_events:
            last_score = score_events[-1]
            if last_score.extra:
                if "score_left" in last_score.extra:
                    self.game_state.update_score("left", last_score.extra["score_left"])
                if "score_right" in last_score.extra:
                    self.game_state.update_score("right", last_score.extra["score_right"])

        # TODO: aggiungere altre informazioni (ruoli, rotazioni, violazioni, ecc.)

        # Restituisci rally arricchito (per ora identico, ma pronto per estensioni)
        return rally

    def _update_game_state(self, rallies: List[Rally], timeline: Timeline):
        """
        Aggiorna GameState basandosi sui rally e sulla timeline.

        Args:
            rallies: Lista di rally arricchiti
            timeline: Timeline con tutti gli eventi
        """
        events = timeline.sorted()

        # Aggiorna punteggio da SCORE_CHANGE
        score_events = [e for e in events if e.type == EventType.SCORE_CHANGE]
        if score_events:
            last_score = score_events[-1]
            if last_score.extra:
                if "score_left" in last_score.extra:
                    self.game_state.update_score("left", last_score.extra["score_left"])
                if "score_right" in last_score.extra:
                    self.game_state.update_score("right", last_score.extra["score_right"])

        # Aggiorna serving_side da SERVE_START
        serve_events = [e for e in events if e.type == EventType.SERVE_START]
        if serve_events:
            last_serve = serve_events[-1]
            if last_serve.extra and "side" in last_serve.extra:
                self.game_state.set_serving_side(last_serve.extra["side"])

        # TODO: aggiornare rotazioni da ROTATION_UPDATE

    def get_game_state(self) -> GameState:
        """Restituisce il GameState corrente."""
        return self.game_state

    def _log_rally_details(self, rallies: List[Rally], timeline: Timeline) -> None:
        """
        Log dettagliato per ogni rally: numero serve, durata, segnali di chiusura.

        Args:
            rallies: Lista di rally da loggare
            timeline: Timeline con tutti gli eventi
        """
        if not rallies:
            return

        events = timeline.sorted()
        self._log("\n" + "="*80)
        self._log(f"📊 DETTAGLI RALLY ({len(rallies)} totali) - voting-based:")
        self._log("="*80)

        for i, rally in enumerate(rallies, start=1):
            duration = rally.end - rally.start

            # Conta serve nel rally
            serves_in_rally = [
                e for e in events
                if e.type == EventType.SERVE_START
                and rally.start <= e.time <= rally.end
            ]
            num_serves = len(serves_in_rally)

            # Trova segnali di chiusura
            closing_signals = []

            # Cerca SCORE_CHANGE
            scores_in_rally = [
                e for e in events
                if e.type == EventType.SCORE_CHANGE
                and rally.start <= e.time <= rally.end
            ]
            if scores_in_rally:
                closing_signals.append(f"SCORE_CHANGE@{scores_in_rally[-1].time:.2f}s")

            # Cerca REF_POINT
            ref_points_in_rally = [
                e for e in events
                if e.type in (EventType.REF_POINT_LEFT, EventType.REF_POINT_RIGHT)
                and rally.start <= e.time <= rally.end
            ]
            if ref_points_in_rally:
                closing_signals.append(f"REF_POINT@{ref_points_in_rally[-1].time:.2f}s")

            # Cerca WHISTLE_END
            whistles_in_rally = [
                e for e in events
                if e.type == EventType.WHISTLE_END
                and rally.start <= e.time <= rally.end
            ]
            if whistles_in_rally:
                closing_signals.append(f"WHISTLE_END@{whistles_in_rally[-1].time:.2f}s")

            # Cerca MOTION_GAP lungo
            gaps_in_rally = [
                e for e in events
                if e.type == EventType.MOTION_GAP
                and rally.start <= e.time <= rally.end
                and e.extra
                and e.extra.get("duration", 0) > 3.0
            ]
            if gaps_in_rally:
                closing_signals.append(f"MOTION_GAP@{gaps_in_rally[-1].time:.2f}s")

            # Determina segnale dominante
            if closing_signals:
                dominant_signal = closing_signals[0]  # prima priorità
            else:
                dominant_signal = "Nessun segnale forte (fallback durata)"

            # Conta eventi di punteggio
            has_score, scoring_events = self._has_scoring_evidence(timeline, rally.start, rally.end)
            score_events_count = len(scoring_events)
            
            score_events_str = ""
            if scoring_events:
                # Mostra primi 2 eventi di punteggio
                score_types = [e.type.value for e in scoring_events[:2]]
                score_times = [f"{e.time:.2f}" for e in scoring_events[:2]]
                score_events_str = f" | score_events={score_events_count} ({', '.join(f'{t}@{time}' for t, time in zip(score_types, score_times))})"
            else:
                score_events_str = " | score_events=0"
            
            # Log dettagliato (voting-based, no serve_count enforcement)
            serve_info = f"serve={num_serves}"
            if serves_in_rally:
                serve_times_str = ", ".join(f"{s.time:.2f}" for s in serves_in_rally[:2])
                if num_serves > 2:
                    serve_times_str += "..."
                serve_info = f"serve={num_serves} ({serve_times_str})"
            
            self._log(
                f"Rally {i:2d}: [{rally.start:7.2f}-{rally.end:7.2f}s] "
                f"dur={duration:5.2f}s | "
                f"{serve_info} | "
                f"chiusura={dominant_signal}{score_events_str}"
            )

        self._log("="*80 + "\n")


__all__ = [
    "MasterCoach",
    "MasterCoachConfig",
]

