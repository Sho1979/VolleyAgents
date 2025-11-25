"""
ACE Detector con Verifica Direzione e Log Dettagliato.

File: volley_agents/fusion/ace_detector.py

Criteri per ACE:
1. Serve forte (confidence >= 0.70)
2. Fischio entro 1.5-6 secondi
3. Nessun altro serve forte nel mezzo
4. Nessun movimento di "ritorno" (ricezione/difesa)

Autore: VolleyAgents
Versione: 1.0
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable

from volley_agents.core.event import Event, EventType


# =============================================================================
# HELPER: Conversione tempo in mm:ss
# =============================================================================

def format_time(seconds: float) -> str:
    """Converte secondi in mm:ss.cc. Es: 1286.40 → '21:26.40'"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def format_time_short(seconds: float) -> str:
    """Converte secondi in mm:ss. Es: 1286.40 → '21:26'"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def format_time_range(start: float, end: float) -> str:
    """Formatta range temporale. Es: '21:26-21:30'"""
    return f"{format_time_short(start)}-{format_time_short(end)}"


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

@dataclass
class AceConfig:
    """Configurazione per ace detection."""

    # Soglie serve
    min_serve_confidence: float = 0.70      # serve deve essere "forte"
    min_other_serve_conf: float = 0.70      # ignora serve deboli nel mezzo

    # Durata ace
    min_ace_duration: float = 1.5           # durata minima ace (secondi)
    max_ace_duration: float = 6.0           # durata massima ace (secondi)

    # Verifica direzione
    check_direction: bool = True
    direction_check_delay: float = 0.5      # secondi dopo serve per iniziare check
    min_return_motion_magnitude: float = 0.8  # magnitudine minima per "ritorno"

    # Pre/post roll per clip
    pre_roll: float = 0.5                   # secondi prima del serve
    post_roll: float = 1.5                  # secondi dopo il fischio

    # Debug
    verbose_log: bool = True                # log dettagliato


# =============================================================================
# ACE DETECTOR
# =============================================================================

class AceDetector:
    """
    Rileva ace nella timeline degli eventi.

    Uso:
        detector = AceDetector(cfg=AceConfig(), log_callback=print)
        aces = detector.find_aces(timeline_events)
    """

    def __init__(
        self,
        cfg: AceConfig = None,
        log_callback: Callable[[str], None] = None
    ):
        self.cfg = cfg or AceConfig()
        self._log_callback = log_callback

    def _log(self, msg: str):
        """Log con prefisso."""
        if self._log_callback:
            self._log_callback(f"[AceDetector] {msg}")

    def _debug(self, msg: str):
        """Log verbose (solo se abilitato)."""
        if self.cfg.verbose_log and self._log_callback:
            self._log_callback(f"[AceDetector] {msg}")

    # =========================================================================
    # API PRINCIPALE
    # =========================================================================

    def find_aces(self, events: List[Event]) -> List[Dict[str, Any]]:
        """
        Trova tutti gli ace nella lista di eventi.

        Args:
            events: lista di Event (dalla timeline)

        Returns:
            Lista di dizionari rally ace
        """
        # Separa eventi per tipo
        serve_events = []
        whistle_events = []
        motion_events = []

        for e in events:
            if e.type == EventType.SERVE_START:
                serve_events.append(e)
            elif e.type == EventType.WHISTLE_END:
                whistle_events.append(e)
            elif e.type in (EventType.HIT_LEFT, EventType.HIT_RIGHT):
                motion_events.append(e)

        self._log(f"📊 Analisi ace: {len(serve_events)} serve, "
                  f"{len(whistle_events)} fischi, {len(motion_events)} hit")

        # Analizza ogni serve
        ace_rallies = []

        for serve in serve_events:
            ace = self._analyze_serve_for_ace(
                serve=serve,
                whistle_events=whistle_events,
                all_serve_events=serve_events,
                motion_events=motion_events
            )
            if ace:
                ace_rallies.append(ace)

        self._log(f"⚡ Totale ace trovati: {len(ace_rallies)}")

        return ace_rallies

    # =========================================================================
    # ANALISI SINGOLO SERVE
    # =========================================================================

    def _analyze_serve_for_ace(
        self,
        serve: Event,
        whistle_events: List[Event],
        all_serve_events: List[Event],
        motion_events: List[Event]
    ) -> Optional[Dict[str, Any]]:
        """Analizza se un singolo serve è un ace."""

        cfg = self.cfg
        serve_time = serve.time
        serve_conf = serve.confidence
        serve_side = self._get_serve_side(serve)

        self._debug("")
        self._debug("=" * 60)
        self._debug(f"🎯 Analisi serve @ {format_time(serve_time)} ({serve_side})")
        self._debug(f"   Confidence: {serve_conf:.2f}")

        # ---------------------------------------------------------------------
        # STEP 1: Verifica confidence serve
        # ---------------------------------------------------------------------
        if serve_conf < cfg.min_serve_confidence:
            self._debug(f"   ❌ SKIP: confidence {serve_conf:.2f} < {cfg.min_serve_confidence}")
            return None

        self._debug(f"   ✅ Step 1: Serve forte (conf={serve_conf:.2f} >= {cfg.min_serve_confidence})")

        # ---------------------------------------------------------------------
        # STEP 2: Cerca fischio nella finestra temporale
        # ---------------------------------------------------------------------
        min_whistle_time = serve_time + cfg.min_ace_duration
        max_whistle_time = serve_time + cfg.max_ace_duration

        self._debug(f"   🔍 Step 2: Cerco WHISTLE_END in finestra "
                    f"[{format_time(min_whistle_time)} - {format_time(max_whistle_time)}]")

        # Trova tutti i fischi nella finestra
        whistles_in_window = []
        for w in whistle_events:
            if min_whistle_time < w.time < max_whistle_time:
                whistles_in_window.append(w)
                self._debug(f"      → Fischio trovato @ {format_time(w.time)} "
                            f"(conf={w.confidence:.2f})")

        if not whistles_in_window:
            self._debug(f"   ❌ SKIP: Nessun fischio trovato nella finestra")
            # Log fischi vicini per debug
            nearby_whistles = [w for w in whistle_events
                              if serve_time < w.time < serve_time + 10]
            if nearby_whistles:
                self._debug(f"      Fischi vicini (entro 10s):")
                for w in nearby_whistles[:5]:
                    delta = w.time - serve_time
                    self._debug(f"        @ {format_time(w.time)} (Δ={delta:.2f}s)")
            return None

        # Prendi il fischio più vicino al serve
        best_whistle = min(whistles_in_window, key=lambda w: w.time)
        whistle_time = best_whistle.time
        ace_duration = whistle_time - serve_time

        self._debug(f"   ✅ Step 2: Fischio trovato @ {format_time(whistle_time)} "
                    f"(durata ace={ace_duration:.2f}s)")

        # ---------------------------------------------------------------------
        # STEP 3: Verifica altri serve forti nel mezzo
        # ---------------------------------------------------------------------
        self._debug(f"   🔍 Step 3: Cerco altri serve forti in "
                    f"[{format_time(serve_time)} - {format_time(whistle_time)}]")

        other_strong_serves = []
        for s in all_serve_events:
            if s.time == serve_time:
                continue  # escludi se stesso
            if serve_time < s.time < whistle_time:
                self._debug(f"      → Serve @ {format_time(s.time)} "
                            f"(conf={s.confidence:.2f})")
                if s.confidence >= cfg.min_other_serve_conf:
                    other_strong_serves.append(s)
                    self._debug(f"         ⚠️ FORTE!")
                else:
                    self._debug(f"         (ignorato: conf < {cfg.min_other_serve_conf})")

        if other_strong_serves:
            self._debug(f"   ❌ SKIP: {len(other_strong_serves)} serve forti nel mezzo")
            return None

        self._debug(f"   ✅ Step 3: Nessun serve forte nel mezzo")

        # ---------------------------------------------------------------------
        # STEP 4: Verifica direzione movimento (nessun ritorno)
        # ---------------------------------------------------------------------
        if cfg.check_direction and serve_side in ('left', 'right'):
            self._debug(f"   🔍 Step 4: Verifico direzione (serve da {serve_side})")

            has_return = self._check_return_motion(
                serve_time=serve_time,
                whistle_time=whistle_time,
                serve_side=serve_side,
                motion_events=motion_events
            )

            if has_return:
                self._debug(f"   ❌ SKIP: Rilevato movimento di ritorno (ricezione)")
                return None

            self._debug(f"   ✅ Step 4: Nessun movimento di ritorno")
        else:
            self._debug(f"   ⏭️ Step 4: Verifica direzione saltata "
                        f"(check_direction={cfg.check_direction}, side={serve_side})")

        # ---------------------------------------------------------------------
        # ACE CONFERMATO!
        # ---------------------------------------------------------------------
        self._log(f"⚡ ACE CONFERMATO @ {format_time(serve_time)} ({serve_side}) → "
                  f"{format_time(whistle_time)} (dur={ace_duration:.2f}s)")

        return {
            "start": serve_time - cfg.pre_roll,
            "end": whistle_time + cfg.post_roll,
            "duration": ace_duration,
            "type": "ace",
            "serve_time": serve_time,
            "serve_time_formatted": format_time(serve_time),
            "whistle_time": whistle_time,
            "whistle_time_formatted": format_time(whistle_time),
            "serve_side": serve_side,
            "side": serve_side,
            "confidence": min(serve_conf, best_whistle.confidence),
        }

    # =========================================================================
    # VERIFICA DIREZIONE MOVIMENTO
    # =========================================================================

    def _check_return_motion(
        self,
        serve_time: float,
        whistle_time: float,
        serve_side: str,
        motion_events: List[Event]
    ) -> bool:
        """
        Verifica se c'è movimento di "ritorno" (ricezione/difesa).

        Logica:
        - Serve da RIGHT → palla va verso LEFT → ritorno = HIT_RIGHT
        - Serve da LEFT → palla va verso RIGHT → ritorno = HIT_LEFT

        Returns:
            True se c'è movimento di ritorno (NON è ace)
        """
        cfg = self.cfg

        # Determina quale hit indica "ritorno"
        if serve_side == 'right':
            return_hit_type = EventType.HIT_RIGHT
            self._debug(f"      Serve da RIGHT → cerco HIT_RIGHT (ritorno)")
        else:  # left
            return_hit_type = EventType.HIT_LEFT
            self._debug(f"      Serve da LEFT → cerco HIT_LEFT (ritorno)")

        # Finestra temporale per check
        check_start = serve_time + cfg.direction_check_delay
        check_end = whistle_time

        self._debug(f"      Finestra check: [{format_time(check_start)} - {format_time(check_end)}]")

        # Cerca hit di ritorno
        return_hits = []
        for motion in motion_events:
            if motion.type == return_hit_type:
                if check_start < motion.time < check_end:
                    # Estrai magnitudine
                    magnitude = getattr(motion, 'magnitude', 0)
                    if hasattr(motion, 'extra') and motion.extra:
                        magnitude = motion.extra.get('magnitude', magnitude)

                    return_hits.append((motion.time, magnitude))
                    self._debug(f"      → HIT @ {format_time(motion.time)} (mag={magnitude:.2f})")

        if not return_hits:
            self._debug(f"      Nessun hit di ritorno trovato")
            return False

        # Verifica se qualche hit supera la soglia
        for hit_time, magnitude in return_hits:
            if magnitude >= cfg.min_return_motion_magnitude:
                self._debug(f"      ⚠️ Ritorno significativo @ {format_time(hit_time)} "
                            f"(mag={magnitude:.2f} >= {cfg.min_return_motion_magnitude})")
                return True

        self._debug(f"      Hit trovati ma sotto soglia magnitudine")
        return False

    # =========================================================================
    # HELPER
    # =========================================================================

    def _get_serve_side(self, serve: Event) -> str:
        """Estrae il side dal serve event."""
        # Prova attributo diretto
        if hasattr(serve, 'side') and serve.side:
            return serve.side

        # Prova nel dict extra
        if hasattr(serve, 'extra') and serve.extra:
            return serve.extra.get('side', 'unknown')

        # Prova a dedurre dal tipo se esiste SERVE_START_LEFT/RIGHT
        if hasattr(serve, 'type'):
            type_str = str(serve.type)
            if 'LEFT' in type_str:
                return 'left'
            elif 'RIGHT' in type_str:
                return 'right'

        return 'unknown'


# =============================================================================
# FUNZIONE STANDALONE (per compatibilità)
# =============================================================================

def find_all_aces(
    events: List[Event],
    cfg: AceConfig = None,
    log_callback: Callable[[str], None] = None
) -> List[Dict[str, Any]]:
    """
    Funzione wrapper per trovare ace.

    Uso:
        aces = find_all_aces(timeline.events, cfg=AceConfig(), log_callback=self._log)
    """
    detector = AceDetector(cfg=cfg, log_callback=log_callback)
    return detector.find_aces(events)


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'AceDetector',
    'AceConfig',
    'find_all_aces',
    'format_time',
    'format_time_short',
    'format_time_range',
]
