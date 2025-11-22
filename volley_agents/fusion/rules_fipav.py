"""
Regole FIPAV: costanti e funzioni pure per validazione rally e regole tecniche.
"""

from typing import Dict, List, Optional, Tuple

from volley_agents.core.game_state import Category


# Dimensioni campo FIPAV (metri)
FIELD_LENGTH = 18.0  # lunghezza campo
FIELD_WIDTH = 9.0  # larghezza campo
NET_WIDTH = 9.5  # larghezza rete (leggermente più larga del campo)
SPACING_BACK = 6.0  # spazio dietro le linee di fondo
SPACING_FRONT = 3.0  # spazio davanti alle linee di fondo
LINE_WIDTH = 0.05  # spessore linee (5 cm)

# Altezza rete per categoria (metri)
NET_HEIGHT = {
    Category.U14F: 2.15,
    Category.U16F: 2.20,
    Category.U18F: 2.24,
    Category.SENF: 2.24,
    Category.U14M: 2.24,
    Category.U16M: 2.35,
    Category.U18M: 2.43,
    Category.SENM: 2.43,
}

# Durata rally tipica (secondi)
MIN_RALLY_DURATION = 1.0  # minimo ragionevole per un rally
MAX_RALLY_DURATION = 120.0  # massimo ragionevole (timeout di emergenza)
TYPICAL_RALLY_DURATION = 8.0  # durata tipica

# Tempi serve (secondi)
MIN_SERVE_DELAY = 0.1  # minimo tra whistle e serve
MAX_SERVE_DELAY = 8.0  # massimo tra whistle e serve (fino a 8 secondi secondo regole)

# Regole di gioco
MAX_TOUCHES_PER_SIDE = 3  # massimo numero di tocchi per parte di campo
MIN_TIME_BETWEEN_TOUCHES = 0.05  # minimo tempo tra tocchi (50ms)
MAX_TIME_BETWEEN_TOUCHES = 2.0  # massimo tempo tra tocchi nella stessa sequenza (2s)
TYPICAL_TIME_BETWEEN_TOUCHES = 0.3  # tempo tipico tra tocchi (300ms)
SET_PROBABILITY_2ND_TOUCH = 0.80  # probabilità che il 2° tocco sia un palleggio (80%)
ATTACK_PROBABILITY_3RD_TOUCH = 0.85  # probabilità che il 3° tocco sia un attacco (85%)


def get_net_height(category: Category) -> float:
    """
    Restituisce l'altezza della rete per la categoria specificata.

    Args:
        category: Categoria FIPAV

    Returns:
        Altezza rete in metri
    """
    return NET_HEIGHT.get(category, 2.24)  # default: femminile seniores


def is_valid_rally_duration(duration: float, min_duration: Optional[float] = None, max_duration: Optional[float] = None) -> bool:
    """
    Verifica se la durata di un rally è valida.

    Args:
        duration: Durata rally in secondi
        min_duration: Durata minima (default: MIN_RALLY_DURATION)
        max_duration: Durata massima (default: MAX_RALLY_DURATION)

    Returns:
        True se la durata è valida
    """
    min_dur = min_duration or MIN_RALLY_DURATION
    max_dur = max_duration or MAX_RALLY_DURATION
    return min_dur <= duration <= max_dur


def is_valid_serve_timing(whistle_time: float, serve_time: float) -> bool:
    """
    Verifica se il timing del serve rispetto al fischio è valido.

    Args:
        whistle_time: Timestamp del fischio
        serve_time: Timestamp del serve

    Returns:
        True se il timing è valido
    """
    delay = serve_time - whistle_time
    return MIN_SERVE_DELAY <= delay <= MAX_SERVE_DELAY


def update_rotation_after_point(
    current_rotation: list,
    serving_side: Optional[str],
    winner_side: Optional[str],
) -> list:
    """
    Aggiorna la rotazione dopo un punto secondo le regole FIPAV.

    Se la squadra che vince il punto era già in battuta, mantiene il servizio.
    Se la squadra che vince il punto non era in battuta, guadagna il servizio e ruota.

    Args:
        current_rotation: Rotazione corrente (6 posizioni)
        serving_side: Chi stava servendo ("left" o "right")
        winner_side: Chi ha vinto il punto ("left" o "right")

    Returns:
        Nuova rotazione (6 posizioni)
    """
    # Per ora, implementazione semplificata
    # In futuro: implementare rotazione completa secondo regole FIPAV
    if winner_side and serving_side and winner_side != serving_side:
        # La squadra che ha vinto non era in battuta: ruota
        # Ruotazione oraria: posizione 1->2, 2->3, ..., 6->1
        return current_rotation[-1:] + current_rotation[:-1]
    # Altrimenti, rotazione invariata
    return current_rotation.copy()


def calculate_pixels_per_meter(
    frame_width: int,
    frame_height: int,
    field_roi: Optional[tuple] = None,
) -> Optional[float]:
    """
    Calcola la conversione pixel/metri basandosi sulla ROI del campo.

    Args:
        frame_width: Larghezza frame in pixel
        frame_height: Altezza frame in pixel
        field_roi: ROI del campo (x, y, w, h) in pixel

    Returns:
        Fattore di conversione pixel/metri o None se non calcolabile
    """
    if field_roi is None:
        return None

    x, y, w, h = field_roi
    if w <= 0 or h <= 0:
        return None

    # Assumendo che la ROI copra l'intero campo (18m x 9m)
    # Usiamo la larghezza del campo come riferimento (9m)
    width_meters = FIELD_WIDTH  # 9 metri
    pixels_per_meter = w / width_meters
    return pixels_per_meter


def convert_pixels_to_meters(pixels: float, pixels_per_meter: Optional[float]) -> Optional[float]:
    """
    Converte pixel in metri.

    Args:
        pixels: Valore in pixel
        pixels_per_meter: Fattore di conversione

    Returns:
        Valore in metri o None se la conversione non è possibile
    """
    if pixels_per_meter is None or pixels_per_meter <= 0:
        return None
    return pixels / pixels_per_meter


def is_valid_touch_sequence(touches: List[Tuple[float, str, str]], side: str) -> Tuple[bool, Optional[str]]:
    """
    Valida una sequenza di tocchi secondo le regole FIPAV.

    Args:
        touches: Lista di (time, touch_type, side) ordinata per tempo
        side: Lato del campo ("left" o "right")

    Returns:
        (is_valid, error_message): True se valida, False con messaggio errore altrimenti
    """
    if not touches:
        return True, None

    # Filtra solo tocchi del lato specificato
    side_touches = [(t, typ, s) for t, typ, s in touches if s == side]
    
    if len(side_touches) > MAX_TOUCHES_PER_SIDE:
        return False, f"Troppi tocchi per {side}: {len(side_touches)} > {MAX_TOUCHES_PER_SIDE}"

    # Verifica timing tra tocchi
    for i in range(len(side_touches) - 1):
        t1, _, _ = side_touches[i]
        t2, _, _ = side_touches[i + 1]
        dt = t2 - t1
        
        if dt < MIN_TIME_BETWEEN_TOUCHES:
            return False, f"Tocchi troppo vicini nel tempo: {dt:.3f}s < {MIN_TIME_BETWEEN_TOUCHES}s"
        
        if dt > MAX_TIME_BETWEEN_TOUCHES:
            # Probabilmente sequenza diversa
            break

    return True, None


def classify_touch_by_position(sequence_position: int, total_touches: int) -> str:
    """
    Classifica un tocco in base alla sua posizione nella sequenza.

    Args:
        sequence_position: Posizione nella sequenza (1-based: 1, 2, 3)
        total_touches: Numero totale di tocchi nella sequenza

    Returns:
        Tipo di tocco più probabile: "reception", "set", "attack", "touch"
    """
    if sequence_position == 1:
        return "reception"  # 1° tocco = ricezione
    elif sequence_position == 2:
        # 2° tocco = palleggio nell'80% dei casi
        return "set"
    elif sequence_position == 3:
        # 3° tocco = attacco nell'85% dei casi
        return "attack"
    else:
        return "touch"  # tocco generico


def validate_rally_touches(touches: List[Tuple[float, str, str]]) -> Tuple[bool, Optional[str], Dict[str, int]]:
    """
    Valida i tocchi in un rally secondo le regole FIPAV.

    Args:
        touches: Lista di (time, touch_type, side) ordinata per tempo

    Returns:
        (is_valid, error_message, touch_count): True se valido, conteggio tocchi per lato
    """
    if not touches:
        return True, None, {"left": 0, "right": 0}

    # Conta tocchi per lato
    touch_count = {"left": 0, "right": 0}
    for _, _, side in touches:
        if side in touch_count:
            touch_count[side] += 1

    # Valida per ogni lato
    for side in ["left", "right"]:
        is_valid, error = is_valid_touch_sequence(touches, side)
        if not is_valid:
            return False, error, touch_count

    return True, None, touch_count


def infer_touch_type_from_motion(magnitude: float, position: int, total_touches: int) -> str:
    """
    Infers il tipo di tocco da magnitude del motion e posizione nella sequenza.

    Args:
        magnitude: Magnitudine del motion
        position: Posizione nella sequenza (1-based)
        total_touches: Numero totale di tocchi

    Returns:
        Tipo di tocco inferito: "reception", "set", "attack", "touch"
    """
    # Palleggio (set) tende ad avere motion più basso (movimento controllato)
    # Attacco tende ad avere motion più alto (movimento potente)
    # Ricezione varia

    base_type = classify_touch_by_position(position, total_touches)
    
    # Se magnitude è molto alta, probabilmente attacco
    if magnitude > 5.0 and position >= 2:
        return "attack"
    
    # Se magnitude è bassa e posizione è 2, probabilmente palleggio
    if magnitude < 2.5 and position == 2:
        return "set"
    
    return base_type


__all__ = [
    "FIELD_LENGTH",
    "FIELD_WIDTH",
    "NET_WIDTH",
    "LINE_WIDTH",
    "NET_HEIGHT",
    "MIN_RALLY_DURATION",
    "MAX_RALLY_DURATION",
    "TYPICAL_RALLY_DURATION",
    "MIN_SERVE_DELAY",
    "MAX_SERVE_DELAY",
    "MAX_TOUCHES_PER_SIDE",
    "MIN_TIME_BETWEEN_TOUCHES",
    "MAX_TIME_BETWEEN_TOUCHES",
    "TYPICAL_TIME_BETWEEN_TOUCHES",
    "SET_PROBABILITY_2ND_TOUCH",
    "ATTACK_PROBABILITY_3RD_TOUCH",
    "get_net_height",
    "is_valid_rally_duration",
    "is_valid_serve_timing",
    "update_rotation_after_point",
    "calculate_pixels_per_meter",
    "convert_pixels_to_meters",
    "is_valid_touch_sequence",
    "classify_touch_by_position",
    "validate_rally_touches",
    "infer_touch_type_from_motion",
]

