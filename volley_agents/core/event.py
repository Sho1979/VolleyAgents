from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    WHISTLE_START = "whistle_start"
    WHISTLE_END = "whistle_end"

    HIT_LEFT = "hit_left"
    HIT_RIGHT = "hit_right"
    MOTION_GAP = "motion_gap"

    SERVE_START = "serve_start"

    # Eventi tocchi dettagliati
    TOUCH_LEFT = "touch_left"  # tocco generico lato sinistro
    TOUCH_RIGHT = "touch_right"  # tocco generico lato destro
    SET_LEFT = "set_left"  # palleggio lato sinistro (2° tocco tipico)
    SET_RIGHT = "set_right"  # palleggio lato destro (2° tocco tipico)
    ATTACK_LEFT = "attack_left"  # attacco/schiacciata lato sinistro (3° tocco tipico)
    ATTACK_RIGHT = "attack_right"  # attacco/schiacciata lato destro (3° tocco tipico)
    RECEPTION_LEFT = "reception_left"  # ricezione lato sinistro (1° tocco tipico)
    RECEPTION_RIGHT = "reception_right"  # ricezione lato destro (1° tocco tipico)
    BLOCK_LEFT = "block_left"  # muro lato sinistro
    BLOCK_RIGHT = "block_right"  # muro lato destro

    REF_MOTION = "ref_motion"          # per ora solo log
    REF_SERVE_READY = "ref_serve_ready"
    REF_SERVE_RELEASE = "ref_serve_release"
    REF_POINT_LEFT = "ref_point_left"
    REF_POINT_RIGHT = "ref_point_right"

    SCORE_CHANGE = "score_change"

    # Eventi campo/tecnici
    FIELD_MODEL_READY = "field_model_ready"

    # Eventi ruoli/rotazioni
    ROTATION_UPDATE = "rotation_update"
    PLAYER_ROLE_IN_POSITION = "player_role_in_position"

    # Eventi corpo/misurazioni
    BODY_MEASURE = "body_measure"

    # Eventi salto
    JUMP_EVENT = "jump_event"
    JUMP_PEAK = "jump_peak"


@dataclass
class Event:
    time: float               # tempo nel video (secondi)
    type: EventType           # tipo di evento
    confidence: float = 1.0   # 0-1
    extra: Optional[dict] = None   # dati aggiuntivi, flessibile

    def __repr__(self):
        return f"Event(t={self.time:.2f}, type={self.type}, conf={self.confidence:.2f})"

