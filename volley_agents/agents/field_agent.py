"""
FieldAgent: agente tecnico per campo/rete FIPAV.

Responsabilità:
- Caricare tabella FIPAV da rules_fipav.py
- Calcolare mappatura pixel → metri
- Pubblicare FIELD_MODEL_READY con informazioni campo/rete
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

from volley_agents.core.event import Event, EventType
from volley_agents.core.game_state import Category
from volley_agents.fusion.rules_fipav import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    NET_WIDTH,
    get_net_height,
    calculate_pixels_per_meter,
)

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline


@dataclass
class FieldAgentConfig:
    """Configurazione per FieldAgent."""

    category: Category = Category.SENF
    field_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in pixel
    net_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in pixel
    frame_width: int = 1920
    frame_height: int = 1080


class FieldAgent:
    """
    Agente tecnico per campo/rete FIPAV.

    Calcola:
    - mappatura pixel → metri
    - posizione rete in pixel
    - altezza rete per categoria
    """

    def __init__(self, config: Optional[FieldAgentConfig] = None):
        self.config = config or FieldAgentConfig()

    def run(
        self,
        timeline: Optional["Timeline"] = None,
    ) -> List[Event]:
        """
        Analizza la configurazione campo e pubblica FIELD_MODEL_READY.

        Args:
            timeline: Timeline opzionale per aggiungere eventi

        Returns:
            Lista di eventi FIELD_MODEL_READY
        """
        events = self.analyze()
        if timeline is not None:
            timeline.extend(events)
        return events

    def analyze(self) -> List[Event]:
        """
        Analizza la configurazione campo e calcola mappatura pixel/metri.

        Returns:
            Lista di eventi FIELD_MODEL_READY
        """
        cfg = self.config

        # Calcola altezza rete per categoria
        net_height_m = get_net_height(cfg.category)

        # Calcola pixel/metri se field_roi è definito
        pixels_per_meter = None
        if cfg.field_roi is not None:
            pixels_per_meter = calculate_pixels_per_meter(
                cfg.frame_width,
                cfg.frame_height,
                cfg.field_roi,
            )

        # Crea evento FIELD_MODEL_READY
        extra = {
            "category": cfg.category.value,
            "net_height_m": net_height_m,
            "field_length_m": FIELD_LENGTH,
            "field_width_m": FIELD_WIDTH,
            "net_width_m": NET_WIDTH,
            "pixels_per_meter": pixels_per_meter,
            "field_roi": cfg.field_roi,
            "net_roi": cfg.net_roi,
            "frame_width": cfg.frame_width,
            "frame_height": cfg.frame_height,
        }

        event = Event(
            time=0.0,  # evento all'inizio
            type=EventType.FIELD_MODEL_READY,
            confidence=1.0,
            extra=extra,
        )

        return [event]

    def get_pixels_per_meter(self) -> Optional[float]:
        """
        Restituisce il fattore di conversione pixel/metri.

        Returns:
            Fattore di conversione o None se non calcolabile
        """
        cfg = self.config
        if cfg.field_roi is None:
            return None
        return calculate_pixels_per_meter(
            cfg.frame_width,
            cfg.frame_height,
            cfg.field_roi,
        )

    def get_net_height(self) -> float:
        """
        Restituisce l'altezza della rete per la categoria corrente.

        Returns:
            Altezza rete in metri
        """
        cfg = self.config
        return get_net_height(cfg.category)


__all__ = [
    "FieldAgent",
    "FieldAgentConfig",
]

