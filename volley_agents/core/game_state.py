"""
GameState: rappresenta lo stato attuale della partita.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Category(str, Enum):
    """Categorie FIPAV."""

    U14F = "U14F"  # Under 14 Femminile
    U16F = "U16F"  # Under 16 Femminile
    U18F = "U18F"  # Under 18 Femminile
    SENF = "SENF"  # Seniores Femminile
    U14M = "U14M"  # Under 14 Maschile
    U16M = "U16M"  # Under 16 Maschile
    U18M = "U18M"  # Under 18 Maschile
    SENM = "SENM"  # Seniores Maschile


class Position(str, Enum):
    """Posizioni in campo (1-6)."""

    P1 = "1"  # Zona battuta (dietro-destra)
    P2 = "2"  # Zona attacco (avanti-destra)
    P3 = "3"  # Zona attacco (avanti-centro)
    P4 = "4"  # Zona attacco (avanti-sinistra)
    P5 = "5"  # Zona difesa (dietro-sinistra)
    P6 = "6"  # Zona difesa (dietro-centro)


class Role(str, Enum):
    """Ruoli dei giocatori."""

    P = "P"  # Palleggiatore
    OH = "OH"  # Opposto
    MB = "MB"  # Centrale
    OPP = "OPP"  # Opposto
    L = "L"  # Libero


@dataclass
class Player:
    """Giocatore con ruolo e numero."""

    player_id: str
    number: int
    role: Role
    name: Optional[str] = None


@dataclass
class Team:
    """Squadra con elenco giocatori."""

    team_id: str  # "left" o "right"
    name: Optional[str] = None
    players: List[Player] = field(default_factory=list)
    rotation: List[Optional[str]] = field(default_factory=lambda: [None] * 6)  # player_id per posizione 1-6


@dataclass
class GameState:
    """
    Stato attuale della partita.

    - score: punteggio corrente (left, right)
    - serving_side: chi sta servendo ("left" o "right")
    - category: categoria FIPAV
    - teams: modello delle squadre
    - current_rotation: rotazione corrente per ogni squadra
    """

    score_left: int = 0
    score_right: int = 0
    serving_side: Optional[str] = None  # "left" o "right"
    category: Category = Category.SENF
    teams: Dict[str, Team] = field(default_factory=dict)
    current_rotation: Dict[str, List[Optional[str]]] = field(
        default_factory=lambda: {"left": [None] * 6, "right": [None] * 6}
    )

    def get_team(self, side: str) -> Optional[Team]:
        """Restituisce la squadra per il lato specificato."""
        return self.teams.get(side)

    def update_score(self, side: str, new_score: int):
        """Aggiorna il punteggio per la squadra specificata."""
        if side == "left":
            self.score_left = new_score
        elif side == "right":
            self.score_right = new_score

    def set_serving_side(self, side: Optional[str]):
        """Imposta chi sta servendo."""
        self.serving_side = side

    def __repr__(self):
        return f"GameState(score={self.score_left}-{self.score_right}, serving={self.serving_side}, cat={self.category.value})"


__all__ = [
    "GameState",
    "Category",
    "Position",
    "Role",
    "Player",
    "Team",
]

