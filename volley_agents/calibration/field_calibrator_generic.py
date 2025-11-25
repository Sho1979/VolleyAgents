"""
VolleyAgents - Field Calibrator (Generic)

=========================================

Sistema di calibrazione campo GENERICO per qualsiasi video di pallavolo.

Il campo è sempre standard (18x9m), quello che cambia è la POSIZIONE CAMERA.

POSIZIONI CAMERA SUPPORTATE:

────────────────────────────

1. SIDE (laterale) - La più comune per riprese amatoriali

   - Vedi arbitro di spalle o di lato

   - Campo A a sinistra, Campo B a destra

   - Rete verticale al centro

   

2. BEHIND (dietro) - Tipica delle riprese TV professionali

   - Vedi campo lontano in alto, campo vicino in basso

   - Rete orizzontale al centro

UTILIZZO:

─────────

    # Metodo 1: Preset

    field = FieldCalibrator(camera=CameraPosition.SIDE)

    

    # Metodo 2: Calibrazione interattiva (GUI)

    field.calibrate_interactive(frame)

    

    # Metodo 3: Calibrazione da punti

    field.calibrate_from_points(net_center=(960, 400), ...)

Autore: VolleyAgents Project

Versione: 4.0 (Generic)

"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List, Callable
import json

# =============================================================================
# ENUMS
# =============================================================================

class CameraPosition(Enum):
    """Posizione camera rispetto al campo"""
    SIDE = "side"           # Laterale (la più comune)
    BEHIND = "behind"       # Dietro il campo
    CUSTOM = "custom"       # Calibrazione manuale

class FieldZone(Enum):
    """Zone del campo"""
    # Campo A (sinistra per SIDE, lontano per BEHIND)
    SERVICE_A = "service_a"     # Zona servizio squadra A
    BACK_A = "back_a"           # Zona difesa squadra A (6m)
    FRONT_A = "front_a"         # Zona attacco squadra A (3m)
    
    # Rete
    NET = "net"
    
    # Campo B (destra per SIDE, vicino per BEHIND)
    FRONT_B = "front_b"         # Zona attacco squadra B (3m)
    BACK_B = "back_b"           # Zona difesa squadra B (6m)
    SERVICE_B = "service_b"     # Zona servizio squadra B
    
    # Fuori campo
    OUT = "out"

class Team(Enum):
    """Squadra"""
    A = "A"  # Squadra A (sinistra/lontano)
    B = "B"  # Squadra B (destra/vicino)

# =============================================================================
# CONFIGURAZIONE ZONE
# =============================================================================

@dataclass
class ZoneBounds:
    """
    Limiti delle zone in coordinate normalizzate (0-1).
    
    Per camera SIDE: i valori sono X (orizzontale)
    Per camera BEHIND: i valori sono Y (verticale)
    """
    service_a: Tuple[float, float] = (0.00, 0.10)
    back_a: Tuple[float, float] = (0.10, 0.27)
    front_a: Tuple[float, float] = (0.27, 0.44)
    net: Tuple[float, float] = (0.44, 0.56)
    front_b: Tuple[float, float] = (0.56, 0.73)
    back_b: Tuple[float, float] = (0.73, 0.90)
    service_b: Tuple[float, float] = (0.90, 1.00)
    
    # Limiti campo sull'altro asse
    court_min: float = 0.25  # Bordo superiore/sinistro campo
    court_max: float = 0.85  # Bordo inferiore/destro campo

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# Camera LATERALE (SIDE) - Default
PRESET_SIDE = ZoneBounds(
    service_a=(0.00, 0.10),
    back_a=(0.10, 0.27),
    front_a=(0.27, 0.44),
    net=(0.44, 0.56),
    front_b=(0.56, 0.73),
    back_b=(0.73, 0.90),
    service_b=(0.90, 1.00),
    court_min=0.28,
    court_max=0.85,
)

# Camera DIETRO (BEHIND)
PRESET_BEHIND = ZoneBounds(
    service_a=(0.02, 0.12),   # FAR service (in alto)
    back_a=(0.12, 0.28),
    front_a=(0.28, 0.44),
    net=(0.44, 0.56),
    front_b=(0.56, 0.72),
    back_b=(0.72, 0.88),
    service_b=(0.88, 0.98),   # NEAR service (in basso)
    court_min=0.10,
    court_max=0.90,
)

# =============================================================================
# FIELD CALIBRATOR
# =============================================================================

class FieldCalibrator:
    """
    Calibratore campo generico.
    
    Uso base:
        field = FieldCalibrator(camera=CameraPosition.SIDE)
        field.set_frame_size(1920, 1080)
        zone = field.get_zone(x, y)
    
    Con calibrazione:
        field = FieldCalibrator()
        field.calibrate_from_points(
            net_left=0.45,
            net_right=0.55,
            court_top=0.30,
            court_bottom=0.85
        )
    """
    
    def __init__(self, 
                 camera: CameraPosition = CameraPosition.SIDE,
                 log_callback: Optional[Callable] = None):
        self._log = log_callback or (lambda x: None)
        self.camera = camera
        self.frame_width: int = 1920
        self.frame_height: int = 1080
        
        # Carica preset
        self._bounds = self._get_preset(camera)
        self._log(f"[Field] Camera: {camera.value}")
    
    def _get_preset(self, camera: CameraPosition) -> ZoneBounds:
        """Carica preset per posizione camera"""
        if camera == CameraPosition.SIDE:
            return ZoneBounds(**vars(PRESET_SIDE))
        elif camera == CameraPosition.BEHIND:
            return ZoneBounds(**vars(PRESET_BEHIND))
        else:
            return ZoneBounds()
    
    # -------------------------------------------------------------------------
    # SETUP
    # -------------------------------------------------------------------------
    
    def set_frame_size(self, width: int, height: int) -> None:
        """Imposta dimensioni frame"""
        self.frame_width = width
        self.frame_height = height
    
    def set_camera(self, camera: CameraPosition) -> None:
        """Cambia posizione camera"""
        self.camera = camera
        self._bounds = self._get_preset(camera)
        self._log(f"[Field] Camera cambiata: {camera.value}")
    
    # -------------------------------------------------------------------------
    # CALIBRAZIONE
    # -------------------------------------------------------------------------
    
    def calibrate_from_points(self,
                               net_left: float,
                               net_right: float,
                               court_top: Optional[float] = None,
                               court_bottom: Optional[float] = None) -> None:
        """
        Calibra le zone specificando i punti chiave.
        
        Per camera SIDE:
            net_left: X normalizzato del bordo sinistro della rete
            net_right: X normalizzato del bordo destro della rete
            court_top: Y normalizzato del bordo superiore campo
            court_bottom: Y normalizzato del bordo inferiore campo
        
        Per camera BEHIND:
            net_left: Y normalizzato del bordo superiore della rete
            net_right: Y normalizzato del bordo inferiore della rete
            court_top/bottom: X normalizzato dei bordi laterali
        
        Esempio:
            field.calibrate_from_points(
                net_left=0.45,
                net_right=0.55,
                court_top=0.30,
                court_bottom=0.85
            )
        """
        # Calcola proporzioni standard del campo
        # Campo: 9m servizio | 6m back | 3m front | RETE | 3m front | 6m back | 9m servizio
        # Proporzion campo giocabile: back=6m, front=3m → back=2/3, front=1/3 di ogni lato
        
        left_space = net_left  # Spazio a sinistra della rete
        right_space = 1.0 - net_right  # Spazio a destra della rete
        
        # Zona servizio = ~10% dello spazio totale per lato
        # Back = ~55% del campo per lato
        # Front = ~35% del campo per lato
        
        service_a_end = left_space * 0.15
        back_a_end = left_space * 0.55
        front_a_end = net_left
        
        front_b_start = net_right
        front_b_end = net_right + right_space * 0.35
        back_b_end = net_right + right_space * 0.85
        service_b_start = net_right + right_space * 0.85
        
        self._bounds = ZoneBounds(
            service_a=(0.0, service_a_end),
            back_a=(service_a_end, back_a_end),
            front_a=(back_a_end, front_a_end),
            net=(net_left, net_right),
            front_b=(front_b_start, front_b_end),
            back_b=(front_b_end, back_b_end),
            service_b=(service_b_start, 1.0),
            court_min=court_top or 0.25,
            court_max=court_bottom or 0.85,
        )
        
        self.camera = CameraPosition.CUSTOM
        self._log(f"[Field] Calibrato: rete={net_left:.2f}-{net_right:.2f}")
    
    def calibrate_from_pixels(self,
                               net_left_px: int,
                               net_right_px: int,
                               court_top_px: Optional[int] = None,
                               court_bottom_px: Optional[int] = None) -> None:
        """Calibra usando coordinate pixel invece che normalizzate"""
        net_left = net_left_px / self.frame_width
        net_right = net_right_px / self.frame_width
        court_top = court_top_px / self.frame_height if court_top_px else None
        court_bottom = court_bottom_px / self.frame_height if court_bottom_px else None
        
        self.calibrate_from_points(net_left, net_right, court_top, court_bottom)
    
    # -------------------------------------------------------------------------
    # ZONE DETECTION
    # -------------------------------------------------------------------------
    
    def get_zone(self, x: int, y: int) -> FieldZone:
        """
        Determina la zona del campo da coordinate pixel.
        
        Returns:
            FieldZone corrispondente alla posizione
        """
        # Normalizza coordinate
        x_norm = x / self.frame_width if self.frame_width > 0 else 0.5
        y_norm = y / self.frame_height if self.frame_height > 0 else 0.5
        
        return self.get_zone_normalized(x_norm, y_norm)
    
    def get_zone_normalized(self, x_norm: float, y_norm: float) -> FieldZone:
        """Determina zona da coordinate normalizzate (0-1)"""
        
        if self.camera in (CameraPosition.SIDE, CameraPosition.CUSTOM):
            return self._get_zone_side(x_norm, y_norm)
        else:  # BEHIND
            return self._get_zone_behind(x_norm, y_norm)
    
    def _get_zone_side(self, x_norm: float, y_norm: float) -> FieldZone:
        """
        Zone per camera LATERALE.
        X = profondità campo (sinistra-destra)
        Y = posizione verticale
        """
        b = self._bounds
        
        # Verifica se dentro il campo (Y)
        if y_norm < b.court_min or y_norm > b.court_max:
            return FieldZone.OUT
        
        # Determina zona in base a X
        if b.service_a[0] <= x_norm < b.service_a[1]:
            return FieldZone.SERVICE_A
        elif b.back_a[0] <= x_norm < b.back_a[1]:
            return FieldZone.BACK_A
        elif b.front_a[0] <= x_norm < b.front_a[1]:
            return FieldZone.FRONT_A
        elif b.net[0] <= x_norm < b.net[1]:
            return FieldZone.NET
        elif b.front_b[0] <= x_norm < b.front_b[1]:
            return FieldZone.FRONT_B
        elif b.back_b[0] <= x_norm < b.back_b[1]:
            return FieldZone.BACK_B
        elif b.service_b[0] <= x_norm <= b.service_b[1]:
            return FieldZone.SERVICE_B
        else:
            return FieldZone.OUT
    
    def _get_zone_behind(self, x_norm: float, y_norm: float) -> FieldZone:
        """
        Zone per camera DIETRO.
        Y = profondità campo (lontano-vicino)
        X = posizione laterale
        """
        b = self._bounds
        
        # Verifica se dentro il campo (X)
        if x_norm < b.court_min or x_norm > b.court_max:
            return FieldZone.OUT
        
        # Determina zona in base a Y
        if b.service_a[0] <= y_norm < b.service_a[1]:
            return FieldZone.SERVICE_A
        elif b.back_a[0] <= y_norm < b.back_a[1]:
            return FieldZone.BACK_A
        elif b.front_a[0] <= y_norm < b.front_a[1]:
            return FieldZone.FRONT_A
        elif b.net[0] <= y_norm < b.net[1]:
            return FieldZone.NET
        elif b.front_b[0] <= y_norm < b.front_b[1]:
            return FieldZone.FRONT_B
        elif b.back_b[0] <= y_norm < b.back_b[1]:
            return FieldZone.BACK_B
        elif b.service_b[0] <= y_norm <= b.service_b[1]:
            return FieldZone.SERVICE_B
        else:
            return FieldZone.OUT
    
    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------
    
    def is_service_zone(self, zone: FieldZone) -> bool:
        """Verifica se è zona servizio"""
        return zone in (FieldZone.SERVICE_A, FieldZone.SERVICE_B)
    
    def is_court_zone(self, zone: FieldZone) -> bool:
        """Verifica se è dentro il campo di gioco"""
        return zone in (FieldZone.BACK_A, FieldZone.FRONT_A,
                        FieldZone.FRONT_B, FieldZone.BACK_B)
    
    def get_team(self, zone: FieldZone) -> Optional[Team]:
        """Determina squadra dalla zona"""
        if zone in (FieldZone.SERVICE_A, FieldZone.BACK_A, FieldZone.FRONT_A):
            return Team.A
        elif zone in (FieldZone.SERVICE_B, FieldZone.BACK_B, FieldZone.FRONT_B):
            return Team.B
        return None
    
    def get_receiving_zones(self, serving_team: Team) -> List[FieldZone]:
        """Zone del campo ricevente"""
        if serving_team == Team.A:
            return [FieldZone.FRONT_B, FieldZone.BACK_B]
        else:
            return [FieldZone.FRONT_A, FieldZone.BACK_A]
    
    def get_receiving_range(self, serving_team: Team) -> Tuple[float, float]:
        """
        Range (normalizzato) del campo ricevente.
        
        Per camera SIDE: range X
        Per camera BEHIND: range Y
        """
        b = self._bounds
        if serving_team == Team.A:
            # A serve → B riceve
            return (b.front_b[0], b.back_b[1])
        else:
            # B serve → A riceve
            return (b.back_a[0], b.front_a[1])
    
    def get_primary_axis(self) -> str:
        """
        Asse principale per la profondità del campo.
        
        Returns:
            'x' per camera SIDE, 'y' per camera BEHIND
        """
        return 'x' if self.camera != CameraPosition.BEHIND else 'y'
    
    # -------------------------------------------------------------------------
    # SAVE/LOAD
    # -------------------------------------------------------------------------
    
    def save(self, path: str) -> bool:
        """Salva configurazione"""
        try:
            data = {
                "camera": self.camera.value,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "bounds": vars(self._bounds)
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            self._log(f"[Field] Errore save: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Carica configurazione"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.camera = CameraPosition(data.get("camera", "side"))
            self.frame_width = data.get("frame_width", 1920)
            self.frame_height = data.get("frame_height", 1080)
            
            b = data.get("bounds", {})
            self._bounds = ZoneBounds(
                service_a=tuple(b.get("service_a", (0, 0.1))),
                back_a=tuple(b.get("back_a", (0.1, 0.27))),
                front_a=tuple(b.get("front_a", (0.27, 0.44))),
                net=tuple(b.get("net", (0.44, 0.56))),
                front_b=tuple(b.get("front_b", (0.56, 0.73))),
                back_b=tuple(b.get("back_b", (0.73, 0.9))),
                service_b=tuple(b.get("service_b", (0.9, 1.0))),
                court_min=b.get("court_min", 0.25),
                court_max=b.get("court_max", 0.85),
            )
            return True
        except Exception as e:
            self._log(f"[Field] Errore load: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # DEBUG
    # -------------------------------------------------------------------------
    
    def get_bounds(self) -> ZoneBounds:
        """Restituisce i bounds attuali"""
        return self._bounds
    
    def print_schema(self) -> None:
        """Stampa schema visivo"""
        b = self._bounds
        axis = "X" if self.camera != CameraPosition.BEHIND else "Y"
        
        print(f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║  FIELD CALIBRATOR - Camera: {self.camera.value.upper():<10} - Asse principale: {axis}              ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  {axis}=0.0                                                              {axis}=1.0  ║
║  │                                                                        │    ║
║  │ ┌────────┬──────────┬──────────┬────────┬──────────┬──────────┬───────┐│    ║
║  │ │ SERVICE│   BACK   │  FRONT   │        │  FRONT   │   BACK   │SERVICE││    ║
║  │ │   A    │    A     │    A     │  RETE  │    B     │    B     │   B   ││    ║
║  │ │        │          │          │        │          │          │       ││    ║
║  │ │  {b.service_a[0]:.2f}   │   {b.back_a[0]:.2f}   │   {b.front_a[0]:.2f}   │  {b.net[0]:.2f}  │   {b.front_b[0]:.2f}   │   {b.back_b[0]:.2f}   │  {b.service_b[0]:.2f} ││    ║
║  │ │ -{b.service_a[1]:.2f}  │  -{b.back_a[1]:.2f}  │  -{b.front_a[1]:.2f}  │ -{b.net[1]:.2f} │  -{b.front_b[1]:.2f}  │  -{b.back_b[1]:.2f}  │ -{b.service_b[1]:.2f}││    ║
║  │ └────────┴──────────┴──────────┴────────┴──────────┴──────────┴───────┘│    ║
║  │                                                                        │    ║
║                                                                                ║
║  Court bounds: {b.court_min:.2f} - {b.court_max:.2f}                                              ║
║                                                                                ║
║  SERVE:  Team A batte da SERVICE_A → palla va verso B                          ║
║          Team B batte da SERVICE_B → palla va verso A                          ║
╚════════════════════════════════════════════════════════════════════════════════╝
        """)

# =============================================================================
# ACE DETECTOR (semplificato)
# =============================================================================

@dataclass
class ServeEvent:
    """Evento servizio"""
    timestamp: float
    team: Team
    confidence: float
    x_norm: float
    y_norm: float
    magnitude: float = 0.0

@dataclass  
class AceEvent:
    """Evento ace confermato"""
    serve: ServeEvent
    whistle_time: float
    duration: float

class AceDetector:
    """
    Rilevatore ACE semplificato.
    
    ACE = Serve + Fischio entro 2-5s + Nessun tocco nel campo ricevente
    """
    
    def __init__(self, field: FieldCalibrator, log_callback: Optional[Callable] = None):
        self.field = field
        self._log = log_callback or (lambda x: None)
        
        # Configurazione
        self.min_ace_duration = 2.0  # Minimo secondi
        self.max_ace_duration = 5.0  # Massimo secondi
        self.min_serve_confidence = 0.75
        self.max_receiving_motion = 1.0
    
    def detect_serve(self, x: int, y: int, timestamp: float, 
                     magnitude: float) -> Optional[ServeEvent]:
        """
        Rileva se un movimento è un servizio.
        
        Returns:
            ServeEvent se rilevato, None altrimenti
        """
        zone = self.field.get_zone(x, y)
        
        if not self.field.is_service_zone(zone):
            return None
        
        if magnitude < 1.5:  # Soglia minima
            return None
        
        team = self.field.get_team(zone)
        confidence = min(0.95, 0.70 + magnitude * 0.08)
        
        x_norm = x / self.field.frame_width
        y_norm = y / self.field.frame_height
        
        serve = ServeEvent(
            timestamp=timestamp,
            team=team,
            confidence=confidence,
            x_norm=x_norm,
            y_norm=y_norm,
            magnitude=magnitude
        )
        
        self._log(f"[Ace] 🎯 Serve {team.value} @ {timestamp:.1f}s (conf={confidence:.2f})")
        return serve
    
    def check_ace(self, serve: ServeEvent, 
                  whistle_time: float,
                  receiving_motion: float) -> Optional[AceEvent]:
        """
        Verifica se un serve + fischio è un ACE.
        
        Args:
            serve: Evento serve
            whistle_time: Timestamp del fischio
            receiving_motion: Quantità di movimento nel campo ricevente
        
        Returns:
            AceEvent se confermato, None altrimenti
        """
        duration = whistle_time - serve.timestamp
        
        # Verifica timing
        if duration < self.min_ace_duration:
            self._log(f"[Ace] ✗ Durata troppo breve: {duration:.2f}s")
            return None
        
        if duration > self.max_ace_duration:
            self._log(f"[Ace] ✗ Durata troppo lunga: {duration:.2f}s")
            return None
        
        # Verifica confidence serve
        if serve.confidence < self.min_serve_confidence:
            self._log(f"[Ace] ✗ Serve confidence bassa: {serve.confidence:.2f}")
            return None
        
        # Verifica movimento nel campo ricevente
        if receiving_motion > self.max_receiving_motion:
            self._log(f"[Ace] ✗ Risposta rilevata: motion={receiving_motion:.2f}")
            return None
        
        ace = AceEvent(
            serve=serve,
            whistle_time=whistle_time,
            duration=duration
        )
        
        self._log(f"[Ace] ⚡ ACE CONFERMATO! Team {serve.team.value} @ {serve.timestamp:.1f}s "
                  f"(dur={duration:.2f}s)")
        return ace
    
    def calculate_receiving_motion(self, 
                                    serve: ServeEvent,
                                    motion_events: List[dict],
                                    window_seconds: float = 1.5) -> float:
        """
        Calcola il movimento nel campo ricevente dopo il serve.
        
        Args:
            serve: Evento serve
            motion_events: Lista di eventi motion (dict con x, y, timestamp, magnitude)
            window_seconds: Finestra temporale da controllare
        
        Returns:
            Media delle magnitudini nel campo ricevente
        """
        recv_range = self.field.get_receiving_range(serve.team)
        primary_axis = self.field.get_primary_axis()
        
        t_start = serve.timestamp
        t_end = serve.timestamp + window_seconds
        
        magnitudes = []
        for evt in motion_events:
            # Controllo tempo
            if not (t_start <= evt['timestamp'] <= t_end):
                continue
            
            # Normalizza coordinate
            x_norm = evt['x'] / self.field.frame_width
            y_norm = evt['y'] / self.field.frame_height
            
            # Controllo zona (asse principale)
            pos = x_norm if primary_axis == 'x' else y_norm
            if recv_range[0] <= pos <= recv_range[1]:
                magnitudes.append(evt.get('magnitude', 1.0))
        
        return sum(magnitudes) / len(magnitudes) if magnitudes else 0.0

# =============================================================================
# HELPER
# =============================================================================

def fmt_time(seconds: float) -> str:
    """Formatta secondi in MM:SS"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("VolleyAgents - Field Calibrator (Generic)")
    print("=" * 80)
    
    # Test camera SIDE (default)
    print("\n[TEST 1] Camera SIDE (laterale)")
    field = FieldCalibrator(camera=CameraPosition.SIDE)
    field.set_frame_size(1920, 1080)
    field.print_schema()
    
    print("\nTest zone detection:")
    tests = [
        (50, 500, "SERVICE_A"),
        (300, 500, "BACK_A"),
        (700, 500, "FRONT_A"),
        (960, 500, "NET"),
        (1200, 500, "FRONT_B"),
        (1500, 500, "BACK_B"),
        (1850, 500, "SERVICE_B"),
        (960, 100, "OUT (sopra)"),
    ]
    
    for x, y, expected in tests:
        zone = field.get_zone(x, y)
        x_norm = x / 1920
        print(f"  ({x:4d}, {y}) x={x_norm:.2f} → {zone.value:12s} (atteso: {expected})")
    
    # Test calibrazione manuale
    print("\n" + "=" * 80)
    print("[TEST 2] Calibrazione manuale")
    
    field2 = FieldCalibrator()
    field2.set_frame_size(1920, 1080)
    field2.calibrate_from_points(
        net_left=0.42,
        net_right=0.58,
        court_top=0.32,
        court_bottom=0.82
    )
    field2.print_schema()
    
    # Test receiving zones
    print("\n[TEST 3] Receiving zones")
    for team in [Team.A, Team.B]:
        zones = field.get_receiving_zones(team)
        range_ = field.get_receiving_range(team)
        print(f"  Team {team.value} serve → riceve in {[z.value for z in zones]}")
        print(f"    Range: {range_[0]:.2f} - {range_[1]:.2f}")
    
    # Test camera BEHIND
    print("\n" + "=" * 80)
    print("[TEST 4] Camera BEHIND")
    field3 = FieldCalibrator(camera=CameraPosition.BEHIND)
    field3.set_frame_size(1920, 1080)
    field3.print_schema()
    
    print("\n" + "=" * 80)
    print("Test completato!")

