"""
VolleyAgents - Calibrazione Interattiva

========================================

Interfaccia grafica per calibrare il campo cliccando sui punti chiave.

UTILIZZO:

─────────

    python -m volley_agents.calibration.field_calibrator_gui video.mp4

    

    oppure da codice:

    

    from volley_agents.calibration.field_calibrator_gui import InteractiveCalibrator

    

    calibrator = InteractiveCalibrator()

    config = calibrator.calibrate_from_video("video.mp4")

    config = calibrator.calibrate_from_frame(frame)

ISTRUZIONI:

───────────

1. Clicca sul BORDO SINISTRO della rete

2. Clicca sul BORDO DESTRO della rete

3. Clicca sul BORDO SUPERIORE del campo

4. Clicca sul BORDO INFERIORE del campo

5. Premi INVIO per confermare o 'r' per ricominciare

Autore: VolleyAgents Project

"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable
from pathlib import Path

# Import del calibratore principale
from volley_agents.calibration.field_calibrator_generic import (
    FieldCalibrator,
    CameraPosition,
    ZoneBounds,
)

# =============================================================================
# CALIBRATION RESULT
# =============================================================================

@dataclass
class CalibrationResult:
    """Risultato della calibrazione"""
    camera: CameraPosition
    net_left: float       # X normalizzato bordo sinistro rete
    net_right: float      # X normalizzato bordo destro rete
    court_top: float      # Y normalizzato bordo superiore
    court_bottom: float   # Y normalizzato bordo inferiore
    frame_width: int
    frame_height: int
    
    def to_bounds(self) -> ZoneBounds:
        """Converte in ZoneBounds"""
        left_space = self.net_left
        right_space = 1.0 - self.net_right
        
        service_a_end = left_space * 0.15
        back_a_end = left_space * 0.55
        
        front_b_end = self.net_right + right_space * 0.35
        back_b_end = self.net_right + right_space * 0.85
        
        return ZoneBounds(
            service_a=(0.0, service_a_end),
            back_a=(service_a_end, back_a_end),
            front_a=(back_a_end, self.net_left),
            net=(self.net_left, self.net_right),
            front_b=(self.net_right, front_b_end),
            back_b=(front_b_end, back_b_end),
            service_b=(back_b_end, 1.0),
            court_min=self.court_top,
            court_max=self.court_bottom,
        )
    
    def save(self, path: str) -> None:
        """Salva configurazione"""
        import json
        data = {
            "camera": self.camera.value,
            "net_left": self.net_left,
            "net_right": self.net_right,
            "court_top": self.court_top,
            "court_bottom": self.court_bottom,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> 'CalibrationResult':
        """Carica configurazione"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return CalibrationResult(
            camera=CameraPosition(data["camera"]),
            net_left=data["net_left"],
            net_right=data["net_right"],
            court_top=data["court_top"],
            court_bottom=data["court_bottom"],
            frame_width=data["frame_width"],
            frame_height=data["frame_height"],
        )

# =============================================================================
# INTERACTIVE CALIBRATOR
# =============================================================================

class InteractiveCalibrator:
    """
    Calibratore interattivo con interfaccia grafica OpenCV.
    
    Permette di calibrare il campo cliccando su 4 punti:
    1. Bordo sinistro rete
    2. Bordo destro rete  
    3. Bordo superiore campo
    4. Bordo inferiore campo
    """
    
    WINDOW_NAME = "VolleyAgents - Calibrazione Campo"
    
    # Colori (BGR)
    COLOR_NET = (0, 255, 255)      # Giallo
    COLOR_COURT = (0, 255, 0)      # Verde
    COLOR_ZONES = (255, 100, 100)  # Blu chiaro
    COLOR_TEXT = (255, 255, 255)   # Bianco
    COLOR_HINT = (100, 100, 255)   # Rosso chiaro
    
    def __init__(self):
        self.clicks: List[Tuple[int, int]] = []
        self.frame: Optional[np.ndarray] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.camera: CameraPosition = CameraPosition.SIDE
        self._done: bool = False
        self._cancelled: bool = False
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback per click del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicks) < 4:
                self.clicks.append((x, y))
                self._update_display()
    
    def _update_display(self):
        """Aggiorna visualizzazione con punti e zone"""
        if self.frame is None:
            return
        
        display = self.frame.copy()
        h, w = display.shape[:2]
        
        # Disegna istruzioni
        instructions = [
            "CALIBRAZIONE CAMPO",
            "─" * 30,
            "1. Clicca BORDO SINISTRO rete",
            "2. Clicca BORDO DESTRO rete",
            "3. Clicca BORDO SUPERIORE campo",
            "4. Clicca BORDO INFERIORE campo",
            "",
            "INVIO = Conferma",
            "R = Ricomincia",
            "ESC = Annulla",
            "C = Cambia camera (SIDE/BEHIND)",
        ]
        
        # Box istruzioni
        cv2.rectangle(display, (10, 10), (350, 280), (0, 0, 0), -1)
        cv2.rectangle(display, (10, 10), (350, 280), self.COLOR_TEXT, 2)
        
        for i, text in enumerate(instructions):
            color = self.COLOR_HINT if i == len(self.clicks) + 2 else self.COLOR_TEXT
            cv2.putText(display, text, (20, 35 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        
        # Mostra camera attuale
        cam_text = f"Camera: {self.camera.value.upper()}"
        cv2.putText(display, cam_text, (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_NET, 2)
        
        # Disegna punti cliccati
        point_labels = ["Rete SX", "Rete DX", "Campo TOP", "Campo BOTTOM"]
        
        for i, (px, py) in enumerate(self.clicks):
            # Cerchio sul punto
            cv2.circle(display, (px, py), 8, self.COLOR_NET, -1)
            cv2.circle(display, (px, py), 10, self.COLOR_TEXT, 2)
            
            # Label
            label = f"{i+1}. {point_labels[i]}"
            cv2.putText(display, label, (px + 15, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 2)
        
        # Se abbiamo almeno 2 punti, disegna la rete
        if len(self.clicks) >= 2:
            net_left, net_right = self.clicks[0][0], self.clicks[1][0]
            # Linea verticale rete (approssimata)
            mid_x = (net_left + net_right) // 2
            cv2.line(display, (mid_x, 0), (mid_x, h), self.COLOR_NET, 2)
            
            # Zone rete
            cv2.line(display, (net_left, 0), (net_left, h), self.COLOR_NET, 1)
            cv2.line(display, (net_right, 0), (net_right, h), self.COLOR_NET, 1)
        
        # Se abbiamo tutti i punti, disegna le zone
        if len(self.clicks) >= 4:
            self._draw_zones(display)
        
        cv2.imshow(self.WINDOW_NAME, display)
    
    def _draw_zones(self, display: np.ndarray):
        """Disegna le zone del campo sul display"""
        h, w = display.shape[:2]
        
        net_left_px, _ = self.clicks[0]
        net_right_px, _ = self.clicks[1]
        _, court_top_px = self.clicks[2]
        _, court_bottom_px = self.clicks[3]
        
        # Normalizza
        net_left = net_left_px / w
        net_right = net_right_px / w
        
        # Calcola bounds
        result = CalibrationResult(
            camera=self.camera,
            net_left=net_left,
            net_right=net_right,
            court_top=court_top_px / h,
            court_bottom=court_bottom_px / h,
            frame_width=w,
            frame_height=h,
        )
        bounds = result.to_bounds()
        
        # Overlay semi-trasparente
        overlay = display.copy()
        
        # Disegna ogni zona
        zones = [
            ("SERVICE A", bounds.service_a, (100, 100, 255)),
            ("BACK A", bounds.back_a, (150, 150, 200)),
            ("FRONT A", bounds.front_a, (100, 200, 150)),
            ("NET", bounds.net, (0, 255, 255)),
            ("FRONT B", bounds.front_b, (100, 200, 150)),
            ("BACK B", bounds.back_b, (150, 150, 200)),
            ("SERVICE B", bounds.service_b, (100, 100, 255)),
        ]
        
        for name, (x_start, x_end), color in zones:
            x1 = int(x_start * w)
            x2 = int(x_end * w)
            y1 = court_top_px
            y2 = court_bottom_px
            
            # Rettangolo zona
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Label zona
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Sfondo per testo
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (cx - tw//2 - 5, cy - th//2 - 5),
                          (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
            cv2.putText(overlay, name, (cx - tw//2, cy + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        
        # Linee bordo campo
        cv2.line(display, (0, court_top_px), (w, court_top_px), self.COLOR_COURT, 2)
        cv2.line(display, (0, court_bottom_px), (w, court_bottom_px), self.COLOR_COURT, 2)
        
        # Mostra messaggio conferma
        msg = "Premi INVIO per confermare, R per ricominciare"
        cv2.putText(display, msg, (w//2 - 250, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_NET, 2)
    
    def calibrate_from_frame(self, frame: np.ndarray, 
                              camera: CameraPosition = CameraPosition.SIDE) -> Optional[CalibrationResult]:
        """
        Calibra da un singolo frame.
        
        Args:
            frame: Frame numpy (BGR)
            camera: Tipo di camera (SIDE o BEHIND)
        
        Returns:
            CalibrationResult se confermato, None se annullato
        """
        self.frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]
        self.camera = camera
        self.clicks = []
        self._done = False
        self._cancelled = False
        
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, min(1280, self.frame_width), 
                         min(720, self.frame_height))
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
        
        self._update_display()
        
        while not self._done and not self._cancelled:
            key = cv2.waitKey(50) & 0xFF
            
            if key == 27:  # ESC
                self._cancelled = True
            elif key == 13 or key == 10:  # INVIO
                if len(self.clicks) >= 4:
                    self._done = True
            elif key == ord('r') or key == ord('R'):
                self.clicks = []
                self._update_display()
            elif key == ord('c') or key == ord('C'):
                # Toggle camera
                if self.camera == CameraPosition.SIDE:
                    self.camera = CameraPosition.BEHIND
                else:
                    self.camera = CameraPosition.SIDE
                self._update_display()
        
        cv2.destroyWindow(self.WINDOW_NAME)
        
        if self._cancelled or len(self.clicks) < 4:
            return None
        
        # Crea risultato
        net_left_px, _ = self.clicks[0]
        net_right_px, _ = self.clicks[1]
        _, court_top_px = self.clicks[2]
        _, court_bottom_px = self.clicks[3]
        
        return CalibrationResult(
            camera=self.camera,
            net_left=net_left_px / self.frame_width,
            net_right=net_right_px / self.frame_width,
            court_top=court_top_px / self.frame_height,
            court_bottom=court_bottom_px / self.frame_height,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )
    
    def calibrate_from_video(self, video_path: str,
                              frame_number: int = 0,
                              camera: CameraPosition = CameraPosition.SIDE) -> Optional[CalibrationResult]:
        """
        Calibra da un video (usa il primo frame o uno specifico).
        
        Args:
            video_path: Path al file video
            frame_number: Numero del frame da usare (default: 0)
            camera: Tipo di camera
        
        Returns:
            CalibrationResult se confermato, None se annullato
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Errore: impossibile aprire {video_path}")
            return None
        
        # Vai al frame richiesto
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Errore: impossibile leggere il frame")
            return None
        
        return self.calibrate_from_frame(frame, camera)
    
    def calibrate_from_image(self, image_path: str,
                              camera: CameraPosition = CameraPosition.SIDE) -> Optional[CalibrationResult]:
        """
        Calibra da un'immagine.
        
        Args:
            image_path: Path all'immagine
            camera: Tipo di camera
        
        Returns:
            CalibrationResult se confermato, None se annullato
        """
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Errore: impossibile aprire {image_path}")
            return None
        
        return self.calibrate_from_frame(frame, camera)

# =============================================================================
# QUICK CALIBRATION (senza GUI)
# =============================================================================

class QuickCalibrator:
    """
    Calibrazione rapida senza GUI.
    Utile per script automatizzati o quando OpenCV GUI non è disponibile.
    """
    
    @staticmethod
    def from_net_position(net_center_x: float, 
                          net_width: float = 0.12,
                          court_top: float = 0.28,
                          court_bottom: float = 0.85) -> CalibrationResult:
        """
        Calibra specificando solo la posizione della rete.
        
        Args:
            net_center_x: Posizione X normalizzata del centro della rete (0-1)
            net_width: Larghezza normalizzata della rete (default 0.12 = 12%)
            court_top: Y normalizzato bordo superiore campo
            court_bottom: Y normalizzato bordo inferiore campo
        
        Esempio:
            # Rete al centro
            result = QuickCalibrator.from_net_position(0.5)
            
            # Rete spostata a destra
            result = QuickCalibrator.from_net_position(0.55)
        """
        net_left = net_center_x - net_width / 2
        net_right = net_center_x + net_width / 2
        
        return CalibrationResult(
            camera=CameraPosition.SIDE,
            net_left=max(0.0, net_left),
            net_right=min(1.0, net_right),
            court_top=court_top,
            court_bottom=court_bottom,
            frame_width=1920,
            frame_height=1080,
        )
    
    @staticmethod
    def from_net_pixels(net_left_px: int, net_right_px: int,
                        court_top_px: int, court_bottom_px: int,
                        frame_width: int = 1920,
                        frame_height: int = 1080) -> CalibrationResult:
        """
        Calibra specificando i pixel.
        
        Args:
            net_left_px: Pixel X bordo sinistro rete
            net_right_px: Pixel X bordo destro rete
            court_top_px: Pixel Y bordo superiore campo
            court_bottom_px: Pixel Y bordo inferiore campo
            frame_width: Larghezza frame
            frame_height: Altezza frame
        """
        return CalibrationResult(
            camera=CameraPosition.SIDE,
            net_left=net_left_px / frame_width,
            net_right=net_right_px / frame_width,
            court_top=court_top_px / frame_height,
            court_bottom=court_bottom_px / frame_height,
            frame_width=frame_width,
            frame_height=frame_height,
        )

# =============================================================================
# APPLY CALIBRATION
# =============================================================================

def apply_calibration(calibration: CalibrationResult, 
                       field: FieldCalibrator = None) -> FieldCalibrator:
    """
    Applica una calibrazione a un FieldCalibrator.
    
    Args:
        calibration: Risultato della calibrazione
        field: FieldCalibrator esistente (opzionale, ne crea uno nuovo se None)
    
    Returns:
        FieldCalibrator configurato
    """
    if field is None:
        field = FieldCalibrator(camera=calibration.camera)
    
    field.set_frame_size(calibration.frame_width, calibration.frame_height)
    field.calibrate_from_points(
        net_left=calibration.net_left,
        net_right=calibration.net_right,
        court_top=calibration.court_top,
        court_bottom=calibration.court_bottom,
    )
    
    return field

# =============================================================================
# MAIN (CLI)
# =============================================================================

def main():
    """Entry point per uso da command line"""
    import sys
    
    print("=" * 60)
    print("VolleyAgents - Calibrazione Interattiva Campo")
    print("=" * 60)
    
    # Parse argomenti
    if len(sys.argv) < 2:
        print("""
Uso:
    python -m volley_agents.calibration.field_calibrator_gui <video.mp4>
    python -m volley_agents.calibration.field_calibrator_gui <immagine.jpg>
    python -m volley_agents.calibration.field_calibrator_gui <video.mp4> --frame 100
    python -m volley_agents.calibration.field_calibrator_gui <video.mp4> --behind
Opzioni:
    --frame N    Usa il frame N del video (default: 0)
    --behind     Usa modalità camera BEHIND invece di SIDE
    --save FILE  Salva configurazione su FILE.json
Istruzioni:
    1. Clicca sul bordo SINISTRO della rete
    2. Clicca sul bordo DESTRO della rete
    3. Clicca sul bordo SUPERIORE del campo
    4. Clicca sul bordo INFERIORE del campo
    5. Premi INVIO per confermare
        """)
        sys.exit(1)
    
    # Parse opzioni
    input_path = sys.argv[1]
    frame_number = 0
    camera = CameraPosition.SIDE
    save_path = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--frame" and i + 1 < len(sys.argv):
            frame_number = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--behind":
            camera = CameraPosition.BEHIND
            i += 1
        elif sys.argv[i] == "--save" and i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Verifica file
    if not Path(input_path).exists():
        print(f"Errore: file non trovato: {input_path}")
        sys.exit(1)
    
    # Calibra
    calibrator = InteractiveCalibrator()
    
    ext = Path(input_path).suffix.lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        result = calibrator.calibrate_from_image(input_path, camera)
    else:
        result = calibrator.calibrate_from_video(input_path, frame_number, camera)
    
    if result is None:
        print("\nCalibrazione annullata.")
        sys.exit(0)
    
    # Mostra risultato
    print("\n" + "=" * 60)
    print("CALIBRAZIONE COMPLETATA")
    print("=" * 60)
    print(f"Camera:       {result.camera.value}")
    print(f"Rete:         {result.net_left:.3f} - {result.net_right:.3f}")
    print(f"Campo Y:      {result.court_top:.3f} - {result.court_bottom:.3f}")
    print(f"Frame:        {result.frame_width}x{result.frame_height}")
    
    # Mostra bounds
    bounds = result.to_bounds()
    print("\nZone calcolate:")
    print(f"  SERVICE_A:  {bounds.service_a[0]:.3f} - {bounds.service_a[1]:.3f}")
    print(f"  BACK_A:     {bounds.back_a[0]:.3f} - {bounds.back_a[1]:.3f}")
    print(f"  FRONT_A:    {bounds.front_a[0]:.3f} - {bounds.front_a[1]:.3f}")
    print(f"  NET:        {bounds.net[0]:.3f} - {bounds.net[1]:.3f}")
    print(f"  FRONT_B:    {bounds.front_b[0]:.3f} - {bounds.front_b[1]:.3f}")
    print(f"  BACK_B:     {bounds.back_b[0]:.3f} - {bounds.back_b[1]:.3f}")
    print(f"  SERVICE_B:  {bounds.service_b[0]:.3f} - {bounds.service_b[1]:.3f}")
    
    # Salva se richiesto
    if save_path:
        if not save_path.endswith('.json'):
            save_path += '.json'
        result.save(save_path)
        print(f"\nConfigurazione salvata: {save_path}")
    else:
        # Chiedi se salvare
        print("\nVuoi salvare la configurazione? (s/n)")
        try:
            answer = input().strip().lower()
            if answer in ['s', 'si', 'y', 'yes']:
                default_name = Path(input_path).stem + "_calibration.json"
                print(f"Nome file [{default_name}]: ", end="")
                name = input().strip()
                if not name:
                    name = default_name
                if not name.endswith('.json'):
                    name += '.json'
                result.save(name)
                print(f"Salvato: {name}")
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Per usare questa calibrazione:")
    print("""
    from volley_agents.calibration import CalibrationResult, apply_calibration
    
    # Carica calibrazione
    calib = CalibrationResult.load("calibration.json")
    
    # Applica a FieldCalibrator
    field = apply_calibration(calib)
    
    # Oppure usa direttamente
    field.calibrate_from_points(
        net_left={:.3f},
        net_right={:.3f},
        court_top={:.3f},
        court_bottom={:.3f}
    )
    """.format(result.net_left, result.net_right, result.court_top, result.court_bottom))

if __name__ == "__main__":
    main()

