"""
VolleyAgents - Esempio Integrazione Completa

=============================================

Questo file mostra come integrare:

1. Calibrazione interattiva (GUI)

2. Field Calibrator

3. Ace Detector



WORKFLOW COMPLETO:

──────────────────

1. L'utente apre il video

2. Appare la GUI per calibrare il campo (4 click)

3. La calibrazione viene salvata

4. L'analisi usa le zone calibrate



Autore: VolleyAgents Project

"""



import cv2
from typing import Optional, List
from pathlib import Path



# Import moduli

from volley_agents.calibration.field_calibrator_generic import (

    FieldCalibrator,

    CameraPosition,

    AceDetector,

    Team,

    FieldZone,

    fmt_time,

)



from volley_agents.calibration.field_calibrator_gui import (

    InteractiveCalibrator,

    QuickCalibrator,

    CalibrationResult,

    apply_calibration,

)





# =============================================================================

# CLASSE PRINCIPALE

# =============================================================================



class VolleyFieldAnalyzer:

    """

    Analizzatore campo pallavolo con calibrazione automatica/manuale.

    

    Utilizzo:

    

        analyzer = VolleyFieldAnalyzer()

        

        # Calibra (apre GUI)

        analyzer.calibrate_from_video("partita.mp4")

        

        # Oppure carica calibrazione esistente

        analyzer.load_calibration("palestra_X.json")

        

        # Analizza

        analyzer.analyze_video("partita.mp4")

    """

    

    def __init__(self, log_callback=None):

        self._log = log_callback or print

        self.field: Optional[FieldCalibrator] = None

        self.ace_detector: Optional[AceDetector] = None

        self.calibration: Optional[CalibrationResult] = None

    

    # -------------------------------------------------------------------------

    # CALIBRAZIONE

    # -------------------------------------------------------------------------

    

    def calibrate_interactive(self, video_or_image: str, 

                               frame_number: int = 0) -> bool:

        """

        Calibra interattivamente usando la GUI.

        

        Args:

            video_or_image: Path al video o immagine

            frame_number: Frame da usare (per video)

        

        Returns:

            True se calibrato con successo

        """

        self._log("[Analyzer] Avvio calibrazione interattiva...")

        

        calibrator = InteractiveCalibrator()

        

        ext = Path(video_or_image).suffix.lower()

        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:

            result = calibrator.calibrate_from_image(video_or_image)

        else:

            result = calibrator.calibrate_from_video(video_or_image, frame_number)

        

        if result is None:

            self._log("[Analyzer] Calibrazione annullata")

            return False

        

        self.calibration = result

        self._apply_calibration()

        

        self._log(f"[Analyzer] ✓ Calibrato: rete={result.net_left:.2f}-{result.net_right:.2f}")

        return True

    

    def calibrate_quick(self, net_center: float = 0.5, 

                         net_width: float = 0.12) -> None:

        """

        Calibrazione rapida specificando solo la posizione della rete.

        

        Args:

            net_center: Posizione X normalizzata del centro rete (0-1)

            net_width: Larghezza normalizzata della rete

        """

        self.calibration = QuickCalibrator.from_net_position(

            net_center_x=net_center,

            net_width=net_width

        )

        self._apply_calibration()

        self._log(f"[Analyzer] ✓ Quick calibration: center={net_center}")

    

    def calibrate_from_pixels(self, net_left_px: int, net_right_px: int,

                               court_top_px: int, court_bottom_px: int,

                               frame_width: int = 1920,

                               frame_height: int = 1080) -> None:

        """Calibra specificando i pixel"""

        self.calibration = QuickCalibrator.from_net_pixels(

            net_left_px, net_right_px,

            court_top_px, court_bottom_px,

            frame_width, frame_height

        )

        self._apply_calibration()

        self._log(f"[Analyzer] ✓ Calibrato da pixel")

    

    def load_calibration(self, path: str) -> bool:

        """Carica calibrazione da file"""

        try:

            self.calibration = CalibrationResult.load(path)

            self._apply_calibration()

            self._log(f"[Analyzer] ✓ Calibrazione caricata: {path}")

            return True

        except Exception as e:

            self._log(f"[Analyzer] ✗ Errore caricamento: {e}")

            return False

    

    def save_calibration(self, path: str) -> bool:

        """Salva calibrazione su file"""

        if self.calibration is None:

            self._log("[Analyzer] Nessuna calibrazione da salvare")

            return False

        try:

            self.calibration.save(path)

            self._log(f"[Analyzer] ✓ Calibrazione salvata: {path}")

            return True

        except Exception as e:

            self._log(f"[Analyzer] ✗ Errore salvataggio: {e}")

            return False

    

    def _apply_calibration(self) -> None:

        """Applica la calibrazione corrente"""

        if self.calibration is None:

            return

        

        self.field = apply_calibration(self.calibration)

        self.ace_detector = AceDetector(self.field, log_callback=self._log)

    

    # -------------------------------------------------------------------------

    # ANALISI

    # -------------------------------------------------------------------------

    

    def get_zone(self, x: int, y: int) -> FieldZone:

        """Ottieni la zona per una posizione pixel"""

        if self.field is None:

            raise RuntimeError("Campo non calibrato! Chiama calibrate_* prima.")

        return self.field.get_zone(x, y)

    

    def get_team(self, x: int, y: int) -> Optional[Team]:

        """Ottieni la squadra per una posizione pixel"""

        zone = self.get_zone(x, y)

        return self.field.get_team(zone)

    

    def detect_serve(self, x: int, y: int, timestamp: float, 

                     magnitude: float) -> Optional[dict]:

        """

        Rileva se un movimento è un servizio.

        

        Returns:

            dict con info serve se rilevato, None altrimenti

        """

        if self.ace_detector is None:

            raise RuntimeError("Campo non calibrato!")

        

        serve = self.ace_detector.detect_serve(x, y, timestamp, magnitude)

        if serve is None:

            return None

        

        return {

            'timestamp': serve.timestamp,

            'team': serve.team.value,

            'confidence': serve.confidence,

            'x': x,

            'y': y,

        }

    

    def check_ace(self, serve: dict, whistle_time: float,

                  motion_events: List[dict]) -> Optional[dict]:

        """

        Verifica se un serve è un ACE.

        

        Args:

            serve: Dict dal detect_serve

            whistle_time: Timestamp del fischio

            motion_events: Lista eventi motion (dict con x, y, timestamp, magnitude)

        

        Returns:

            dict con info ace se confermato, None altrimenti

        """

        if self.ace_detector is None:

            raise RuntimeError("Campo non calibrato!")

        

        # Ricostruisci ServeEvent

        from volley_agents.calibration.field_calibrator_generic import ServeEvent

        serve_event = ServeEvent(

            timestamp=serve['timestamp'],

            team=Team(serve['team']),

            confidence=serve['confidence'],

            x_norm=serve['x'] / self.field.frame_width,

            y_norm=serve['y'] / self.field.frame_height,

        )

        

        # Calcola motion nel campo ricevente

        recv_motion = self.ace_detector.calculate_receiving_motion(

            serve_event, motion_events

        )

        

        # Verifica ace

        ace = self.ace_detector.check_ace(serve_event, whistle_time, recv_motion)

        

        if ace is None:

            return None

        

        return {

            'serve_time': ace.serve.timestamp,

            'whistle_time': ace.whistle_time,

            'duration': ace.duration,

            'team': ace.serve.team.value,

        }

    

    # -------------------------------------------------------------------------

    # DEBUG

    # -------------------------------------------------------------------------

    

    def draw_zones_on_frame(self, frame) -> None:

        """

        Disegna le zone sul frame (per debug).

        

        Args:

            frame: Frame numpy (modificato in-place)

        """

        if self.field is None or self.calibration is None:

            return

        

        h, w = frame.shape[:2]

        bounds = self.calibration.to_bounds()

        

        court_top = int(bounds.court_min * h)

        court_bottom = int(bounds.court_max * h)

        

        # Zone con colori

        zones = [

            (bounds.service_a, (100, 100, 255), "SRV A"),

            (bounds.back_a, (150, 150, 200), "BACK A"),

            (bounds.front_a, (100, 200, 150), "FRONT A"),

            (bounds.net, (0, 255, 255), "NET"),

            (bounds.front_b, (100, 200, 150), "FRONT B"),

            (bounds.back_b, (150, 150, 200), "BACK B"),

            (bounds.service_b, (100, 100, 255), "SRV B"),

        ]

        

        overlay = frame.copy()

        

        for (x_start, x_end), color, label in zones:

            x1 = int(x_start * w)

            x2 = int(x_end * w)

            

            cv2.rectangle(overlay, (x1, court_top), (x2, court_bottom), color, -1)

            

            cx = (x1 + x2) // 2

            cy = (court_top + court_bottom) // 2

            cv2.putText(overlay, label, (cx - 30, cy),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    

    def print_calibration(self) -> None:

        """Stampa info calibrazione"""

        if self.calibration is None:

            print("Nessuna calibrazione attiva")

            return

        

        c = self.calibration

        print(f"Camera:  {c.camera.value}")

        print(f"Rete:    {c.net_left:.3f} - {c.net_right:.3f}")

        print(f"Campo Y: {c.court_top:.3f} - {c.court_bottom:.3f}")

        print(f"Frame:   {c.frame_width}x{c.frame_height}")





# =============================================================================

# ESEMPIO UTILIZZO

# =============================================================================



def esempio_workflow_completo():

    """

    Esempio di workflow completo:

    1. Calibra il campo

    2. Analizza gli eventi

    """

    

    print("=" * 60)

    print("ESEMPIO WORKFLOW COMPLETO")

    print("=" * 60)

    

    # 1. Crea analyzer

    analyzer = VolleyFieldAnalyzer()

    

    # 2. Calibra (scegli un metodo)

    

    # Metodo A: GUI interattiva

    # analyzer.calibrate_interactive("partita.mp4")

    

    # Metodo B: Quick calibration

    analyzer.calibrate_quick(net_center=0.50, net_width=0.12)

    

    # Metodo C: Da pixel

    # analyzer.calibrate_from_pixels(

    #     net_left_px=850, net_right_px=1070,

    #     court_top_px=300, court_bottom_px=900

    # )

    

    # Metodo D: Carica da file

    # analyzer.load_calibration("palestra_montichiari.json")

    

    # 3. Mostra calibrazione

    analyzer.print_calibration()

    

    # 4. Test zone detection

    print("\n--- Test Zone Detection ---")

    test_points = [

        (100, 500),   # SERVICE_A

        (500, 500),   # BACK_A o FRONT_A

        (960, 500),   # NET

        (1400, 500),  # FRONT_B o BACK_B

        (1800, 500),  # SERVICE_B

    ]

    

    for x, y in test_points:

        zone = analyzer.get_zone(x, y)

        team = analyzer.get_team(x, y)

        team_str = team.value if team else "N/A"

        print(f"  ({x:4d}, {y}) → {zone.value:12s} (Team {team_str})")

    

    # 5. Simula rilevamento serve

    print("\n--- Simula Serve Detection ---")

    serve = analyzer.detect_serve(x=100, y=500, timestamp=120.5, magnitude=3.5)

    if serve:

        print(f"  Serve rilevato: Team {serve['team']} @ {serve['timestamp']:.1f}s")

    

    # 6. Salva calibrazione per riutilizzo

    # analyzer.save_calibration("mia_palestra.json")

    

    print("\n" + "=" * 60)





def esempio_integrazione_head_coach():

    """

    Esempio di come integrare nel HeadCoach esistente.

    """

    

    code = '''

# ============================================================================

# INTEGRAZIONE NEL HEAD_COACH

# ============================================================================



# 1. IMPORT

from volley_agents.calibration import FieldCalibrator, AceDetector

from volley_agents.calibration.field_calibrator_gui import InteractiveCalibrator, CalibrationResult



class HeadCoach:

    def __init__(self, ...):

        # ... codice esistente ...

        

        self.field: FieldCalibrator = None

        self.ace_detector: AceDetector = None

        self._calibration_path = "field_calibration.json"

    

    def setup_video(self, video_path: str):

        """Setup iniziale del video"""

        

        # ... codice esistente per aprire video ...

        

        # Prova a caricare calibrazione esistente

        try:

            calib = CalibrationResult.load(self._calibration_path)

            self.field = FieldCalibrator()

            self.field.set_frame_size(self.frame_width, self.frame_height)

            self.field.calibrate_from_points(

                calib.net_left, calib.net_right,

                calib.court_top, calib.court_bottom

            )

            self._log("Calibrazione caricata da file")

        except FileNotFoundError:

            # Prima volta: chiedi calibrazione interattiva

            self._calibrate_interactive(video_path)

    

    def _calibrate_interactive(self, video_path: str):

        """Calibra interattivamente"""

        calibrator = InteractiveCalibrator()

        result = calibrator.calibrate_from_video(video_path)

        

        if result:

            result.save(self._calibration_path)

            

            self.field = FieldCalibrator()

            self.field.set_frame_size(result.frame_width, result.frame_height)

            self.field.calibrate_from_points(

                result.net_left, result.net_right,

                result.court_top, result.court_bottom

            )

            

            self.ace_detector = AceDetector(self.field, log_callback=self._log)

            self._log("Campo calibrato e salvato!")

        else:

            self._log("Calibrazione saltata, uso preset default")

            self.field = FieldCalibrator()  # Usa preset SIDE

    

    def process_motion_event(self, x: int, y: int, timestamp: float, magnitude: float):

        """Processa evento motion"""

        

        # Ottieni zona

        zone = self.field.get_zone(x, y)

        

        # Rileva serve

        if self.ace_detector:

            serve = self.ace_detector.detect_serve(x, y, timestamp, magnitude)

            if serve:

                self._pending_serves.append(serve)

'''

    

    print("=" * 60)

    print("CODICE PER INTEGRAZIONE HEAD_COACH")

    print("=" * 60)

    print(code)





# =============================================================================

# MAIN

# =============================================================================



if __name__ == "__main__":

    esempio_workflow_completo()

    print("\n")

    esempio_integrazione_head_coach()

