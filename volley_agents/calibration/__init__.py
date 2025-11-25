"""
VolleyAgents - Field Calibration Module

Modulo per la calibrazione interattiva del campo di pallavolo.

Esporta:
- FieldCalibrator: Calibratore campo generico
- InteractiveCalibrator: Calibrazione con GUI
- QuickCalibrator: Calibrazione rapida senza GUI
- CalibrationResult: Risultato della calibrazione
- AceDetector: Rilevatore ACE basato su zone
"""

from volley_agents.calibration.field_calibrator_generic import (
    FieldCalibrator,
    CameraPosition,
    FieldZone,
    Team,
    ZoneBounds,
    AceDetector,
    ServeEvent,
    AceEvent,
    fmt_time,
)

from volley_agents.calibration.field_calibrator_gui import (
    InteractiveCalibrator,
    QuickCalibrator,
    CalibrationResult,
    apply_calibration,
)

from volley_agents.calibration.field_auto import (
    FieldAutoCalibrator,
    FieldAutoConfig,
)

__all__ = [
    # Generic
    'FieldCalibrator',
    'CameraPosition',
    'FieldZone',
    'Team',
    'ZoneBounds',
    'AceDetector',
    'ServeEvent',
    'AceEvent',
    'fmt_time',
    # GUI
    'InteractiveCalibrator',
    'QuickCalibrator',
    'CalibrationResult',
    'apply_calibration',
    # Auto-calibration
    'FieldAutoCalibrator',
    'FieldAutoConfig',
]

