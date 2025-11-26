"""RallyDetectorV2: Supervisore multi-segnale per rilevamento rally.

Combina:
- GameStateAgent (VideoMAE): PLAY/NO-PLAY classification (segnale principale)
- AudioAgent: WHISTLE_END per conferma fine rally
- BallAgentV2: Ball tracking per validazione
- MotionAgent: Hit events (opzionale)

NON richiede ROI manuali come ServeAgent!
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class PlayState(Enum):
    """Stato del gioco rilevato da VideoMAE."""

    NO_PLAY = "no-play"
    PLAY = "play"
    SERVICE = "service"
    UNKNOWN = "unknown"


@dataclass
class Rally:
    """Rappresenta un rally rilevato."""

    start: float
    end: float
    side: str = "unknown"
    confidence: float = 1.0
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return (
            f"Rally({self.start:.1f}-{self.end:.1f}, "
            f"dur={self.duration:.1f}s, side={self.side}, conf={self.confidence:.2f})"
        )


@dataclass
class RallyDetectorV2Config:
    """Configurazione RallyDetectorV2."""

    # Soglie temporali
    min_rally_duration: float = 3.0
    max_rally_duration: float = 60.0
    merge_gap: float = 4.0  # Merge rally se gap < 4s

    # Pesi segnali (0-1)
    weight_game_state: float = 1.0  # Più affidabile
    weight_whistle: float = 0.8
    weight_ball: float = 0.5
    weight_motion: float = 0.3

    # Soglie conferma
    whistle_window: float = 5.0  # Cerca fischio entro 5s dalla fine PLAY
    ball_activity_threshold: float = 0.3  # Min 30% ball detection durante rally

    # Logging
    enable_logging: bool = True
    log_callback: Optional[Callable[[str], None]] = None


class RallyDetectorV2:
    """
    Supervisore multi-segnale per rilevamento rally.

    Pipeline:
    1. Estrai transizioni PLAY/NO-PLAY da GameStateAgent
    2. Genera rally candidati dalle transizioni
    3. Refine con fischi (WHISTLE_END)
    4. Valida con ball tracking
    5. Merge rally vicini
    6. Filtra per durata
    """

    def __init__(self, config: Optional[RallyDetectorV2Config] = None):
        self.config = config or RallyDetectorV2Config()

    def log(self, msg: str) -> None:
        """Log con callback opzionale."""
        if self.config.enable_logging:
            if self.config.log_callback:
                self.config.log_callback(msg)
            else:
                print(msg)

    def run(
        self,
        game_state_events: List[Any],
        whistle_events: Optional[List[Any]] = None,
        ball_events: Optional[List[Any]] = None,
        motion_events: Optional[List[Any]] = None,  # noqa: ARG002  (per futuro)
        serve_events: Optional[List[Any]] = None,
    ) -> List[Rally]:
        """
        Esegui detection rally dai segnali multi-agente.

        Args:
            game_state_events: Eventi GAME_STATE da GameStateAgent
            whistle_events: Eventi WHISTLE_END da AudioAgent
            ball_events: Eventi BALL_DETECTED da BallAgentV2
            motion_events: Eventi HIT da MotionAgent
            serve_events: Eventi SERVE_START da ServeAgent (opzionale)

        Returns:
            Lista di Rally rilevati
        """
        self.log("[RallyDetectorV2] 🚀 Inizio detection...")

        # Step 1: Estrai stati PLAY dal GameStateAgent
        play_segments = self._extract_play_segments(game_state_events)
        self.log(f"[RallyDetectorV2] 📊 Segmenti PLAY trovati: {len(play_segments)}")

        if not play_segments:
            self.log("[RallyDetectorV2] ⚠️ Nessun segmento PLAY trovato!")
            return []

        # Step 2: Genera rally candidati
        candidates = self._generate_candidates(play_segments)
        self.log(f"[RallyDetectorV2] 🎯 Rally candidati: {len(candidates)}")

        # Step 3: Refine con fischi
        if whistle_events:
            candidates = self._refine_with_whistles(candidates, whistle_events)
            self.log(f"[RallyDetectorV2] 🎵 Dopo refine fischi: {len(candidates)}")

        # Step 4: Valida con ball tracking
        if ball_events:
            candidates = self._validate_with_ball(candidates, ball_events)
            self.log(f"[RallyDetectorV2] 🏐 Dopo validazione ball: {len(candidates)}")

        # Step 5: Determina side se abbiamo serve events
        if serve_events:
            candidates = self._assign_side_from_serve(candidates, serve_events)

        # Step 6: Merge rally vicini
        candidates = self._merge_close_rallies(candidates)
        self.log(f"[RallyDetectorV2] 🔀 Dopo merge: {len(candidates)}")

        # Step 7: Filtra per durata
        final = self._filter_by_duration(candidates)
        self.log(f"[RallyDetectorV2] ✅ Rally finali: {len(final)}")

        # Log dettagli
        for i, r in enumerate(final, 1):
            self.log(f"  Rally {i}: {r.start:.1f}-{r.end:.1f} ({r.duration:.1f}s) side={r.side}")

        return final

    def _extract_play_segments(self, events: List[Any]) -> List[Tuple[float, float]]:
        """
        Estrai segmenti temporali dove state=PLAY.

        Raggruppa eventi PLAY consecutivi in segmenti continui.
        Considera che GameStateAgent ha stride di 2s, quindi gap di 4-6s sono normali.
        """
        if not events:
            return []

        # Filtra solo eventi con state=play
        play_events: List[float] = []
        for e in events:
            state: Optional[str] = None
            if hasattr(e, "extra") and isinstance(e.extra, dict):
                state = e.extra.get("state")
            elif isinstance(e, dict):
                state = e.get("extra", {}).get("state") or e.get("state")

            if state == PlayState.PLAY.value or state == "play":
                time = e.time if hasattr(e, "time") else float(e.get("time", 0))
                play_events.append(time)

        if not play_events:
            return []

        # Ordina per tempo
        play_events.sort()

        # Raggruppa eventi vicini
        # Con stride=2s, gap fino a 4s potrebbe essere stesso rally
        max_gap = 4.0

        segments: List[Tuple[float, float]] = []
        seg_start = play_events[0]
        seg_end = play_events[0]

        for t in play_events[1:]:
            if t - seg_end <= max_gap:  # Stesso segmento
                seg_end = t
            else:  # Nuovo segmento
                # Padding: -1s start (pre-roll), +2s end (window VideoMAE)
                segments.append((seg_start - 1.0, seg_end + 2.0))
                seg_start = t
                seg_end = t

        # Ultimo segmento
        segments.append((seg_start - 1.0, seg_end + 2.0))

        return segments

    def _generate_candidates(self, play_segments: List[Tuple[float, float]]) -> List[Rally]:
        """Genera rally candidati dai segmenti PLAY."""
        candidates: List[Rally] = []

        for start, end in play_segments:
            rally = Rally(
                start=start,
                end=end,
                confidence=1.0,
                signals={"game_state": True},
            )
            candidates.append(rally)

        return candidates

    def _refine_with_whistles(self, candidates: List[Rally], whistle_events: List[Any]) -> List[Rally]:
        """
        Affina end time dei rally usando WHISTLE_END.

        Se troviamo un fischio dopo il segmento PLAY ma entro window,
        ESTENDIAMO il rally fino al fischio (il fischio segna la fine reale).
        """
        # Estrai tempi fischi
        whistle_times: List[float] = []
        for e in whistle_events:
            etype = e.type.value if hasattr(e, "type") and hasattr(e.type, "value") else str(
                getattr(e, "type", "")
            )
            if "WHISTLE" in etype.upper():
                time = e.time if hasattr(e, "time") else float(e.get("time", 0))
                whistle_times.append(time)

        whistle_times.sort()

        refined: List[Rally] = []
        for rally in candidates:
            best_whistle: Optional[float] = None

            # Cerca il PRIMO fischio DOPO l'inizio del rally
            # che sia entro una finestra ragionevole dalla fine stimata
            for wt in whistle_times:
                # Fischio deve essere:
                # 1. Dopo start del rally
                # 2. Vicino a end (o anche un po' dopo - il PLAY potrebbe finire prima del fischio)
                if wt > rally.start + 2.0:  # Almeno 2s dopo start
                    # Se fischio è entro 15s dalla fine stimata, potrebbe essere il fischio giusto
                    if wt <= rally.end + 12.0:
                        best_whistle = wt
                        break  # Prendi il primo fischio valido

            if best_whistle is not None:
                # ESTENDI il rally fino al fischio (non ridurlo!)
                rally.end = max(rally.end, best_whistle)
                rally.signals["whistle"] = best_whistle
                rally.confidence = min(1.0, rally.confidence + 0.15)

            refined.append(rally)

        return refined

    def _validate_with_ball(self, candidates: List[Rally], ball_events: List[Any]) -> List[Rally]:
        """
        Valida rally verificando presenza palla.

        Se durante il rally c'è alta % di ball detection, aumenta confidence.
        Se troppo bassa, penalizza.
        """
        # Estrai tempi ball detected
        ball_times: List[float] = []
        for e in ball_events:
            etype = e.type.value if hasattr(e, "type") and hasattr(e.type, "value") else str(
                getattr(e, "type", "")
            )
            etype_up = etype.upper()
            if "BALL" in etype_up and "DETECTED" in etype_up:
                time = e.time if hasattr(e, "time") else float(e.get("time", 0))
                ball_times.append(time)

        ball_times.sort()

        validated: List[Rally] = []
        for rally in candidates:
            # Conta ball detections durante rally
            ball_in_rally = sum(1 for bt in ball_times if rally.start <= bt <= rally.end)

            # Stima frame attesi (assumendo 5fps)
            expected_frames = rally.duration * 5.0
            if expected_frames > 0:
                ball_ratio = ball_in_rally / expected_frames
                rally.signals["ball_ratio"] = ball_ratio

                if ball_ratio >= self.config.ball_activity_threshold:
                    rally.confidence = min(1.0, rally.confidence + 0.05)
                elif ball_ratio < 0.1:
                    rally.confidence *= 0.8

            validated.append(rally)

        return validated

    def _assign_side_from_serve(self, candidates: List[Rally], serve_events: List[Any]) -> List[Rally]:
        """
        Assegna side (left/right) basandosi sul serve più vicino all'inizio.
        """
        # Estrai serve con side
        serves: List[Tuple[float, str]] = []
        for e in serve_events:
            etype = e.type.value if hasattr(e, "type") and hasattr(e.type, "value") else str(
                getattr(e, "type", "")
            )
            if "SERVE" in etype.upper():
                time = e.time if hasattr(e, "time") else float(e.get("time", 0))
                side: Optional[str] = None
                if hasattr(e, "extra") and isinstance(e.extra, dict):
                    side = e.extra.get("side")
                elif isinstance(e, dict):
                    extra = e.get("extra", {}) or {}
                    side = extra.get("side") or e.get("side")
                if side:
                    serves.append((time, side))

        serves.sort(key=lambda x: x[0])

        for rally in candidates:
            best_serve: Optional[Tuple[float, str]] = None
            min_dist = float("inf")

            # Cerca serve più vicino a start (entro 5s prima o 2s dopo)
            for st, side in serves:
                if rally.start - 5.0 <= st <= rally.start + 2.0:
                    dist = abs(st - rally.start)
                    if dist < min_dist:
                        min_dist = dist
                        best_serve = (st, side)

            if best_serve is not None:
                rally.side = best_serve[1]
                rally.signals["serve"] = best_serve[0]

        return candidates

    def _merge_close_rallies(self, candidates: List[Rally]) -> List[Rally]:
        """
        Unisci rally troppo vicini (gap < merge_gap).
        """
        if not candidates:
            return []

        # Ordina per start time
        candidates.sort(key=lambda r: r.start)

        merged: List[Rally] = [candidates[0]]

        for rally in candidates[1:]:
            last = merged[-1]

            # Se gap troppo piccolo, merge
            if rally.start - last.end < self.config.merge_gap:
                # Estendi last
                last.end = max(last.end, rally.end)
                last.confidence = max(last.confidence, rally.confidence)
                # Merge signals
                for k, v in rally.signals.items():
                    if k not in last.signals:
                        last.signals[k] = v
                # Keep side se unknown
                if last.side == "unknown" and rally.side != "unknown":
                    last.side = rally.side
            else:
                merged.append(rally)

        return merged

    def _filter_by_duration(self, candidates: List[Rally]) -> List[Rally]:
        """Filtra rally per durata min/max e qualità segnali."""
        filtered: List[Rally] = []
        for r in candidates:
            # Check durata
            if not (self.config.min_rally_duration <= r.duration <= self.config.max_rally_duration):
                continue

            # Se non ha fischio E durata < 8s, probabilmente è un falso positivo
            if "whistle" not in r.signals and r.duration < 8.0:
                continue

            filtered.append(r)

        return filtered

    def compare_with_ground_truth(
        self,
        detected: List[Rally],
        ground_truth: List[Dict[str, Any]],
        tolerance: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Confronta rally rilevati con ground truth.

        Args:
            detected: Rally rilevati
            ground_truth: Lista di dict con start, end, side
            tolerance: Tolleranza in secondi per match

        Returns:
            Dict con precision, recall, f1, matches, missed, extra
        """
        matches: List[Dict[str, Any]] = []
        matched_gt = set()
        matched_det = set()

        for i, det in enumerate(detected):
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue

                gt_start = float(gt["start"])
                gt_end = float(gt["end"])

                # Check overlap o vicinanza
                start_ok = abs(det.start - gt_start) <= tolerance
                end_ok = abs(det.end - gt_end) <= tolerance

                # Oppure overlap > 50%
                overlap_start = max(det.start, gt_start)
                overlap_end = min(det.end, gt_end)
                overlap = max(0.0, overlap_end - overlap_start)
                gt_dur = gt_end - gt_start
                overlap_ratio = overlap / gt_dur if gt_dur > 0 else 0.0

                if (start_ok and end_ok) or overlap_ratio > 0.5:
                    matches.append(
                        {
                            "det_idx": i,
                            "gt_idx": j,
                            "det": det,
                            "gt": gt,
                            "start_diff": det.start - gt_start,
                            "end_diff": det.end - gt_end,
                            "overlap_ratio": overlap_ratio,
                            "side_match": det.side == gt.get("side", "unknown"),
                        }
                    )
                    matched_gt.add(j)
                    matched_det.add(i)
                    break

        tp = len(matches)
        fp = len(detected) - tp  # Detected ma non in GT
        fn = len(ground_truth) - tp  # GT non detected

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        missed_gt = [j for j in range(len(ground_truth)) if j not in matched_gt]
        extra_det = [i for i in range(len(detected)) if i not in matched_det]

        side_correct = sum(1 for m in matches if m["side_match"])
        side_accuracy = side_correct / len(matches) if matches else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matches": matches,
            "missed_gt_indices": missed_gt,
            "extra_det_indices": extra_det,
            "side_accuracy": side_accuracy,
            "avg_start_diff": float(
                np.mean([m["start_diff"] for m in matches]) if matches else 0.0
            ),
            "avg_end_diff": float(
                np.mean([m["end_diff"] for m in matches]) if matches else 0.0
            ),
        }


def format_comparison_report(
    result: Dict[str, Any],
    ground_truth: List[Dict[str, Any]],
    detected: List[Rally],
) -> str:
    """Formatta report di confronto."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("📊 RALLY DETECTOR V2 - CONFRONTO CON GROUND TRUTH")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"🎯 Precision: {result['precision']:.1%} "
        f"({result['tp']}/{result['tp'] + result['fp']})"
    )
    lines.append(
        f"📈 Recall:    {result['recall']:.1%} "
        f"({result['tp']}/{result['tp'] + result['fn']})"
    )
    lines.append(f"🏆 F1 Score:  {result['f1']:.3f}")
    lines.append(f"🔄 Side Acc:  {result['side_accuracy']:.1%}")
    lines.append(f"⏱️ Avg Start Δ: {result['avg_start_diff']:+.1f}s")
    lines.append(f"⏱️ Avg End Δ:   {result['avg_end_diff']:+.1f}s")
    lines.append("")

    if result["matches"]:
        lines.append("✅ MATCH DETAILS:")
        for m in result["matches"]:
            gt = m["gt"]
            det = m["det"]
            lines.append(
                f"  GT[{m['gt_idx'] + 1}]: "
                f"{gt['start']:.1f}-{gt['end']:.1f} ({gt.get('side', '?')})"
            )
            lines.append(
                f"  DET[{m['det_idx'] + 1}]: "
                f"{det.start:.1f}-{det.end:.1f} ({det.side})"
            )
            lines.append(
                f"    Δstart={m['start_diff']:+.1f}s, "
                f"Δend={m['end_diff']:+.1f}s, "
                f"overlap={m['overlap_ratio']:.0%}"
            )
            lines.append("")

    if result["missed_gt_indices"]:
        lines.append("❌ MISSED (Ground Truth non rilevati):")
        for idx in result["missed_gt_indices"]:
            gt = ground_truth[idx]
            lines.append(
                f"  GT[{idx + 1}]: {gt['start']:.1f}-{gt['end']:.1f} ({gt.get('side', '?')})"
            )
        lines.append("")

    if result["extra_det_indices"]:
        lines.append("⚠️ EXTRA (Falsi positivi):")
        for idx in result["extra_det_indices"]:
            det = detected[idx]
            lines.append(f"  DET[{idx + 1}]: {det.start:.1f}-{det.end:.1f}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


__all__ = [
    "RallyDetectorV2",
    "RallyDetectorV2Config",
    "Rally",
    "PlayState",
    "format_comparison_report",
]

