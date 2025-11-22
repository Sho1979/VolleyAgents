from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from volley_agents.core.event import Event, EventType

if TYPE_CHECKING:
    from volley_agents.core.timeline import Timeline

AudioArray = Union[np.ndarray, Sequence[float]]


@dataclass
class WhistleDetectorConfig:
    """
    Parametri principali per la rilevazione dei fischi.
    """

    frame_duration: float = 0.046  # ~2048 campioni @44.1 kHz
    hop_duration: float = 0.010
    band_min_hz: int = 3000
    band_max_hz: int = 6000
    threshold_sigma: float = 3.0
    min_event_duration: float = 0.12
    min_silence_duration: float = 0.05
    max_confidence_multiplier: float = 2.5


class AudioAgent:
    """
    Analizza un file WAV e traduce i picchi nella banda dei fischi
    in eventi `WHISTLE_START/WHISTLE_END`.
    """

    def __init__(self, config: Optional[WhistleDetectorConfig] = None):
        self.config = config or WhistleDetectorConfig()

    def run(
        self,
        wav_path: Union[str, Path],
        timeline: Optional["Timeline"] = None,
    ) -> List[Event]:
        """
        Analizza un file WAV su disco.
        """

        samples, sample_rate = self._load_wav(Path(wav_path))
        events = self.analyze(samples, sample_rate)
        if timeline is not None:
            timeline.extend(events)
        return events

    def load_wav(self, wav_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Carica e normalizza un file WAV, restituendo (samples, sample_rate).
        Metodo pubblico wrapper per _load_wav.
        """
        return self._load_wav(Path(wav_path))

    def load_and_analyze(self, wav_path: Union[str, Path]) -> List[Event]:
        """
        Carica un file WAV e lo analizza, restituendo gli eventi rilevati.
        Metodo di convenienza che combina load_wav() e analyze().
        """
        samples, sample_rate = self._load_wav(Path(wav_path))
        return self.analyze(samples, sample_rate)

    def analyze(self, samples: AudioArray, sample_rate: int) -> List[Event]:
        """
        Analizza un array numpy/sequence già in memoria.
        """

        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        frame_size = max(1, int(sample_rate * self.config.frame_duration))
        hop_size = max(1, int(sample_rate * self.config.hop_duration))

        if frame_size <= hop_size:
            frame_size = hop_size * 2

        band_energies, frame_times = self._band_energy(
            audio,
            sample_rate,
            frame_size,
            hop_size,
        )
        return self._to_events(band_energies, frame_times)

    @staticmethod
    def _normalize(raw: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(raw))
        if max_val == 0:
            return raw
        return raw / max_val

    def _load_wav(self, path: Path) -> Tuple[np.ndarray, int]:
        if not path.exists():
            raise FileNotFoundError(path)

        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)

        dtype_map = {
            1: np.int8,
            2: np.int16,
            4: np.int32,
        }
        if sample_width not in dtype_map:
            raise ValueError(f"Larghezza campione non supportata: {sample_width} byte")

        audio = np.frombuffer(raw, dtype=dtype_map[sample_width]).astype(np.float32)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        audio = self._normalize(audio)
        return audio, sample_rate

    def _band_energy(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_size: int,
        hop_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        window = np.hanning(frame_size)
        freqs = np.fft.rfftfreq(frame_size, d=1 / sample_rate)
        band_mask = (freqs >= self.config.band_min_hz) & (freqs <= self.config.band_max_hz)

        energies: List[float] = []
        times: List[float] = []
        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start : start + frame_size] * window
            spectrum = np.fft.rfft(frame)
            power = np.abs(spectrum) ** 2
            band_energy = float(np.sum(power[band_mask]))
            center = start + frame_size / 2
            times.append(center / sample_rate)
            energies.append(band_energy)
        return np.asarray(energies, dtype=np.float32), np.asarray(times, dtype=np.float32)

    def _to_events(
        self,
        energies: np.ndarray,
        times: np.ndarray,
    ) -> List[Event]:
        if len(energies) == 0:
            return []

        cfg = self.config
        noise_floor = float(np.percentile(energies, 70))
        threshold = (
            noise_floor * cfg.threshold_sigma
            if noise_floor > 0
            else max(np.mean(energies) * 1.5, 1e-9)
        )
        threshold = max(threshold, 1e-9)
        events: List[Event] = []

        active = False
        start_time = 0.0
        last_above = 0.0
        peak = 0.0
        silence = 0.0
        hop_duration = times[1] - times[0] if len(times) > 1 else cfg.hop_duration

        for energy, time_sec in zip(energies, times):
            is_above = energy >= threshold
            if is_above:
                last_above = time_sec
                peak = max(peak, energy)
                silence = 0.0
                if not active:
                    active = True
                    start_time = time_sec
            else:
                silence += hop_duration

            if active and (silence >= cfg.min_silence_duration or time_sec == times[-1]):
                duration = last_above - start_time
                if duration >= cfg.min_event_duration:
                    confidence = min(1.0, (peak / max(threshold, 1e-9)) / cfg.max_confidence_multiplier)
                    meta = {
                        "energy_peak": peak,
                        "duration": duration,
                        "threshold": threshold,
                    }
                    events.append(
                        Event(
                            time=start_time,
                            type=EventType.WHISTLE_START,
                            confidence=confidence,
                            extra=meta,
                        )
                    )
                    events.append(
                        Event(
                            time=last_above,
                            type=EventType.WHISTLE_END,
                            confidence=confidence,
                            extra=meta,
                        )
                    )
                active = False
                peak = 0.0
                silence = 0.0

        return events


__all__ = [
    "AudioAgent",
    "WhistleDetectorConfig",
]

