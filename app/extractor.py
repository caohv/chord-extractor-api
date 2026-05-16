import json
import subprocess

import librosa
import numpy as np
from chord_extractor.extractors import Chordino

SR = 22050
BPM_SAMPLE_SECONDS = 60.0


def _probe_duration(audio_path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            audio_path,
        ],
        stderr=subprocess.PIPE,
    )
    return float(json.loads(out)["format"]["duration"])


def _decode_pcm(audio_path: str, duration: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", audio_path]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.float32)


def _beat_track(y: np.ndarray) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=SR)
    return round(float(np.atleast_1d(tempo)[0]), 2)


def _perceptual_bpm(bpm: float, low: float = 60.0, high: float = 140.0) -> float:
    """Collapse tempo-octave errors into a perceptual range.

    Beat trackers often return 2x (or rarely 1/2x) of the human-felt tempo —
    e.g. a 80 BPM ballad gets reported as 160. Halve/double until the result
    falls in [low, high] when possible. Caveat: genuinely fast songs (drum &
    bass, hardcore, ~150-180 BPM) will be over-halved by this rule; callers
    that care should inspect `bpm_raw` and override.
    """
    if bpm <= 0:
        return bpm
    while bpm > high and bpm / 2 >= low:
        bpm /= 2
    while bpm < low and bpm * 2 <= high:
        bpm *= 2
    return round(bpm, 2)


def extract_bpm(audio_path: str) -> dict:
    duration = _probe_duration(audio_path)
    sample = BPM_SAMPLE_SECONDS if duration > BPM_SAMPLE_SECONDS else None
    y = _decode_pcm(audio_path, sample)
    raw = _beat_track(y)
    return {"duration": duration, "bpm": _perceptual_bpm(raw), "bpm_raw": raw}


def _patch_madmom_compat() -> None:
    # madmom 0.16.1 still references stdlib/numpy names that were removed in
    # Python 3.10 and numpy >=1.20/1.24. Patch them before importing madmom.
    import collections
    import collections.abc as cabc

    for name in (
        "MutableSequence",
        "MutableMapping",
        "Mapping",
        "Iterable",
        "Sequence",
        "Set",
        "Callable",
    ):
        if not hasattr(collections, name) and hasattr(cabc, name):
            setattr(collections, name, getattr(cabc, name))

    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]

    # numpy >=1.24 raises on inhomogeneous lists in np.asarray; madmom relies
    # on the old object-array fallback. Wrap np.asarray once.
    if not getattr(np, "_madmom_asarray_patched", False):
        _orig = np.asarray

        def _patched(a, dtype=None, order=None):  # type: ignore[no-untyped-def]
            try:
                return _orig(a, dtype=dtype, order=order)
            except ValueError:
                return _orig(a, dtype=object)

        np.asarray = _patched  # type: ignore[assignment]
        np._madmom_asarray_patched = True  # type: ignore[attr-defined]


def extract_meter(audio_path: str) -> dict:
    # madmom imports are slow; import lazily so /health and other endpoints
    # don't pay the cost.
    _patch_madmom_compat()

    from madmom.features.downbeats import (
        DBNDownBeatTrackingProcessor,
        RNNDownBeatProcessor,
    )

    duration = _probe_duration(audio_path)

    act = RNNDownBeatProcessor()(audio_path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4, 6], fps=100)
    beats = proc(act)  # shape (N, 2): [timestamp, position_in_bar]

    if len(beats) == 0:
        return {
            "duration": duration,
            "beats_per_bar": 0,
            "time_signature": "unknown",
            "confidence": 0.0,
            "downbeats": [],
        }

    positions = [int(round(p)) for _, p in beats]
    # madmom's DBN outputs positions in 1..K where K is the beats_per_bar it
    # selected from the candidates passed to DBNDownBeatTrackingProcessor.
    beats_per_bar = max(positions)
    # 6 beats/bar in DBN ≈ 6/8 (compound duple); 3 → 3/4; 4 → 4/4.
    ts_map = {3: "3/4", 4: "4/4", 6: "6/8"}
    time_signature = ts_map.get(beats_per_bar, f"{beats_per_bar}/4")

    downbeats = [float(t) for t, p in beats if int(round(p)) == 1]
    # Confidence: ratio of detected downbeats vs the count expected for a
    # perfectly regular cycle of length `beats_per_bar`. Capped at 1.0; an
    # extra leading/trailing downbeat can otherwise nudge it slightly over.
    expected_bars = len(beats) / beats_per_bar
    confidence = (
        round(min(len(downbeats) / expected_bars, 1.0), 3)
        if expected_bars
        else 0.0
    )

    return {
        "duration": duration,
        "beats_per_bar": beats_per_bar,
        "time_signature": time_signature,
        "confidence": confidence,
        "downbeats": downbeats,
    }


def extract_chords(audio_path: str) -> dict:
    chordino = Chordino(roll_on=1)
    raw = chordino.extract(audio_path)

    duration = _probe_duration(audio_path)
    y = _decode_pcm(audio_path)
    raw_bpm = _beat_track(y)

    return {
        "duration": duration,
        "bpm": _perceptual_bpm(raw_bpm),
        "bpm_raw": raw_bpm,
        "chords": [
            {"chord": c.chord, "timestamp": float(c.timestamp)}
            for c in raw
        ],
    }
