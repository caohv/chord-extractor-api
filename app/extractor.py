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


def extract_bpm(audio_path: str) -> dict:
    duration = _probe_duration(audio_path)
    sample = BPM_SAMPLE_SECONDS if duration > BPM_SAMPLE_SECONDS else None
    y = _decode_pcm(audio_path, sample)
    return {"duration": duration, "bpm": _beat_track(y)}


def extract_chords(audio_path: str) -> dict:
    chordino = Chordino(roll_on=1)
    raw = chordino.extract(audio_path)

    duration = _probe_duration(audio_path)
    y = _decode_pcm(audio_path)
    bpm = _beat_track(y)

    return {
        "duration": duration,
        "bpm": bpm,
        "chords": [
            {"chord": c.chord, "timestamp": float(c.timestamp)}
            for c in raw
        ],
    }
