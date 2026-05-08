import json
import subprocess

from chord_extractor.extractors import Chordino


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


def extract_chords(audio_path: str) -> dict:
    chordino = Chordino(roll_on=1)
    raw = chordino.extract(audio_path)

    duration = _probe_duration(audio_path)

    return {
        "duration": duration,
        "chords": [
            {"chord": c.chord, "timestamp": float(c.timestamp)}
            for c in raw
        ],
    }
