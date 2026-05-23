import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
from chord_extractor.extractors import Chordino

SR = 22050
BPM_SAMPLE_SECONDS = 60.0

# allin1 filters its 10 raw HARMONIX labels down to functional sections; the
# `start`/`end` markers wrap leading/trailing silence and aren't musically
# meaningful, so they're dropped before returning.
_DROPPED_SECTION_LABELS = {"start", "end"}

# allin1's `harmonix-all` ensembles 8 fold checkpoints; on CPU that's ~8× the
# inference time of a single fold for ~1-3 F1 points of accuracy. Default to a
# single fold for latency; opt into the ensemble by setting
# `ALLIN1_MODEL=harmonix-all` in the environment.
_DEFAULT_ALLIN1_MODEL = "harmonix-fold0"


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
    bpm = _beat_track(y)

    return {
        "duration": duration,
        "bpm": bpm,
        "chords": [
            {"chord": c.chord, "timestamp": float(c.timestamp)}
            for c in raw
        ],
    }


def _patch_natten_cpu() -> None:
    # NATTEN 0.17.5's CPU wheel still probes torch.cuda.get_device_capability
    # at import time to detect Triton support, which raises on CPU-only
    # PyTorch ("Torch not compiled with CUDA enabled"). Stub it to a
    # below-Triton-threshold value so the import sees no Triton.
    import torch

    if not torch.cuda.is_available():
        torch.cuda.get_device_capability = lambda *_a, **_k: (0, 0)


def _patch_allin1_demix() -> None:
    # allin1.demix.demix shells out to `python -m demucs.separate --name
    # htdemucs` and reads back stems from `<demix_dir>/htdemucs/<stem>/*.wav`.
    # htdemucs is the slowest demucs variant; on CPU it dominates /sections
    # wall-clock. hdemucs_mmi (the pre-transformer Hybrid Demucs) is ~2×
    # faster on CPU, ~80 MB instead of ~250 MB, ~1 SDR less separation
    # quality — still enough for downstream structural analysis. We chose
    # hdemucs_mmi over mdx_q because mdx_q is a 4-model bag that loads all
    # sub-models in parallel and peaks RAM at ~6 GB, OOM-ing on hosts with
    # <8 GB (e.g. Cloudflare Containers standard-1 = 4 GB). hdemucs_mmi is
    # a single model and stays around ~1 GB peak, matching htdemucs's
    # memory profile. We swap both the CLI flag and the expected output
    # subdir, leaving everything else (4-stem layout) intact.
    #
    # Reach for the modules via sys.modules — allin1/__init__.py runs
    # `from .analyze import analyze`, which clobbers the `allin1.analyze`
    # *attribute* (it now refers to the function, not the submodule). The
    # `from .demix import demix` inside `allin1.analyze` then binds `demix`
    # into the analyze module's namespace; rebinding that namespace entry
    # via sys.modules is what actually swaps the function the patched
    # analyze() will call.
    import subprocess
    import sys

    import allin1.analyze  # noqa: F401  — populates sys.modules
    import allin1.demix  # noqa: F401  — populates sys.modules

    demix_module = sys.modules["allin1.demix"]
    analyze_module = sys.modules["allin1.analyze"]

    if getattr(demix_module.demix, "_demix_patched", False):
        return

    def _demix(paths, demix_dir, device):
        model_name = "hdemucs_mmi"
        todos: list = []
        demix_paths: list = []
        for path in paths:
            out_dir = demix_dir / model_name / path.stem
            demix_paths.append(out_dir)
            if out_dir.is_dir() and all(
                (out_dir / f"{s}.wav").is_file()
                for s in ("bass", "drums", "other", "vocals")
            ):
                continue
            todos.append(path)
        if todos:
            subprocess.run(
                [
                    sys.executable, "-m", "demucs.separate",
                    "--out", demix_dir.as_posix(),
                    "--name", model_name,
                    "--device", str(device),
                    *[p.as_posix() for p in todos],
                ],
                check=True,
            )
        return demix_paths

    _demix._demix_patched = True  # type: ignore[attr-defined]
    demix_module.demix = _demix
    analyze_module.demix = _demix


def extract_sections(audio_path: str) -> dict:
    # allin1 imports madmom via its spectrogram module, so we hit the same
    # Python 3.10+/numpy compat issues as extract_meter. Patch before the
    # import chain runs. NATTEN's CPU import also needs a CUDA-probe stub.
    _patch_madmom_compat()
    _patch_natten_cpu()

    # allin1 (PyTorch + Demucs + neighborhood-attention model) is heavy; lazy
    # import so /health and the lighter endpoints aren't dragged into its
    # startup cost.
    import allin1
    import torch

    _patch_allin1_demix()

    duration = _probe_duration(audio_path)

    # Auto-pick GPU if available — the same image build (CPU wheels) reports
    # `is_available() == False`, so this stays a no-op for CPU deploys. On
    # the GPU image variant (Dockerfile.gpu, CUDA wheels) it flips to
    # 'cuda' transparently; the device flows through to Demucs's
    # subprocess as well via our monkey-patched _demix.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # allin1 defaults demix_dir/spec_dir to ./demix and ./spec relative to
    # cwd. In an HTTP server that turns concurrent requests into a race over
    # shared scratch dirs; isolate each call in its own tempdir and let
    # keep_byproducts=False clean up.
    model_name = os.environ.get("ALLIN1_MODEL") or _DEFAULT_ALLIN1_MODEL

    with tempfile.TemporaryDirectory(prefix="allin1-") as scratch:
        scratch_path = Path(scratch)
        result = allin1.analyze(
            audio_path,
            model=model_name,
            demix_dir=scratch_path / "demix",
            spec_dir=scratch_path / "spec",
            device=device,
            keep_byproducts=False,
            # multiprocess spawns helper procs for spectrogram extraction.
            # Inside uvicorn's threadpool that adds fork overhead with no
            # win on single-file requests, and risks issues under gunicorn
            # workers — keep it single-process.
            multiprocess=False,
        )

    segments = [
        {"start": float(s.start), "end": float(s.end), "label": s.label}
        for s in result.segments
        if s.label not in _DROPPED_SECTION_LABELS
    ]

    return {
        "duration": duration,
        "bpm": int(result.bpm),
        "beats": [float(b) for b in result.beats],
        "downbeats": [float(b) for b in result.downbeats],
        "segments": segments,
    }
