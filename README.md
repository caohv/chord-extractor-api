# chord-extractor-api

HTTP API for extracting chords, tempo, meter, and functional structure (intro/verse/chorus/bridge) from audio files. Clients send either a presigned audio URL (S3, etc.) or a YouTube watch URL; the API fetches the audio, runs the requested analysis, and returns JSON.

## Stack

- Python 3.11 (pinned by `chord-extractor 0.1.3`, cannot move to 3.12+)
- FastAPI + uvicorn
- [`chord-extractor`](https://github.com/ohollo/chord-extractor) (wraps Chordino + NNLS Chroma Vamp plugins) for `/extract`
- [`madmom`](https://github.com/CPJKU/madmom) (RNN/DBN downbeat tracker) for `/meter`
- [`allin1`](https://github.com/mir-aidj/all-in-one) (PyTorch + Demucs + neighborhood-attention model) for `/sections`
- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for YouTube ingestion
- pixi (deps), Docker (deploy)

## API

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /extract`
Body:
```json
{ "url": "https://bucket.s3.amazonaws.com/audio.mp3?X-Amz-..." }
```

Or a YouTube URL:
```json
{ "url": "https://www.youtube.com/watch?v=29QfzY0IrC0" }
```

YouTube hosts recognised: `youtube.com`, `www.youtube.com`, `m.youtube.com`, `music.youtube.com`, `youtu.be`. The API uses `yt-dlp` to fetch the best available audio stream (typically m4a or webm/opus).

Response 200:
```json
{
  "duration": 217.34,
  "bpm": 120.5,
  "chords": [
    { "chord": "N", "timestamp": 0.0 },
    { "chord": "C", "timestamp": 0.74 },
    { "chord": "G", "timestamp": 4.21 }
  ]
}
```

`bpm` is the raw value from `librosa.beat.beat_track` on a 22.05 kHz mono mixdown decoded via `ffmpeg` pipe. Beat trackers don't disambiguate tempo octaves — a 80 BPM ballad with busy ornamentation may be reported as 160 BPM, and genuinely fast 170 BPM tracks land at 170. Callers that need a perceptual tempo should pick between `bpm / 2`, `bpm`, and `bpm * 2` based on their own heuristic.

### `POST /bpm`
Same body as `/extract`. Returns only tempo. Decodes the first 60 s of audio (full audio download is unavoidable for YouTube; for direct URLs the full file is fetched but only 60 s is decoded).

```json
{ "duration": 217.34, "bpm": 120.5 }
```

Use this when you only need tempo — typically ~5x faster than `/extract` because Chordino is skipped.

### `POST /meter`
Same body as `/extract`. Detects time signature and downbeat positions using madmom's `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`. Decodes the full audio. Considers `beats_per_bar ∈ {3, 4, 6}` and lets the DBN pick.

```json
{
  "duration": 219.13,
  "beats_per_bar": 3,
  "time_signature": "3/4",
  "confidence": 1.0,
  "downbeats": [0.92, 2.55, 4.24, 5.92]
}
```

Mapping: `3 → 3/4` (simple triple), `4 → 4/4` (simple quadruple), `6 → 6/8` (compound duple). Note: 6/8 vs 3/4 is genuinely ambiguous from beat tracking alone; madmom's RNN trained on annotated data usually picks correctly but isn't perfect.

This endpoint runs an RNN forward pass over the full audio and is the slowest of the three (~5–10 s on native amd64, longer under emulation).

### `POST /sections`
Body: same as `/extract` (`{"url": ...}`), plus an optional `lyrics` array of canonical lines for forced alignment (see "Lyric alignment" below). Runs music structure analysis with [`allin1`](https://github.com/mir-aidj/all-in-one) (state-of-the-art ISMIR 2023 model). Returns functional segments labeled `intro`, `verse`, `chorus`, `bridge`, `inst`, `solo`, `break`, `outro` (the raw `start`/`end` silence markers are dropped). Pipeline per request: Demucs source separation (hdemucs_mmi — allin1's default htdemucs is monkey-patched out for ~2× CPU speedup at similar memory footprint) → spectrogram extraction → neighborhood-attention transformer → boundary detection + label classification.

```json
{
  "duration": 217.34,
  "bpm": 120,
  "beats": [0.33, 0.75, 1.14, ...],
  "downbeats": [0.33, 1.94, 3.53, ...],
  "segments": [
    { "start": 0.33,   "end": 13.13,  "label": "intro" },
    { "start": 13.13,  "end": 37.53,  "label": "chorus" },
    { "start": 37.53,  "end": 51.53,  "label": "verse" },
    { "start": 51.53,  "end": 64.34,  "label": "verse" },
    { "start": 64.34,  "end": 89.93,  "label": "chorus" },
    { "start": 89.93,  "end": 105.93, "label": "bridge" },
    { "start": 105.93, "end": 154.67, "label": "chorus" }
  ]
}
```

This is the heaviest endpoint by far. The default model is `harmonix-fold0` (single fold) plus hdemucs_mmi separation, which lands a 4-min track in roughly ~30–60 s on native amd64 CPU — Demucs source separation (~15–30 s) still dominates wall-clock, leaving ~10–20 s for the structural model itself. Under emulation (e.g. Docker Desktop on Apple Silicon) expect 5–10× that.

Override the model via `ALLIN1_MODEL`:
- `ALLIN1_MODEL=harmonix-fold0` (default) — single fold, fastest.
- `ALLIN1_MODEL=harmonix-all` — 8-fold ensemble, ~1–3 F1 points more accurate on Harmonix boundary/label benchmarks, but ~8× the inference cost (a 4-min track jumps to ~3–5 min total on native amd64 CPU).
- `ALLIN1_MODEL=harmonix-foldN` (N in 0..7) — pick any individual fold.

Model weights (~1.9 GB total: ~80 MB hdemucs_mmi + ~250 MB htdemucs fallback + ~80 MB for all 8 allin1 fold checkpoints + ~1.5 GB faster-whisper medium) are pre-baked into the image, so first request pays no download cost regardless of which `ALLIN1_MODEL`/`WHISPER_MODEL` you select. The Demucs separator is hard-coded to hdemucs_mmi via a runtime monkey-patch — htdemucs weights stay in the image for future opt-in but aren't selectable yet.

`bpm` here is an integer reported by allin1's beat tracker, distinct from the librosa-based float returned by `/extract` and `/bpm`.

#### Lyric alignment (`lyrics` body field)

Pass canonical lyrics in the request body to have each line aligned to a timestamp and a structural label. Whisper transcription is used **only as a timing source** — the response text is always your canonical line, not Whisper's transcript (which can mis-recognize Vietnamese words). The matching runs Whisper at word-level on the Demucs vocals stem, then Needleman-Wunsch aligns your canonical tokens to Whisper's word stream.

Request:

```json
{
  "url": "https://...",
  "lyrics": [
    "Mình ra thành phố tìm về một nơi thật an nhiên",
    "Tạm quên ngày tháng, bỏ lại đằng sau những muộn phiền",
    "Chỉ mong thật sẽ luôn bên nhau như ngày đầu"
  ]
}
```

Pass each line as a separate array entry, **with repetitions expanded** — if the chorus appears 3 times in the song, include it 3 times in the array, in order. The server does not auto-repeat sections.

Response:

```json
{
  "duration": 269.28,
  "bpm": 107,
  "beats": [...],
  "downbeats": [...],
  "segments": [
    { "start":  19.49, "end":  37.87, "label": "verse" },
    { "start": 108.99, "end": 126.77, "label": "chorus" }
  ],
  "lyrics": [
    { "start":  20.10, "end":  24.50, "text": "Mình ra thành phố tìm về một nơi thật an nhiên", "label": "verse" },
    { "start":  24.50, "end":  29.30, "text": "Tạm quên ngày tháng, bỏ lại đằng sau những muộn phiền", "label": "verse" },
    { "start": null, "end": null, "text": "Chỉ mong thật sẽ luôn bên nhau như ngày đầu", "label": "unaligned" }
  ]
}
```

When `lyrics` is omitted from the body, the `lyrics` field in the response is `null` and Whisper does not run. A line that couldn't be matched to any Whisper word (e.g. backing-vocals Whisper missed) gets `start=end=null` and `label="unaligned"`.

Whisper config via env:
- `WHISPER_MODEL` (default `medium`) — one of `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`. Smaller = faster, less reliable word boundaries.
- `WHISPER_LANGUAGE` (default `vi`) — ISO 639-1 code, or unset to let Whisper auto-detect.

Latency cost: ~30-45 s on native amd64 CPU (faster-whisper `medium` int8 with word timestamps on a 4-min vocals stem), ~5-10 s on RTX 3050 Ti fp16. The endpoint roughly doubles in wall-clock when lyrics are requested.

Supported direct-URL audio formats: `mp3`, `wav`, `ogg`, `flac`, `m4a`, `webm`. Hard limit 100 MB per file (applies to both direct URLs and YouTube downloads).

Error codes:
- `413` — file exceeds 100 MB
- `415` — unsupported format (direct URL only)
- `422` — invalid URL
- `502` — download failed (network error, YouTube unavailable, geo-block, age-gate, etc.)
- `500` — extraction failed

### `POST /download`
Body: same as `/extract` (`{"url": ...}`). Runs only the ingestion step and streams back the **raw audio bytes** — no analysis. Uses the same yt-dlp pipeline (Deno-backed YouTube JS-challenge solving) as the analysis endpoints, so it doubles as a reliable YouTube→audio fetcher for external backends.

Response is the audio file (not JSON): `Content-Type` is the resolved audio type (`audio/mp4` for yt-dlp's preferred m4a, or the original suffix for direct URLs) and `X-Audio-Ext` carries the bare extension (`m4a`, `mp3`, …). Same `413`/`415`/`422`/`502` error codes as above.

```bash
curl -X POST https://chord-extractor.tuner.vn/download \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=..."}' \
  -o audio.m4a
```

## Local dev

### Recommended: run via Docker

`chord-extractor` ships a pre-compiled Chordino binary only for **Linux 64-bit**. On macOS the plugin is missing unless installed manually, so the easiest path is the container:

```bash
docker build -t chord-extractor-api .
docker run --rm -p 8000:8000 chord-extractor-api
curl http://localhost:8000/health
```

### Native via pixi (Linux, or macOS with the Vamp plugin installed)

```bash
pixi install
pixi run dev          # uvicorn --reload, port 8000
```

On macOS, for `chord-extractor` to find Chordino, install the plugin pack manually into `~/Library/Audio/Plug-Ins/Vamp/`:

1. Download Chordino + NNLS Chroma from https://code.soundsoftware.ac.uk/projects/nnls-chroma/files
2. Copy the `.dylib` into `~/Library/Audio/Plug-Ins/Vamp/`
3. Verify: run `pixi run python -c "import vamp; print(vamp.list_plugins())"` — `nnls-chroma:chordino` must appear

If the plugin is not installed, `POST /extract` will return 500 with a Vamp error in the logs.

`POST /sections` requires `allin1` + PyTorch CPU + NATTEN, which are pip-installed on top of the pixi env inside the Docker build (not via pixi, since NATTEN's CPU wheels live outside PyPI). Native macOS dev for `/sections` is **not supported** — the endpoint will fail with `ImportError` if you run uvicorn outside Docker. Use the Docker workflow for any work touching `/sections`.

## Test

```bash
pixi run -e dev test           # pytest
pixi run -e dev lint           # ruff check
```

## Deploy

CI builds and pushes `ghcr.io/trancong12102/chord-extractor-api:latest` (workflow_dispatch on `.github/workflows/build-image.yml`). Production deploy uses `docker-compose.prod.yml` with a Cloudflare Tunnel sidecar for public ingress:

```bash
cp .env.prod.example .env.prod   # fill TUNNEL_TOKEN
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs -f api
```

The Cloudflare Tunnel must route a public hostname to `http://api:8000` (matches the `api` service name in compose). To roll a new image:

```bash
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d   # recreates with new image
```

## Notes

- Extraction is CPU-bound; `/extract` and `/sections` both run ~30–60 s for a 4-minute song on native amd64. To handle concurrent requests, scale horizontally or move to a job queue (Celery/Hatchet).
- Docker image is ~3 GB once PyTorch CPU, Demucs, and the pre-baked model weights are included. Cloudflare Containers basic (1 GB RAM) is insufficient for `/sections` — use `standard-1` or larger.
- The API does not perform AWS authentication; the URL must be presigned or publicly fetchable over HTTP.
- Chord notation follows Chordino: `N` = no chord / silence; chords look like `C`, `Am`, `G7`, `Dm7`, `F#`, `Bb`, etc.
