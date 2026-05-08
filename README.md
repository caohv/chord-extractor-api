# chord-extractor-api

HTTP API for extracting chords from audio files. Clients send either a presigned audio URL (S3, etc.) or a YouTube watch URL; the API fetches the audio, runs chord-extractor (Chordino Vamp plugin), and returns `{ duration, chords: [{chord, timestamp}] }`.

## Stack

- Python 3.11 (pinned by `chord-extractor 0.1.3`, cannot move to 3.12+)
- FastAPI + uvicorn
- [`chord-extractor`](https://github.com/ohollo/chord-extractor) (wraps Chordino + NNLS Chroma Vamp plugins)
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
  "chords": [
    { "chord": "N", "timestamp": 0.0 },
    { "chord": "C", "timestamp": 0.74 },
    { "chord": "G", "timestamp": 4.21 }
  ]
}
```

Supported direct-URL audio formats: `mp3`, `wav`, `ogg`, `flac`, `m4a`, `webm`. Hard limit 100 MB per file (applies to both direct URLs and YouTube downloads).

Error codes:
- `413` — file exceeds 100 MB
- `415` — unsupported format (direct URL only)
- `422` — invalid URL
- `502` — download failed (network error, YouTube unavailable, geo-block, age-gate, etc.)
- `500` — extraction failed

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

- Extraction is CPU-bound, ~30–60 s for a 4-minute song. To handle concurrent requests, scale horizontally or move to a job queue (Celery/Hatchet).
- The API does not perform AWS authentication; the URL must be presigned or publicly fetchable over HTTP.
- Chord notation follows Chordino: `N` = no chord / silence; chords look like `C`, `Am`, `G7`, `Dm7`, `F#`, `Bb`, etc.
