---
"chord-extractor-api": minor
---

Add the `/sections` endpoint (allin1 music-structure analysis, with optional faster-whisper canonical-lyric force-alignment via the `lyrics` request-body field) and a CUDA `Dockerfile.gpu` variant.

Fix YouTube ingestion by installing Deno in the image so yt-dlp can solve YouTube's player JS challenges (nsig/sig descrambling). Without a JS runtime, `/bpm` (and `/extract`, `/sections`) intermittently 502 with "This video is not available" on videos whose only available formats require descrambling.
