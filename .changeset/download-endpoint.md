---
"chord-extractor-api": minor
---

Add the `POST /download` endpoint. It runs only the ingestion step (the same yt-dlp pipeline the analysis endpoints use, including the Deno-backed YouTube JS-challenge solving) and streams back the raw audio bytes instead of an analysis result. This lets callers reuse the service as a reliable YouTube→audio fetcher for external backends. Response is the audio file with a format-appropriate `Content-Type` plus an `X-Audio-Ext` header; same error codes as the other endpoints (`413`/`415`/`422`/`502`).
