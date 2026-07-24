---
"chord-extractor-api": minor
---

Add the `POST /download` endpoint. It runs only the ingestion step (the same yt-dlp pipeline the analysis endpoints use, including the Deno-backed YouTube JS-challenge solving) and streams back the raw audio bytes instead of an analysis result. This lets callers reuse the service as a reliable YouTube→audio fetcher for external backends. Response is the audio file with a format-appropriate `Content-Type` plus an `X-Audio-Ext` header; same error codes as the other endpoints (`413`/`415`/`422`/`502`).

Fix YouTube "Sign in to confirm you're not a bot" failures from datacenter IPs: default the yt-dlp `player_client` to `android,ios,tv` (the mobile/tv innertube clients aren't gated like the default `web` client), overridable via `YTDLP_PLAYER_CLIENT`. Also support an authenticated cookies.txt via `YTDLP_COOKIES_FILE`. This affects all download paths (`/download`, `/bpm`, `/extract`, `/meter`, `/sections`).
