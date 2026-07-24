import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from .downloader import (
    DownloadError,
    FileTooLargeError,
    UnsupportedFormatError,
    download_to_temp,
)
from .extractor import (
    extract_bpm,
    extract_chords,
    extract_meter,
    extract_sections,
)
from .schemas import (
    BpmResponse,
    ExtractRequest,
    ExtractResponse,
    MeterResponse,
    SectionsResponse,
)

logger = logging.getLogger("chord-extractor-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Chord Extractor API", version="0.1.0")

# Suffix -> Content-Type for the raw audio bytes returned by /download.
_AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "webm": "audio/webm",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    url = str(req.url)
    logger.info("extract requested url=%s", url)
    async with download_to_temp(url) as path:
        result = await run_in_threadpool(extract_chords, path)
    return ExtractResponse(**result)


@app.post("/bpm", response_model=BpmResponse)
async def bpm(req: ExtractRequest) -> BpmResponse:
    url = str(req.url)
    logger.info("bpm requested url=%s", url)
    async with download_to_temp(url) as path:
        result = await run_in_threadpool(extract_bpm, path)
    return BpmResponse(**result)


@app.post("/meter", response_model=MeterResponse)
async def meter(req: ExtractRequest) -> MeterResponse:
    url = str(req.url)
    logger.info("meter requested url=%s", url)
    async with download_to_temp(url) as path:
        result = await run_in_threadpool(extract_meter, path)
    return MeterResponse(**result)


@app.post("/sections", response_model=SectionsResponse)
async def sections(req: ExtractRequest) -> SectionsResponse:
    url = str(req.url)
    logger.info(
        "sections requested url=%s lyrics_lines=%s",
        url,
        len(req.lyrics) if req.lyrics else 0,
    )
    async with download_to_temp(url) as path:
        result = await run_in_threadpool(extract_sections, path, req.lyrics)
    return SectionsResponse(**result)


@app.post("/download")
async def download(req: ExtractRequest) -> Response:
    """Download the source audio and return the raw bytes (no analysis).

    Reuses the same yt-dlp ingestion as the analysis endpoints (Deno-backed
    solving of YouTube's player-JS challenges), so callers can use this service
    as a reliable YouTube->audio fetcher — e.g. to feed an external
    alignment/transcription backend — instead of a third-party downloader.
    The `Content-Type` and `X-Audio-Ext` header reflect the resolved format
    (yt-dlp prefers m4a; direct URLs keep their original suffix).
    """
    url = str(req.url)
    logger.info("download requested url=%s", url)
    async with download_to_temp(url) as path:
        p = Path(path)
        ext = p.suffix.lstrip(".").lower() or "mp3"
        data = await run_in_threadpool(p.read_bytes)
    return Response(
        content=data,
        media_type=_AUDIO_MEDIA_TYPES.get(ext, "application/octet-stream"),
        headers={
            "Content-Disposition": f'attachment; filename="audio.{ext}"',
            "X-Audio-Ext": ext,
        },
    )


@app.exception_handler(UnsupportedFormatError)
async def _unsupported(_: Request, exc: UnsupportedFormatError) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(FileTooLargeError)
async def _too_large(_: Request, exc: FileTooLargeError) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(DownloadError)
async def _download_failed(_: Request, exc: DownloadError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Extraction failed"})
