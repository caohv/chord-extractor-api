import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

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
    logger.info("sections requested url=%s", url)
    async with download_to_temp(url) as path:
        result = await run_in_threadpool(extract_sections, path)
    return SectionsResponse(**result)


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
