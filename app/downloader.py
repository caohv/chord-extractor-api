import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError as YTDownloadError
from yt_dlp.utils import ExtractorError, GeoRestrictedError, UnavailableVideoError

MAX_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_SUFFIXES = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class DownloadError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class UnsupportedFormatError(Exception):
    pass


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in YOUTUBE_HOSTS


def _suffix_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if not suffix:
        return ".mp3"
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFormatError(f"Unsupported audio format: {suffix}")
    return suffix


def _download_youtube_sync(url: str) -> str:
    result: dict[str, str] = {}

    def _hook(d: dict) -> None:
        if d.get("status") == "finished":
            info = d.get("info_dict") or {}
            path = info.get("filepath") or d.get("filename")
            if path:
                result["filepath"] = path

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".%(ext)s")
    base = tmp.name
    tmp.close()
    # Remove the placeholder; yt-dlp will create the real file with the resolved ext.
    try:
        os.unlink(base)
    except FileNotFoundError:
        pass

    opts: dict[str, Any] = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": base,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "progress_hooks": [_hook],
    }

    try:
        with YoutubeDL(cast(Any, opts)) as ydl:
            ydl.download([url])
    except GeoRestrictedError as exc:
        raise DownloadError(f"YouTube geo-restricted: {exc}") from exc
    except UnavailableVideoError as exc:
        raise DownloadError(f"YouTube video unavailable: {exc}") from exc
    except ExtractorError as exc:
        raise DownloadError(f"YouTube extractor failed: {exc}") from exc
    except YTDownloadError as exc:
        raise DownloadError(f"YouTube download failed: {exc}") from exc

    path = result.get("filepath")
    if not path or not Path(path).exists():
        raise DownloadError("yt-dlp finished without producing a file")

    if Path(path).stat().st_size > MAX_BYTES:
        Path(path).unlink(missing_ok=True)
        raise FileTooLargeError(f"File exceeds {MAX_BYTES} bytes")

    return path


@asynccontextmanager
async def download_to_temp(url: str):
    if _is_youtube_url(url):
        path = await asyncio.to_thread(_download_youtube_sync, url)
        tmp_path = Path(path)
        try:
            yield str(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
            for leftover in tmp_path.parent.glob(f"{tmp_path.stem}.*"):
                leftover.unlink(missing_ok=True)
        return

    suffix = _suffix_from_url(url)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        written = 0
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise DownloadError(
                            f"Download failed: HTTP {resp.status_code}"
                        )
                    with tmp_path.open("wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            written += len(chunk)
                            if written > MAX_BYTES:
                                raise FileTooLargeError(
                                    f"File exceeds {MAX_BYTES} bytes"
                                )
                            fh.write(chunk)
            except httpx.HTTPError as exc:
                raise DownloadError(str(exc)) from exc

        yield str(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
