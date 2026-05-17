"""Single yt-dlp call that fetches video + auto-captions + metadata.

The existing ``pipeline/transcript_ingest.fetch_from_youtube`` only fetches
captions. The API needs the video too. Doing both in one
``YoutubeDL.extract_info`` avoids paying the extractor cost twice and
double-throttling against YouTube.

Lives in ``api/`` so the CLI path stays unchanged.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from pipeline.transcript_ingest import _parse_vtt


@dataclass
class DownloadResult:
    video_path: Path
    transcript: dict | None        # canonical transcript dict, or None if no captions
    metadata: dict                 # {"video_id", "url", "title", "duration_seconds"}


class LiveStreamRejected(Exception):
    pass


class VideoUnavailable(Exception):
    pass


def _build_transcript(info: dict, url: str, vtt_text: str | None) -> dict | None:
    if not vtt_text:
        return None
    items = _parse_vtt(vtt_text)
    if not items:
        return None
    return {
        "video_id": info.get("id"),
        "url": url,
        "title": info.get("title", ""),
        "duration_seconds": float(info.get("duration") or 0.0),
        "transcript_source": "yt_dlp_auto_caption",
        "granularity": "line",
        "items": items,
    }


def download(url: str, dest_video_path: Path, socket_timeout: float = 60.0) -> DownloadResult:
    """Download MP4 + auto-captions + metadata into a single network round-trip.

    The captions land in a tempdir and are parsed into the canonical
    transcript dict (or returned as ``None`` if absent). The video lands at
    ``dest_video_path``.
    """
    import yt_dlp

    dest_video_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(dest_video_path.with_suffix(".%(ext)s")),
            "writeautomaticsub": True,
            "writesubtitles": False,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"],
            "subtitlesformat": "vtt",
            "paths": {"subtitle": tmpdir},
            "socket_timeout": socket_timeout,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as e:
            raise VideoUnavailable(f"yt-dlp failed: {e}") from None

        if info.get("is_live"):
            raise LiveStreamRejected("livestreams are not supported")

        video_id = info.get("id") or "video"
        # yt-dlp wrote the video to dest_video_path.with_suffix("." + ext).
        actual_ext = info.get("ext", "mp4")
        produced = dest_video_path.with_suffix(f".{actual_ext}")
        if not produced.is_file():
            # Fall back to whatever's actually on disk under the same stem.
            candidates = list(dest_video_path.parent.glob(f"{dest_video_path.stem}.*"))
            video_files = [c for c in candidates if c.suffix.lower() not in {".vtt", ".tmp", ".part"}]
            if not video_files:
                raise VideoUnavailable("yt-dlp completed but no video file on disk")
            produced = video_files[0]

        vtt_files = sorted(Path(tmpdir).glob(f"{video_id}*.vtt"))
        vtt_text = vtt_files[0].read_text() if vtt_files else None
        transcript = _build_transcript(info, url, vtt_text)

    metadata = {
        "video_id": info.get("id"),
        "url": url,
        "title": info.get("title", ""),
        "duration_seconds": float(info.get("duration") or 0.0),
    }
    return DownloadResult(video_path=produced, transcript=transcript, metadata=metadata)
