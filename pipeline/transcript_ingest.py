"""Transcript ingestion: load from a local JSON file or fetch from YouTube via yt-dlp.

Both paths return the same dict shape so downstream code is path-agnostic:

    {
        "video_id": str | int,
        "url": str,
        "title": str,
        "duration_seconds": float,
        "transcript_source": str,
        "granularity": "line",
        "items": [{"text": str, "start": float, "end": float}, ...],
    }
"""

import json
import re
import tempfile
from pathlib import Path


_CUE_RE = re.compile(
    r"(\d+:\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+)[^\n]*\n((?:.+\n?)+?)(?:\n|\Z)",
    re.MULTILINE,
)
_INLINE_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def load_from_file(path):
    """Load a canonical transcript dict from a JSON file on disk."""
    with open(path) as f:
        return json.load(f)


def _timestamp_to_seconds(ts):
    """Convert an 'HH:MM:SS.mmm' VTT timestamp to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_vtt(vtt_text):
    """Parse VTT caption text into a list of {text, start, end} items.

    Strips inline VTT tags (e.g. <00:00:01.000>) and collapses internal whitespace.
    """
    items = []
    for match in _CUE_RE.finditer(vtt_text):
        start = _timestamp_to_seconds(match.group(1))
        end = _timestamp_to_seconds(match.group(2))
        text = _INLINE_TAG_RE.sub("", match.group(3))
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if text:
            items.append({"text": text, "start": start, "end": end})
    return items


def fetch_from_youtube(url):
    """Fetch a YouTube video's auto-caption transcript via yt-dlp.

    Returns the canonical transcript dict shape with
    ``transcript_source = "yt_dlp_auto_caption"``.
    """
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": False,
            "subtitleslangs": ["en", "en-US", "en-GB", "en-orig"],
            "subtitlesformat": "vtt",
            "outtmpl": str(Path(tmpdir) / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)

        video_id = info.get("id")
        vtt_files = sorted(Path(tmpdir).glob(f"{video_id}*.vtt"))
        if not vtt_files:
            raise RuntimeError(f"No auto-captions found for {url}")
        vtt_text = vtt_files[0].read_text()

    items = _parse_vtt(vtt_text)
    return {
        "video_id": info.get("id"),
        "url": url,
        "title": info.get("title", ""),
        "duration_seconds": float(info.get("duration") or 0.0),
        "transcript_source": "yt_dlp_auto_caption",
        "granularity": "line",
        "items": items,
    }
