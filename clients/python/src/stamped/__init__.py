"""Stamped — Python client for the Stamped brand-detection API.

Submit a YouTube URL plus a reference image URL, follow the job's progress,
and get back the timestamp intervals where the product/brand appears.

    from stamped import StampedClient

    with StampedClient("http://localhost:8000") as client:
        result = client.detect(
            youtube_url="https://www.youtube.com/watch?v=...",
            reference_image_url="https://.../product.png",
            on_progress=lambda e: print(f"{e.progress}% {e.stage}"),
        )
        for ts in result.timestamps:
            print(f"{ts.start:.1f}s - {ts.end:.1f}s  ({ts.confidence:.2f})")
"""

from __future__ import annotations

from .client import Job, StampedClient
from .errors import APIError, JobFailedError, JobTimeoutError, StampedError
from .models import ProgressEvent, Result, Timestamp

__version__ = "0.1.0"

__all__ = [
    "StampedClient",
    "Job",
    "ProgressEvent",
    "Result",
    "Timestamp",
    "StampedError",
    "APIError",
    "JobFailedError",
    "JobTimeoutError",
]
