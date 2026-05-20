"""Typed data structures returned by the SDK."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgressEvent:
    """One job-progress update — from a status poll or the SSE stream.

    ``kind`` is the SSE event name: "progress", "completed", or "failed".
    ``status`` is the job's lifecycle state: queued | running | completed | failed.
    """

    kind: str
    job_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    message: str | None = None
    error: str | None = None


@dataclass
class Timestamp:
    """A time interval where the product/brand appears in the video."""

    start: float
    end: float
    confidence: float


@dataclass
class Result:
    """The final detection result for a completed job."""

    job_id: str
    status: str
    brand_description: str
    timestamps: list[Timestamp]
