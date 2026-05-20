"""Exception types raised by the Stamped SDK."""

from __future__ import annotations


class StampedError(Exception):
    """Base class for every error raised by this SDK."""


class APIError(StampedError):
    """The API returned a non-success HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class JobFailedError(StampedError):
    """The detection job finished in the 'failed' state."""

    def __init__(self, job_id: str, error: str | None) -> None:
        self.job_id = job_id
        self.error = error
        super().__init__(f"job {job_id} failed: {error}")


class JobTimeoutError(StampedError):
    """A job did not reach a terminal state within the allotted time."""
