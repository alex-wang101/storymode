"""Synchronous client for the Stamped brand-detection API."""

from __future__ import annotations

import json
import time
from typing import Callable, Iterator, Optional

import httpx

from .errors import APIError, JobFailedError, JobTimeoutError
from .models import ProgressEvent, Result, Timestamp

_TERMINAL_STATES = frozenset({"completed", "failed"})


# ---- HTTP / SSE helpers -----------------------------------------------------

def _raise_for_status(response: httpx.Response) -> None:
    """Raise APIError on a non-2xx response, extracting the JSON ``detail``."""
    if response.status_code < 400:
        return
    # A streamed response hasn't had its body read yet; .read() is a no-op
    # for a regular response whose content is already loaded.
    response.read()
    try:
        detail = response.json().get("detail", response.text)
    except json.JSONDecodeError:
        detail = response.text or response.reason_phrase
    raise APIError(response.status_code, detail)


def _parse_sse(response: httpx.Response) -> Iterator[tuple[str, str]]:
    """Yield ``(event_name, data)`` pairs from a text/event-stream response.

    Lines starting with ``:`` are comments (the server's keepalive heartbeat)
    and are skipped. An event is dispatched on each blank line.
    """
    event = "message"
    data: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].lstrip())
    if data:  # trailing event with no final blank line
        yield event, "\n".join(data)


def _to_event(kind: str, data: dict) -> ProgressEvent:
    return ProgressEvent(
        kind=kind,
        job_id=data["job_id"],
        status=data["status"],
        stage=data.get("stage"),
        progress=data.get("progress", 0),
        message=data.get("message"),
        error=data.get("error"),
    )


def _to_result(data: dict) -> Result:
    return Result(
        job_id=data["job_id"],
        status=data["status"],
        brand_description=data["brand_description"],
        timestamps=[Timestamp(**t) for t in data["timestamps"]],
    )


# ---- Client -----------------------------------------------------------------

class StampedClient:
    """Synchronous client for a Stamped API server.

    Example::

        with StampedClient("http://localhost:8000") as client:
            result = client.detect(
                youtube_url="https://www.youtube.com/watch?v=...",
                reference_image_url="https://.../product.png",
                on_progress=lambda e: print(e.progress, e.stage),
            )
            for ts in result.timestamps:
                print(ts.start, ts.end, ts.confidence)
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def __enter__(self) -> "StampedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    # -- low-level endpoint wrappers -----------------------------------------

    def submit(self, youtube_url: str, reference_image_url: str) -> "Job":
        """POST a detection job. Returns immediately with a Job handle."""
        response = self._http.post(
            "/v1/brand-detections",
            json={
                "youtube_url": youtube_url,
                "reference_image_url": reference_image_url,
            },
        )
        _raise_for_status(response)
        return Job(self, response.json()["job_id"])

    def status(self, job_id: str) -> ProgressEvent:
        """GET the current job status (no result payload)."""
        response = self._http.get(f"/v1/jobs/{job_id}")
        _raise_for_status(response)
        return _to_event("progress", response.json())

    def result(self, job_id: str) -> Result:
        """GET the final result. Raises APIError(409) if the job isn't done yet."""
        response = self._http.get(f"/v1/jobs/{job_id}/result")
        _raise_for_status(response)
        return _to_result(response.json())

    def stream(self, job_id: str) -> Iterator[ProgressEvent]:
        """Yield ProgressEvents from the SSE stream until the job is terminal.

        ``timeout=None`` on the request disables the read timeout — an SSE
        connection is long-lived by design (the server keeps it open with a
        15 s keepalive heartbeat).
        """
        with self._http.stream(
            "GET", f"/v1/jobs/{job_id}/events", timeout=None
        ) as response:
            _raise_for_status(response)
            for kind, raw in _parse_sse(response):
                yield _to_event(kind, json.loads(raw))

    # -- high-level convenience ----------------------------------------------

    def detect(
        self,
        youtube_url: str,
        reference_image_url: str,
        *,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        timeout: Optional[float] = None,
    ) -> Result:
        """Submit a job, follow it to completion, and return the Result.

        ``on_progress`` is invoked for every progress update. Raises
        JobFailedError if the job fails, JobTimeoutError on timeout.
        """
        return self.submit(youtube_url, reference_image_url).wait(
            on_progress=on_progress, timeout=timeout
        )


class Job:
    """A handle to one submitted detection job."""

    def __init__(self, client: StampedClient, job_id: str) -> None:
        self._client = client
        self.job_id = job_id

    def __repr__(self) -> str:
        return f"Job(job_id={self.job_id!r})"

    def status(self) -> ProgressEvent:
        """Fetch the current status of this job."""
        return self._client.status(self.job_id)

    def stream(self) -> Iterator[ProgressEvent]:
        """Iterate progress events for this job over SSE."""
        return self._client.stream(self.job_id)

    def result(self) -> Result:
        """Fetch the final result. Raises APIError(409) if not finished."""
        return self._client.result(self.job_id)

    def wait(
        self,
        *,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        timeout: Optional[float] = None,
    ) -> Result:
        """Block until the job finishes, then return its Result.

        Follows the SSE stream for progress. Raises JobFailedError if the job
        ends in failure, JobTimeoutError if it does not finish within
        ``timeout`` seconds.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        for event in self.stream():
            if on_progress is not None:
                on_progress(event)
            if event.status == "failed":
                raise JobFailedError(self.job_id, event.error)
            if event.status == "completed":
                return self._client.result(self.job_id)
            if deadline is not None and time.monotonic() > deadline:
                raise JobTimeoutError(
                    f"job {self.job_id} did not finish within {timeout}s"
                )
        # The stream ended without a terminal event (e.g. the connection was
        # dropped server-side). Fall back to the durable result endpoint.
        return self._client.result(self.job_id)
