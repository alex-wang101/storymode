# stamped-video

Python client for the **Stamped** brand-detection API — find the timestamp
intervals where a product or brand appears in a YouTube video.

## Install

```bash
pip install stamped-video
```

The import name is `stamped`:

```python
from stamped import StampedClient
```

## Quickstart

```python
from stamped import StampedClient

with StampedClient("http://localhost:8000") as client:
    result = client.detect(
        youtube_url="https://www.youtube.com/watch?v=4GBf9ZO2UN8",
        reference_image_url="https://example.com/product.png",
        on_progress=lambda e: print(f"{e.progress:3d}%  {e.stage}"),
    )

    print(result.brand_description)
    for ts in result.timestamps:
        print(f"  {ts.start:.1f}s - {ts.end:.1f}s  (confidence {ts.confidence:.2f})")
```

`detect()` submits the job, follows it to completion over Server-Sent Events,
and returns the final result.

## Lower-level API

For more control, drive the job lifecycle yourself:

```python
with StampedClient("http://localhost:8000") as client:
    job = client.submit(youtube_url=..., reference_image_url=...)  # returns at once

    for event in job.stream():          # iterate SSE progress events
        print(event.kind, event.status, event.stage, event.progress)

    print(job.status())                 # one-off status poll
    result = job.result()               # final result (durable fallback)
```

- `client.submit(...)` / `job` — submit a job, get a handle back immediately.
- `job.status()` — current status snapshot, no result payload.
- `job.stream()` — generator of `ProgressEvent`s over SSE.
- `job.result()` — the final `Result`; raises `APIError` (409) if not done.
- `job.wait(on_progress=, timeout=)` — block until done, return the `Result`.

## Errors

All exceptions subclass `StampedError`:

- `APIError` — non-2xx HTTP response (`.status_code`, `.detail`).
- `JobFailedError` — the job finished in the `failed` state (`.error`).
- `JobTimeoutError` — `wait()` exceeded its timeout.

## Requirements

- Python 3.9+
- [`httpx`](https://www.python-httpx.org/) (installed automatically)
