"""FastAPI app — async brand-detection jobs with SSE progress.

Flow:
  POST /v1/brand-detections   -> 202 + job_id, pipeline runs in the background
  GET  /v1/jobs/{id}          -> status snapshot (no result payload)
  GET  /v1/jobs/{id}/events   -> SSE progress stream, ends with completed/failed
  GET  /v1/jobs/{id}/result   -> final result (durable fallback if SSE dropped)

The HTTP request is never held open for the pipeline run. The reference image
is supplied as a URL and fetched server-side with SSRF guards.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import image_fetch, jobs


STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

logger = logging.getLogger("videofind.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---- Request / response models ---------------------------------------------

class BrandDetectionRequest(BaseModel):
    youtube_url: str = Field(min_length=1)
    object_image_url: str = Field(
        min_length=1,
        description="Photo of the physical product — drives the VLM description "
        "and OWLv2 detection prompts (what shape to look for).",
    )
    brand_image_url: str = Field(
        min_length=1,
        description="The brand logo / mark — embedded by NaFlex to verify that a "
        "detected object carries the right brand.",
    )


class AcceptedResponse(BaseModel):
    job_id: str
    status: str
    events_url: str
    result_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str            # queued | running | completed | failed
    stage: str | None = None
    progress: int = 0
    message: str | None = None
    error: str | None = None


class TimestampInterval(BaseModel):
    start: float
    end: float
    confidence: float


class ResultResponse(BaseModel):
    job_id: str
    status: str
    brand_description: str
    timestamps: list[TimestampInterval]


# ---- App --------------------------------------------------------------------

app = FastAPI(
    title="videofind",
    description="Async brand/product detection in YouTube videos, with SSE progress.",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    return await call_next(request)


async def _fetch_reference(url: str, role: str) -> Path:
    """Fetch one reference image, mapping any failure to a 400 naming its role."""
    try:
        return await asyncio.to_thread(
            image_fetch.fetch_image_to_temp, url, max_bytes=MAX_IMAGE_BYTES
        )
    except image_fetch.ImageFetchError as e:
        raise HTTPException(
            status_code=400, detail=f"{role} image fetch failed: {e}"
        )


@app.post("/v1/brand-detections", status_code=202, response_model=AcceptedResponse)
async def create_brand_detection(
    req: BrandDetectionRequest, background_tasks: BackgroundTasks
) -> AcceptedResponse:
    """Accept a job and return 202 immediately; the pipeline runs in the background."""
    job_id = jobs.new_job_id()

    # Fetch both reference images up front so a bad URL fails fast with 400
    # (rather than surfacing later as a job failure). image_fetch enforces
    # https-only, public-IP-only, image/* content-type, and a 10 MB cap.
    object_tmp = await _fetch_reference(req.object_image_url, "object")
    try:
        brand_tmp = await _fetch_reference(req.brand_image_url, "brand")
    except HTTPException:
        object_tmp.unlink(missing_ok=True)  # don't leak the first temp file
        raise

    job_dir = jobs.ARTIFACTS_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    object_path = job_dir / f"object{object_tmp.suffix}"
    brand_path = job_dir / f"brand{brand_tmp.suffix}"
    # shutil.move handles cross-device temp dirs (/tmp may be a separate mount).
    shutil.move(str(object_tmp), object_path)
    shutil.move(str(brand_tmp), brand_path)

    jobs.create_job(job_id)
    background_tasks.add_task(
        jobs.run_brand_detection_pipeline,
        job_id,
        req.youtube_url,
        object_path,
        brand_path,
    )
    return AcceptedResponse(
        job_id=job_id,
        status="queued",
        events_url=f"/v1/jobs/{job_id}/events",
        result_url=f"/v1/jobs/{job_id}/result",
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Current job status — no result payload (use /result for that)."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        stage=job["stage"],
        progress=job["progress"],
        message=job["message"],
        error=job["error"],
    )


@app.get("/v1/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of job progress.

    Emits ``progress`` events as the job advances, then a terminal
    ``completed`` (with the result inline) or ``failed`` event, then closes.
    """
    if jobs.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return StreamingResponse(
        jobs.sse_event_stream(job_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering of the stream
        },
    )


@app.get("/v1/jobs/{job_id}/result", response_model=ResultResponse)
async def get_job_result(job_id: str) -> ResultResponse:
    """Final result. Durable fallback when the SSE connection drops or the
    page is refreshed."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"job is not complete yet (status: {job['status']})",
        )
    result = job["result"]
    return ResultResponse(
        job_id=job_id,
        status="completed",
        brand_description=result["brand_description"],
        timestamps=result["timestamps"],
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the minimal frontend demo."""
    return FileResponse(STATIC_DIR / "client.html")
