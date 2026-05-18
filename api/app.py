"""FastAPI app: POST /jobs/local, POST /jobs/cloud, GET /jobs/{id}.

reference_image_path is a server-side filesystem path — keep this bound to
localhost unless you've thought through the path-traversal implications.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr

from . import debug, jobs
from .reference_gen import UnsupportedKeyPrefix


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = Path(os.environ.get("VIDEOFIND_REFERENCE_ROOT", REPO_ROOT)).resolve()
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
WAIT_TIMEOUT_SECONDS = 30  # bounded long poll — client re-calls if still running

logger = logging.getLogger("videofind.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# Request models
class DetectLocalRequest(BaseModel):
    video_url: str = Field(min_length=1)
    reference_image_path: str = Field(min_length=1)
    product_name_hint: str | None = None
    score_threshold: float = 0.6
    max_gap_seconds: float = 2.0
    min_frames: int = 2


class DetectCloudRequest(DetectLocalRequest):
    api_key: SecretStr  # repr renders as '**********'; never logged or echoed


# Response model
class JobStatus(BaseModel):
    job_id: str
    status: str           # queued | running | done | failed
    stage: str | None = None
    error: str | None = None
    result: dict | None = None   # present when status == "done"


# Auth
_bearer = HTTPBearer(auto_error=False)


def require_bearer(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = os.environ.get("VIDEOFIND_TOKEN")
    if not expected:
        return
    if creds is None or creds.credentials != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def _resolve_reference_image(raw: str) -> Path:
    """Resolve against the allow-list root. 404 on anything outside it.

    Resolve before is_file check so symlinks outside the root also fail.
    """
    try:
        resolved = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="reference image path not readable")
    try:
        resolved.relative_to(REFERENCE_ROOT)
    except ValueError:
        raise HTTPException(status_code=404, detail="reference image path outside allowed root")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="reference image not found")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="reference image exceeds 10 MB")
    return resolved


async def _submit(req: DetectLocalRequest, api_key: SecretStr | None) -> JobStatus:
    image_path = _resolve_reference_image(req.reference_image_path)
    try:
        job_id = await jobs.submit_job(
            video_url=req.video_url,
            reference_image_path=image_path,
            api_key=api_key,
            product_name_hint=req.product_name_hint,
            score_threshold=req.score_threshold,
            max_gap_seconds=req.max_gap_seconds,
            min_frames=req.min_frames,
        )
    except jobs.JobBusy:
        raise HTTPException(status_code=429, detail="job queue is full")
    except UnsupportedKeyPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JobStatus(job_id=job_id, **jobs._jobs[job_id])


# App
@asynccontextmanager
async def lifespan(_: FastAPI):
    asyncio.create_task(jobs.job_worker())
    yield


app = FastAPI(
    title="videofind",
    description=(
        "Detect product appearances in a video. Submit a job, then poll "
        "GET /jobs/{id} until status is 'done' or 'failed'."
    ),
    lifespan=lifespan,
)


app.include_router(debug.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    return await call_next(request)


@app.post("/jobs/local", response_model=JobStatus, dependencies=[Depends(require_bearer)])
async def submit_local(req: DetectLocalRequest) -> JobStatus:
    return await _submit(req, api_key=None)


@app.post("/jobs/cloud", response_model=JobStatus, dependencies=[Depends(require_bearer)])
async def submit_cloud(req: DetectCloudRequest) -> JobStatus:
    return await _submit(req, api_key=req.api_key)


@app.get("/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_bearer)])
async def get_job(job_id: str) -> JobStatus:
    job = jobs._jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)


@app.get("/jobs/{job_id}/wait", response_model=JobStatus, dependencies=[Depends(require_bearer)])
async def wait_for_job(job_id: str) -> JobStatus:
    """Bounded long poll: hold the connection up to WAIT_TIMEOUT_SECONDS, then return
    current state (terminal or not). Client re-calls if status is still running.
    """
    job = jobs._jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] in ("done", "failed"):
        return JobStatus(job_id=job_id, **job)

    loop = asyncio.get_event_loop()
    deadline = loop.time() + WAIT_TIMEOUT_SECONDS
    cond = jobs._job_events[job_id]
    async with cond:
        while jobs._jobs[job_id]["status"] not in ("done", "failed"):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(cond.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break
    return JobStatus(job_id=job_id, **jobs._jobs[job_id])
