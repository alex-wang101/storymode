"""FastAPI app: ``POST /detect/cloud`` and ``POST /detect/local``.

Run with:    uvicorn api.app:app --host 127.0.0.1

* Localhost-only by default. The ``reference_image_path`` field accepts a
  server-side path; if the port is ever exposed, that's a path-traversal
  risk. The allow-list root resolution (default: repo root) is the
  mitigation -- paths must resolve under it.
* Single in-flight job (``api/jobs.py`` semaphore). The second concurrent
  request gets 429.
* Static bearer token via ``VIDEOFIND_TOKEN`` env var. If unset, auth is
  skipped (local-dev convenience).
* The api_key field on the cloud body is ``SecretStr`` so its repr renders
  as ``'**********'``; nothing in this module unwraps it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr

from . import jobs
from .providers.errors import ProviderError
from .reference_gen import UnsupportedKeyPrefix, VlmJsonError
from .video_download import LiveStreamRejected, VideoUnavailable


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = Path(os.environ.get("VIDEOFIND_REFERENCE_ROOT", REPO_ROOT)).resolve()
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

logger = logging.getLogger("videofind.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Request and response models
class _DetectBase(BaseModel):
    video_url: str = Field(min_length=1)
    reference_image_path: str = Field(min_length=1)
    product_name_hint: str | None = None
    score_threshold: float = 0.6
    max_gap_seconds: float = 2.0
    min_frames: int = 2


class DetectLocalRequest(_DetectBase):
    pass


class DetectCloudRequest(_DetectBase):
    api_key: SecretStr  # SecretStr.__repr__ -> '**********'; never logged or echoed.


class DetectResponse(BaseModel):
    job_id: str
    video: dict
    reference: dict
    params: dict
    sampling_mode: str
    intervals: list[dict]

# Auth
_bearer = HTTPBearer(auto_error=False)


def require_bearer(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = os.environ.get("VIDEOFIND_TOKEN")
    if not expected:
        return  # auth disabled
    if creds is None or creds.credentials != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")

# Helpers
def _resolve_reference_image(raw: str) -> Path:
    """Resolve ``raw`` against the allow-list root. 404 on anything outside it.

    Order is intentional: we resolve first (which handles ``..`` traversal)
    before checking ``is_file``, so symlinks pointing outside the root also
    fail the prefix check.
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


async def _dispatch(
    req: _DetectBase,
    api_key: SecretStr | None,
) -> DetectResponse:
    image_path = _resolve_reference_image(req.reference_image_path)
    try:
        result = await jobs.run_job(
            video_url=req.video_url,
            reference_image_path=image_path,
            api_key=api_key,
            product_name_hint=req.product_name_hint,
            score_threshold=req.score_threshold,
            max_gap_seconds=req.max_gap_seconds,
            min_frames=req.min_frames,
        )
    except jobs.JobBusy:
        raise HTTPException(status_code=429, detail="another job is in flight")
    except UnsupportedKeyPrefix as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LiveStreamRejected as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VideoUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VlmJsonError as e:
        raise HTTPException(status_code=422, detail=f"VLM output invalid: {e}")
    except ProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except TimeoutError:
        raise HTTPException(status_code=504, detail="job timed out")
    return DetectResponse(**result)


# App
app = FastAPI(
    title="videofind",
    description=(
        "Find time intervals where a reference object appears in a video. "
        "Two endpoints: /detect/local (Qwen2-VL-2B GPTQ-Int8 locally) and "
        "/detect/cloud (Anthropic / OpenAI / Google, routed by api_key prefix). "
        "Bind to 127.0.0.1 unless you trust the network: reference_image_path "
        "accepts a server-side filesystem path."
    ),
)


@app.middleware("http")
async def log_request_id(request: Request, call_next):
    logger.info("%s %s", request.method, request.url.path)
    return await call_next(request)


@app.post("/detect/local", response_model=DetectResponse, dependencies=[Depends(require_bearer)])
async def detect_local(req: DetectLocalRequest) -> DetectResponse:
    return await _dispatch(req, api_key=None)


@app.post("/detect/cloud", response_model=DetectResponse, dependencies=[Depends(require_bearer)])
async def detect_cloud(req: DetectCloudRequest) -> DetectResponse:
    # req.api_key is a SecretStr; it is forwarded as-is to jobs.run_job ->
    # reference_gen -> the provider adapter, and unwrapped exactly once at
    # the SDK construction site in that adapter.
    return await _dispatch(req, api_key=req.api_key)
