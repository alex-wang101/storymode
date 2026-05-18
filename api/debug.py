"""Per-stage debug endpoints for testing and inspecting pipeline stages in isolation.

Each endpoint runs its stage plus any uncached prerequisites, then returns
the stage output directly as a synchronous long-poll response.

Because previous stages are cached to disk, calling /debug/score after
/debug/detect only runs brand scoring — download, VLM, sponsors, frames,
and detect all short-circuit on their cache hits.

WARNING: these endpoints bypass the job queue and load models directly.
Don't run a debug endpoint while a full job is in progress — both will
compete for MPS memory.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, SecretStr

from . import cache, jobs
from .reference_gen import BACKEND_LOCAL, UnsupportedKeyPrefix, backend_for_key


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = Path(os.environ.get("VIDEOFIND_REFERENCE_ROOT", REPO_ROOT)).resolve()
MAX_IMAGE_BYTES = 10 * 1024 * 1024

_bearer = HTTPBearer(auto_error=False)


def _require_bearer(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = os.environ.get("VIDEOFIND_TOKEN")
    if not expected:
        return
    if creds is None or creds.credentials != expected:
        raise HTTPException(401, "invalid or missing bearer token")


logger = logging.getLogger("videofind.debug")

router = APIRouter(prefix="/debug", tags=["debug"], dependencies=[Depends(_require_bearer)])


class StageRequest(BaseModel):
    video_url: str = Field(min_length=1)
    reference_image_path: str = Field(min_length=1)
    product_name_hint: str | None = None
    api_key: SecretStr | None = None
    # threshold params — only used by /debug/intervals
    score_threshold: float = 0.6
    max_gap_seconds: float = 2.0
    min_frames: int = 2


def _resolve_image(raw: str) -> Path:
    try:
        resolved = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        raise HTTPException(404, "reference image path not readable")
    try:
        resolved.relative_to(REFERENCE_ROOT)
    except ValueError:
        raise HTTPException(404, "reference image path outside allowed root")
    if not resolved.is_file():
        raise HTTPException(404, "reference image not found")
    if resolved.stat().st_size > MAX_IMAGE_BYTES:
        raise HTTPException(413, "reference image exceeds 10 MB")
    return resolved


def _job_ctx(req: StageRequest) -> tuple[Path, str, Path, Path]:
    """Return (image_path, job_id, job_dir, video_path) for this request."""
    image_path = _resolve_image(req.reference_image_path)
    image_sha = cache.hash_image_bytes(image_path.read_bytes())
    try:
        backend_tag = BACKEND_LOCAL if req.api_key is None else backend_for_key(req.api_key)
    except UnsupportedKeyPrefix as e:
        raise HTTPException(400, str(e))
    job_id = cache.job_id_for(req.video_url, image_sha, backend_tag)
    return image_path, job_id, cache.cache_dir(job_id), jobs._video_temp_path(job_id)


# ---- Endpoints --------------------------------------------------------------

@router.post("/download")
async def debug_download(req: StageRequest) -> dict:
    """Download video + transcript. Returns metadata and whether captions were found."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/download: start", job_id)
    try:
        resolved_video, transcript, metadata = await jobs._io_task(req.video_url, job_dir, video_path)
        logger.info("[%s] debug/download: done — has_transcript=%s", job_id, transcript is not None)
        return {
            "job_id": job_id,
            "video_path": str(resolved_video),
            "has_transcript": transcript is not None,
            "metadata": metadata,
        }
    except Exception:
        logger.exception("[%s] debug/download: failed", job_id)
        raise


@router.post("/vlm")
async def debug_vlm(req: StageRequest) -> dict:
    """Run VLM on the reference image. Returns the generated reference JSON."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/vlm: start", job_id)
    try:
        backend_tag, reference = await jobs._mps_task(
            image_path, req.api_key, req.product_name_hint, job_dir
        )
        logger.info("[%s] debug/vlm: done — backend=%s", job_id, backend_tag)
        return {
            "job_id": job_id,
            "backend": backend_tag,
            "reference": reference,
        }
    except Exception:
        logger.exception("[%s] debug/vlm: failed", job_id)
        raise


@router.post("/sponsors")
async def debug_sponsors(req: StageRequest) -> dict:
    """Detect sponsor intervals. Returns sponsor windows and sampling mode."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/sponsors: start", job_id)
    try:
        _, transcript, _ = await jobs._io_task(req.video_url, job_dir, video_path)
        sponsors = []
        if transcript is not None:
            sponsors = await asyncio.to_thread(jobs._stage_sponsors, transcript, job_dir)
        mode = "sponsor" if sponsors else "whole_video"
        logger.info("[%s] debug/sponsors: done — sampling_mode=%s count=%d", job_id, mode, len(sponsors))
        return {
            "job_id": job_id,
            "sampling_mode": mode,
            "sponsors": sponsors,
        }
    except Exception:
        logger.exception("[%s] debug/sponsors: failed", job_id)
        raise


@router.post("/frames")
async def debug_frames(req: StageRequest) -> dict:
    """Sample frames from the video. Returns frame count and timestamps."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/frames: start", job_id)
    try:
        resolved_video, transcript, _ = await jobs._io_task(req.video_url, job_dir, video_path)
        sponsors = []
        if transcript is not None:
            sponsors = await asyncio.to_thread(jobs._stage_sponsors, transcript, job_dir)
        frames = await asyncio.to_thread(jobs._stage_frames, resolved_video, sponsors)
        logger.info("[%s] debug/frames: done — frame_count=%d", job_id, len(frames))
        return {
            "job_id": job_id,
            "frame_count": len(frames),
            "timestamps": [f.timestamp for f in frames],
        }
    except Exception:
        logger.exception("[%s] debug/frames: failed", job_id)
        raise


@router.post("/detect")
async def debug_detect(req: StageRequest) -> dict:
    """Run OWLv2 text-guided detection. Returns all bounding boxes and scores."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/detect: start", job_id)
    try:
        resolved_video, transcript, _ = await jobs._io_task(req.video_url, job_dir, video_path)
        _, reference = await jobs._mps_task(image_path, req.api_key, req.product_name_hint, job_dir)
        sponsors = []
        if transcript is not None:
            sponsors = await asyncio.to_thread(jobs._stage_sponsors, transcript, job_dir)
        frames = await asyncio.to_thread(jobs._stage_frames, resolved_video, sponsors)
        dets = await asyncio.to_thread(jobs._stage_owlv2, frames, reference, job_dir)
        logger.info("[%s] debug/detect: done — detection_count=%d", job_id, len(dets))
        return {
            "job_id": job_id,
            "detection_count": len(dets),
            "detections": [jobs._obj_det_to_dict(d) for d in dets],
        }
    except Exception:
        logger.exception("[%s] debug/detect: failed", job_id)
        raise


@router.post("/score")
async def debug_score(req: StageRequest) -> dict:
    """Run brand scoring (NaFlex + OCR). Returns naflex_score and ocr_score per detection."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/score: start", job_id)
    try:
        resolved_video, transcript, _ = await jobs._io_task(req.video_url, job_dir, video_path)
        _, reference = await jobs._mps_task(image_path, req.api_key, req.product_name_hint, job_dir)
        sponsors = []
        if transcript is not None:
            sponsors = await asyncio.to_thread(jobs._stage_sponsors, transcript, job_dir)
        frames = await asyncio.to_thread(jobs._stage_frames, resolved_video, sponsors)
        dets = await asyncio.to_thread(jobs._stage_owlv2, frames, reference, job_dir)
        logo_emb_path = job_dir / "logo_embedding.npy"
        brand_dets = await asyncio.to_thread(
            jobs._stage_brand_scoring, dets, frames, reference, logo_emb_path, job_dir
        )
        logger.info("[%s] debug/score: done — detection_count=%d", job_id, len(brand_dets))
        return {
            "job_id": job_id,
            "detection_count": len(brand_dets),
            "detections": [jobs._brand_det_to_dict(d) for d in brand_dets],
        }
    except Exception:
        logger.exception("[%s] debug/score: failed", job_id)
        raise


@router.post("/intervals")
async def debug_intervals(req: StageRequest) -> dict:
    """Build intervals from cached detections. Useful for sweeping thresholds without re-running models."""
    image_path, job_id, job_dir, video_path = _job_ctx(req)
    logger.info("[%s] debug/intervals: start", job_id)
    try:
        resolved_video, transcript, _ = await jobs._io_task(req.video_url, job_dir, video_path)
        _, reference = await jobs._mps_task(image_path, req.api_key, req.product_name_hint, job_dir)
        sponsors = []
        if transcript is not None:
            sponsors = await asyncio.to_thread(jobs._stage_sponsors, transcript, job_dir)
        frames = await asyncio.to_thread(jobs._stage_frames, resolved_video, sponsors)
        dets = await asyncio.to_thread(jobs._stage_owlv2, frames, reference, job_dir)
        logo_emb_path = job_dir / "logo_embedding.npy"
        brand_dets = await asyncio.to_thread(
            jobs._stage_brand_scoring, dets, frames, reference, logo_emb_path, job_dir
        )
        intervals = jobs._stage_intervals(
            brand_dets, req.score_threshold, req.max_gap_seconds, req.min_frames
        )
        logger.info("[%s] debug/intervals: done — interval_count=%d", job_id, len(intervals))
        return {
            "job_id": job_id,
            "interval_count": len(intervals),
            "intervals": intervals,
        }
    except Exception:
        logger.exception("[%s] debug/intervals: failed", job_id)
        raise
