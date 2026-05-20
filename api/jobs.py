"""In-memory job store, SSE fan-out, and the background pipeline orchestrator.

Async job model:
  - A job record lives in the in-memory ``_jobs`` dict. No persistence layer,
    no external queue — this is a single-process prototype.
  - ``run_brand_detection_pipeline`` is the single background orchestrator,
    started via FastAPI ``BackgroundTasks``. It runs the real pipeline stages
    sequentially, each offloaded to a worker thread with ``asyncio.to_thread``
    so the event loop (and every open SSE stream) stays responsive.
  - State changes go through ``update_job``, which both mutates the record
    and pushes a snapshot to every SSE subscriber queue for that job.

Single-flight: the ML stages load multi-GB models onto a 16 GB MPS budget,
so two concurrent jobs would OOM. ``_job_lock`` serializes orchestrator runs
— a second job's status simply stays "queued" until the lock frees. This is
the one concurrency control kept; there is intentionally no queue/Redis/etc.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from fastapi import Request

from pipeline import (
    brand_detector,
    frame_sampler,
    interval_builder,
    object_detector,
    sponsor_detect,
)

from . import reference_gen, video_download


logger = logging.getLogger("videofind.jobs")

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = REPO_ROOT / "data" / "artifacts"

# Interval-builder knobs. The /v1 request only carries the two URLs, so these
# pipeline parameters use fixed defaults.
SCORE_THRESHOLD = 0.6
MAX_GAP_SECONDS = 2.0
MIN_FRAMES = 2

TERMINAL_STATES = frozenset({"completed", "failed"})

# stage -> (progress %, human message). Drives both the job record and SSE.
STAGES: dict[str, tuple[int, str]] = {
    "queued": (0, "Job queued"),
    "downloading_video": (10, "Downloading YouTube video"),
    "describing_reference_image": (25, "Describing the reference image with the VLM"),
    "sponsor_detection": (40, "Detecting sponsor segments in the transcript"),
    "frame_sampling": (50, "Sampling video frames"),
    "running_owlv2_detection": (65, "Running OWLv2 detection over sampled frames"),
    "running_siglip_verification": (85, "Verifying detections with SigLIP-2 NaFlex"),
    "aggregating_timestamps": (95, "Aggregating detections into timestamp intervals"),
    "completed": (100, "Brand detection completed"),
}

# ---- In-memory state --------------------------------------------------------

_jobs: dict[str, dict[str, Any]] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}
# Serializes orchestrator runs so only one job touches the GPU at a time.
_job_lock = asyncio.Lock()


def new_job_id() -> str:
    return uuid.uuid4().hex


def create_job(job_id: str) -> dict[str, Any]:
    """Register a fresh job in the "queued" state."""
    job = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "message": "Job queued",
        "result": None,
        "error": None,
        "created_at": time.time(),
    }
    _jobs[job_id] = job
    _subscribers.setdefault(job_id, [])
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **changes: Any) -> None:
    """Mutate the job record and notify every SSE subscriber for this job."""
    job = _jobs[job_id]
    job.update(changes)
    if "stage" in changes:
        logger.info("[%s] stage -> %s", job_id, changes["stage"])
    if changes.get("status") in TERMINAL_STATES:
        logger.info("[%s] %s", job_id, changes["status"])
    snapshot = dict(job)
    for queue in list(_subscribers.get(job_id, [])):
        queue.put_nowait(snapshot)


def _set_stage(job_id: str, stage: str) -> None:
    progress, message = STAGES[stage]
    update_job(job_id, status="running", stage=stage, progress=progress, message=message)


# ---- SSE --------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _format_event(job: dict[str, Any]) -> str:
    """Render a job snapshot as one SSE event, named by the job's status."""
    job_id = job["job_id"]
    status = job["status"]
    if status == "completed":
        return _sse("completed", {
            "job_id": job_id,
            "status": status,
            "stage": job["stage"],
            "progress": job["progress"],
            "message": job["message"],
            "result": job["result"],
            "result_url": f"/v1/jobs/{job_id}/result",
        })
    if status == "failed":
        return _sse("failed", {
            "job_id": job_id,
            "status": status,
            "stage": job["stage"],
            "error": job["error"],
        })
    return _sse("progress", {
        "job_id": job_id,
        "status": status,
        "stage": job["stage"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job["error"],
    })


async def sse_event_stream(job_id: str, request: Request) -> AsyncIterator[str]:
    """Yield SSE frames for one client connection.

    Emits the current state immediately (so a reconnect / page refresh is
    correct), then streams each subsequent state change. A 15 s idle
    heartbeat keeps the connection alive through proxies. The stream closes
    once the job reaches a terminal state or the client disconnects.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(job_id, []).append(queue)
    try:
        job = _jobs[job_id]
        yield _format_event(job)
        if job["status"] in TERMINAL_STATES:
            return

        while True:
            if await request.is_disconnected():
                break
            try:
                snapshot = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _format_event(snapshot)
            if snapshot["status"] in TERMINAL_STATES:
                break
    finally:
        subs = _subscribers.get(job_id)
        if subs and queue in subs:
            subs.remove(queue)


# ---- Pipeline plumbing ------------------------------------------------------

def _free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _run_sponsors(transcript: dict) -> list[dict]:
    model, tokenizer, classifier = sponsor_detect.load_models()
    try:
        regions = sponsor_detect.find_sponsor_intervals(
            transcript, model=model, tokenizer=tokenizer, classifier=classifier
        )
    finally:
        del model, tokenizer, classifier
        _free_memory()
    return [r for r in regions if r.get("category") == "sponsor"]


def _run_frames(video_path: Path, sponsors: list[dict]):
    if sponsors:
        return frame_sampler.sample_frames(video_path, sponsors)
    return frame_sampler.sample_frames_whole_video(video_path)


def _run_owlv2(frames, reference: dict):
    text_queries = [reference["primary_detection_prompt"]] + list(
        reference.get("alternate_prompts", [])
    )
    model, processor = object_detector.load_owlv2()
    try:
        return object_detector.detect_objects(
            frames, model=model, processor=processor, text_queries=text_queries
        )
    finally:
        del model, processor
        _free_memory()


def _run_verification(detections, frames, reference: dict, brand_image_path: Path):
    ocr_reader, naflex_model, naflex_processor = brand_detector.load_brand_models()
    try:
        # Embed the brand image (logo) — this is the vector each detected crop
        # is cosine-compared against. No separate cached .npy file.
        logo_emb = brand_detector.embed_logo(
            naflex_model, naflex_processor, brand_image_path
        )
        return brand_detector.score_detections(
            detections,
            frames,
            ocr_reader=ocr_reader,
            naflex_model=naflex_model,
            naflex_processor=naflex_processor,
            logo_emb=logo_emb,
            brand_keywords=list(reference.get("brand_text_keywords", [])),
        )
    finally:
        del ocr_reader, naflex_model, naflex_processor
        _free_memory()


def _run_intervals(brand_dets) -> list[dict]:
    intervals, _ = interval_builder.build_intervals(
        brand_dets,
        score_attr="naflex_score",
        score_threshold=SCORE_THRESHOLD,
        max_gap_seconds=MAX_GAP_SECONDS,
        min_frames=MIN_FRAMES,
        include_text=False,
    )
    return intervals


def _save_result(job_id: str, result: dict) -> None:
    """Persist the final result to disk atomically (temp file + os.replace)."""
    path = ARTIFACTS_ROOT / job_id / "result.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, path)


# ---- Orchestrator -----------------------------------------------------------

async def run_brand_detection_pipeline(
    job_id: str,
    youtube_url: str,
    object_image_path: Path,
    brand_image_path: Path,
) -> None:
    """Single background task: run the whole pipeline for one job.

    Two reference images, each wired to exactly one role:
      - ``object_image_path`` (product photo) -> the VLM describe stage, which
        produces the OWLv2 detection prompts and OCR brand keywords.
      - ``brand_image_path`` (brand logo) -> the NaFlex verification stage,
        which embeds it as the vector detected crops are scored against.

    Dependent stages run sequentially with ``await``. Each blocking ML stage
    is dispatched through ``asyncio.to_thread`` so it never stalls the event
    loop or the SSE streams. ``_job_lock`` ensures only one job runs at a
    time (16 GB MPS budget); a waiting job's status stays "queued".
    """
    async with _job_lock:
        job_dir = ARTIFACTS_ROOT / job_id
        resolved_video: Path | None = None
        try:
            # Stage 1: download video + auto-captions + metadata (yt-dlp).
            _set_stage(job_id, "downloading_video")
            download = await asyncio.to_thread(
                video_download.download, youtube_url, job_dir / "video"
            )
            resolved_video = download.video_path
            transcript = download.transcript

            # Stage 2: VLM describes the object image. generate_reference_local
            # is already async (it spawns the Qwen subprocess) — await directly.
            _set_stage(job_id, "describing_reference_image")
            reference = await reference_gen.generate_reference_local(
                object_image_path, None, job_dir / "reference_raw.txt"
            )

            # Stage 3: sponsor segments (no-op when the video has no transcript).
            _set_stage(job_id, "sponsor_detection")
            sponsors: list[dict] = []
            if transcript is not None:
                sponsors = await asyncio.to_thread(_run_sponsors, transcript)

            # Stage 4: decode frames (inside sponsor regions, else whole video).
            _set_stage(job_id, "frame_sampling")
            frames = await asyncio.to_thread(_run_frames, resolved_video, sponsors)

            # Stage 5: OWLv2 text-guided detection on every sampled frame.
            _set_stage(job_id, "running_owlv2_detection")
            detections = await asyncio.to_thread(_run_owlv2, frames, reference)

            # Stage 6: SigLIP-2 NaFlex (+ OCR) verification of each box.
            # NaFlex embeds the brand image; OCR matches the VLM's keywords.
            _set_stage(job_id, "running_siglip_verification")
            brand_dets = await asyncio.to_thread(
                _run_verification, detections, frames, reference, brand_image_path
            )

            # Stage 7: group score-positive frames into timestamp intervals.
            _set_stage(job_id, "aggregating_timestamps")
            intervals = await asyncio.to_thread(_run_intervals, brand_dets)

            result = {
                "brand_description": reference.get("primary_detection_prompt", ""),
                "timestamps": [
                    {
                        "start": iv["start"],
                        "end": iv["end"],
                        "confidence": iv["max_naflex_score"],
                    }
                    for iv in intervals
                ],
            }

            # Persist server-side BEFORE emitting the completed SSE event.
            # Why both: SSE delivery is best-effort — the client connection can
            # drop mid-run — so the on-disk copy is the durable source of truth
            # that GET /v1/jobs/{id}/result can always read. The completed SSE
            # event also carries the result inline so a still-connected client
            # gets it immediately without a second request.
            _save_result(job_id, result)
            update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                message=STAGES["completed"][1],
                result=result,
            )
        except Exception as e:
            # BackgroundTasks swallows exceptions, so the orchestrator must
            # record its own terminal failure state.
            logger.exception("[%s] pipeline failed", job_id)
            update_job(
                job_id,
                status="failed",
                stage="failed",
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            # The downloaded video is large and regenerable — drop it. The
            # reference image and result.json stay under data/artifacts/.
            try:
                if resolved_video is not None and resolved_video.is_file():
                    resolved_video.unlink()
            except OSError:
                pass
