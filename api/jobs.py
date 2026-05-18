"""Per-job orchestrator.

Phase 1 — asyncio.gather(io_task, mps_task):
  io_task:  yt-dlp download (video + transcript + metadata)
  mps_task: VLM → reference.json, then NaFlex → logo_embedding.npy (sequential)

Phase 2 — sequential pipeline stages:
  sponsors → frames → OWLv2 → brand scoring → interval build

Queue(maxsize=1) keeps only one job in memory at a time (16 GB budget).
wait_for(30 min) prevents a stuck yt-dlp from pinning the worker forever.

State notifications: each job has an asyncio.Condition in _job_events.
_notify() updates _jobs and wakes any WebSocket handlers watching that job.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import SecretStr

from pipeline import brand_detector, frame_sampler, object_detector, sponsor_detect
from pipeline.brand_detector import embed_logo, load_brand_models
from pipeline.interval_builder import build_intervals
from pipeline.schema import BrandDetection, ObjectDetection

from . import cache
from . import reference_gen
from . import video_download
from .reference_gen import BACKEND_LOCAL, UnsupportedKeyPrefix, backend_for_key


logger = logging.getLogger("videofind.jobs")

JOB_TIMEOUT_SECONDS = 30 * 60

_jobs: dict[str, dict[str, Any]] = {}
_job_events: dict[str, asyncio.Condition] = {}  # one Condition per job_id
_queue: asyncio.Queue = asyncio.Queue(maxsize=1)


class JobBusy(Exception):
    pass


async def _notify(job_id: str, **updates: Any) -> None:
    """Update job state and wake all long-poll handlers watching this job."""
    _jobs[job_id].update(updates)
    if "stage" in updates:
        logger.info("[%s] stage → %s", job_id, updates["stage"])
    if updates.get("status") in ("done", "failed"):
        logger.info("[%s] %s", job_id, updates["status"])
    cond = _job_events.get(job_id)
    if cond is not None:
        async with cond:
            cond.notify_all()


def _video_temp_path(job_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "videofind" / job_id


def _obj_det_to_dict(d: ObjectDetection) -> dict:
    return {
        "timestamp": d.timestamp,
        "region_index": d.region_index,
        "box": list(d.box),
        "owlv2_score": d.owlv2_score,
    }


def _brand_det_to_dict(d: BrandDetection) -> dict:
    return {
        **_obj_det_to_dict(d),
        "ocr_score": d.ocr_score,
        "ocr_texts": d.ocr_texts,
        "naflex_score": d.naflex_score,
    }


def _brand_det_from_dict(d: dict) -> BrandDetection:
    return BrandDetection(
        timestamp=d["timestamp"],
        region_index=d["region_index"],
        box=tuple(d["box"]),
        owlv2_score=d["owlv2_score"],
        ocr_score=d["ocr_score"],
        ocr_texts=d["ocr_texts"],
        naflex_score=d["naflex_score"],
    )


# ---- Phase 1 tasks ----------------------------------------------------------

async def _io_task(
    video_url: str,
    job_dir: Path,
    video_path: Path,
) -> tuple[Path, dict | None, dict]:
    """Download video + transcript + metadata in one yt-dlp call."""
    path_marker = job_dir / "video_path.txt"
    transcript_path = job_dir / "transcript.json"
    metadata_path = job_dir / "metadata.json"

    if (
        path_marker.is_file()
        and Path(path_marker.read_text().strip()).is_file()
        and cache.has_json(metadata_path)
    ):
        logger.debug("cache hit: download (%s)", job_dir.name)
        existing_video = Path(path_marker.read_text().strip())
        metadata = cache.read_json(metadata_path)
        transcript = cache.read_json(transcript_path) if cache.has_json(transcript_path) else None
        return existing_video, transcript, metadata

    result = await asyncio.to_thread(video_download.download, video_url, video_path)
    path_marker.write_text(str(result.video_path))
    cache.write_json_atomic(metadata_path, result.metadata)
    if result.transcript is not None:
        cache.write_json_atomic(transcript_path, result.transcript)
    else:
        cache.write_json_atomic(transcript_path, None)
    return result.video_path, result.transcript, result.metadata


async def _mps_task(
    image_path: Path,
    api_key: SecretStr | None,
    hint: str | None,
    job_dir: Path,
) -> tuple[str, dict]:
    # Sequential: VLM and NaFlex both need MPS/GPTQ kernels — can't overlap.
    reference_path = job_dir / "reference.json"
    logo_emb_path = job_dir / "logo_embedding.npy"
    raw_output_path = job_dir / "reference_raw.txt"

    if cache.has_json(reference_path):
        reference = cache.read_json(reference_path)
        backend_tag = reference.pop("__backend__", BACKEND_LOCAL)
    else:
        if api_key is None:
            reference = await reference_gen.generate_reference_local(
                image_path, hint, raw_output_path
            )
            backend_tag = BACKEND_LOCAL
        else:
            backend_tag, reference = await reference_gen.generate_reference_cloud(
                image_path, api_key, hint
            )
        to_persist = dict(reference)
        to_persist["__backend__"] = backend_tag
        cache.write_json_atomic(reference_path, to_persist)

    if not logo_emb_path.is_file():
        await asyncio.to_thread(_embed_reference_image, image_path, logo_emb_path)

    return backend_tag, reference


def _embed_reference_image(image_path: Path, out_path: Path) -> None:
    _, naflex_model, naflex_proc = load_brand_models()
    emb = embed_logo(naflex_model, naflex_proc, image_path)
    np.save(out_path, emb.detach().cpu().numpy())
    del naflex_model, naflex_proc
    _free_memory()


# ---- Phase 2 stages ---------------------------------------------------------

def _free_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _stage_sponsors(transcript: dict, job_dir: Path) -> list[dict]:
    out_path = job_dir / "sponsors.json"
    if cache.has_json(out_path):
        logger.debug("cache hit: sponsors.json (%s)", job_dir.name)
        return cache.read_json(out_path)

    model, tokenizer, classifier = sponsor_detect.load_models()
    regions = sponsor_detect.find_sponsor_intervals(
        transcript, model=model, tokenizer=tokenizer, classifier=classifier
    )
    del model, tokenizer, classifier
    _free_memory()

    sponsors = [r for r in regions if r.get("category") == "sponsor"]
    cache.write_json_atomic(out_path, sponsors)
    return sponsors


def _stage_frames(video_path: Path, sponsors: list[dict]):
    if sponsors:
        return frame_sampler.sample_frames(video_path, sponsors)
    return frame_sampler.sample_frames_whole_video(video_path)


def _stage_owlv2(frames, reference: dict, job_dir: Path) -> list[ObjectDetection]:
    out_path = job_dir / "detections.jsonl"
    if cache.has_jsonl(out_path):
        logger.debug("cache hit: detections.jsonl (%s)", job_dir.name)
        return [
            ObjectDetection(
                timestamp=r["timestamp"],
                region_index=r["region_index"],
                box=tuple(r["box"]),
                owlv2_score=r["owlv2_score"],
            )
            for r in cache.read_jsonl(out_path)
        ]

    text_queries = [reference["primary_detection_prompt"]] + list(
        reference.get("alternate_prompts", [])
    )
    model, proc = object_detector.load_owlv2()
    dets = object_detector.detect_objects(
        frames, model=model, processor=proc, text_queries=text_queries
    )
    del model, proc
    _free_memory()

    with cache.open_jsonl_writer(out_path) as f:
        for d in dets:
            f.write(json.dumps(_obj_det_to_dict(d)) + "\n")
    return dets


def _stage_brand_scoring(
    dets: list[ObjectDetection],
    frames,
    reference: dict,
    logo_emb_path: Path,
    job_dir: Path,
) -> list[BrandDetection]:
    out_path = job_dir / "brand_detections.jsonl"
    if cache.has_jsonl(out_path):
        logger.debug("cache hit: brand_detections.jsonl (%s)", job_dir.name)
        return [_brand_det_from_dict(r) for r in cache.read_jsonl(out_path)]

    ocr_reader, naflex_model, naflex_proc = brand_detector.load_brand_models()
    logo_emb_np = np.load(logo_emb_path)
    logo_emb = torch.from_numpy(logo_emb_np).to(naflex_model.device)

    brand_dets = brand_detector.score_detections(
        dets,
        frames,
        ocr_reader=ocr_reader,
        naflex_model=naflex_model,
        naflex_processor=naflex_proc,
        logo_emb=logo_emb,
        brand_keywords=list(reference.get("brand_text_keywords", [])),
    )
    del ocr_reader, naflex_model, naflex_proc, logo_emb
    _free_memory()

    with cache.open_jsonl_writer(out_path) as f:
        for d in brand_dets:
            f.write(json.dumps(_brand_det_to_dict(d)) + "\n")
    return brand_dets


def _stage_intervals(brand_dets, score_threshold, max_gap_seconds, min_frames):
    intervals, _ = build_intervals(
        brand_dets,
        score_attr="naflex_score",
        score_threshold=score_threshold,
        max_gap_seconds=max_gap_seconds,
        min_frames=min_frames,
        include_text=False,
    )
    return intervals


# ---- Internal pipeline runner -----------------------------------------------

async def _run_job_inner(
    *,
    video_url: str,
    reference_image_path: Path,
    api_key: SecretStr | None,
    product_name_hint: str | None,
    score_threshold: float,
    max_gap_seconds: float,
    min_frames: int,
    job_id: str,
    job_dir: Path,
    video_path: Path,
    backend_tag: str,
) -> dict:
    # Stage 1: download video+transcript concurrently with VLM processing the reference image
    await _notify(job_id, stage="download")
    io = _io_task(video_url, job_dir, video_path)
    mps = _mps_task(reference_image_path, api_key, product_name_hint, job_dir)
    (resolved_video, transcript, metadata), (_backend_tag, reference) = await asyncio.gather(io, mps)

    await _notify(job_id, stage="sponsors")
    sponsors: list[dict] = []
    if transcript is not None:
        sponsors = await asyncio.to_thread(_stage_sponsors, transcript, job_dir)

    await _notify(job_id, stage="frames")
    frames = await asyncio.to_thread(_stage_frames, resolved_video, sponsors)

    await _notify(job_id, stage="detect")
    dets = await asyncio.to_thread(_stage_owlv2, frames, reference, job_dir)

    await _notify(job_id, stage="score")
    logo_emb_path = job_dir / "logo_embedding.npy"
    brand_dets = await asyncio.to_thread(
        _stage_brand_scoring, dets, frames, reference, logo_emb_path, job_dir
    )

    await _notify(job_id, stage="intervals")
    intervals = _stage_intervals(brand_dets, score_threshold, max_gap_seconds, min_frames)

    try:
        if resolved_video.is_file():
            resolved_video.unlink()
    except OSError:
        pass

    reference_clean = {k: v for k, v in reference.items() if k != "__backend__"}
    return {
        "job_id": job_id,
        "video": metadata,
        "reference": reference_clean,
        "params": {
            "score_threshold": score_threshold,
            "max_gap_seconds": max_gap_seconds,
            "min_frames": min_frames,
        },
        "sampling_mode": "sponsor" if sponsors else "whole_video",
        "intervals": intervals,
    }


# ---- Public API -------------------------------------------------------------

async def submit_job(
    *,
    video_url: str,
    reference_image_path: Path,
    api_key: SecretStr | None,
    product_name_hint: str | None,
    score_threshold: float,
    max_gap_seconds: float,
    min_frames: int,
) -> str:
    """Queue a job and return its job_id immediately.

    Returns the same job_id if this (video, image, backend) tuple is already
    tracked — the caller can connect to WS /jobs/{id}/ws without re-submitting.
    Raises JobBusy if the queue is full, UnsupportedKeyPrefix on a bad key prefix.
    """
    image_bytes = reference_image_path.read_bytes()
    image_sha = cache.hash_image_bytes(image_bytes)
    backend_tag = BACKEND_LOCAL if api_key is None else backend_for_key(api_key)
    job_id = cache.job_id_for(video_url, image_sha, backend_tag)

    if job_id in _jobs:
        return job_id

    try:
        _queue.put_nowait((job_id, dict(
            video_url=video_url,
            reference_image_path=reference_image_path,
            api_key=api_key,
            product_name_hint=product_name_hint,
            score_threshold=score_threshold,
            max_gap_seconds=max_gap_seconds,
            min_frames=min_frames,
            job_id=job_id,
            job_dir=cache.cache_dir(job_id),
            video_path=_video_temp_path(job_id),
            backend_tag=backend_tag,
        )))
    except asyncio.QueueFull:
        raise JobBusy("job queue is full")

    _jobs[job_id] = {"status": "queued", "stage": None}
    _job_events[job_id] = asyncio.Condition()
    return job_id


async def job_worker() -> None:
    """Long-running background task. Pulls one job at a time from the queue."""
    while True:
        job_id, kwargs = await _queue.get()
        logger.info("[%s] dequeued", job_id)
        await _notify(job_id, status="running")
        try:
            result = await asyncio.wait_for(
                _run_job_inner(**kwargs),
                timeout=JOB_TIMEOUT_SECONDS,
            )
            await _notify(job_id, status="done", result=result)
        except Exception as e:
            stage = _jobs[job_id].get("stage")
            logger.exception("[%s] failed at stage %s", job_id, stage)
            await _notify(job_id, status="failed", error=type(e).__name__)
        finally:
            _queue.task_done()
