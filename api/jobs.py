"""Per-job orchestrator.

Each call to ``run_job`` is one async function with two phases:

Phase 1 -- preparation, ``asyncio.gather(io_bundle, mps_bundle)``:
  * io_bundle: yt-dlp video + transcript + metadata in one call
  * mps_bundle (sequential -- MPS would serialize anyway):
      VLM generates reference.json  ->  NaFlex embeds reference image

Phase 2 -- sequential pipeline (each step wrapped in ``asyncio.to_thread``):
  1. T5 sponsor detection (skip + fall back to whole video if no transcript)
  2. Frame sampling (sponsor-based or whole-video)
  3. OWLv2 text-guided detection
  4. NaFlex + OCR brand scoring (NaFlex stays loaded from Phase 1 to avoid
     a second cold start)
  5. Build intervals on naflex_score

Concurrency: an ``asyncio.Semaphore(1)`` wraps the whole job so the
4-model memory budget on a 16 GB M4 Pro is never doubled. Bumping to
multi-job requires a queue and per-job worker pinning -- deferred.

Timeout: ``asyncio.wait_for(JOB_TIMEOUT_SECONDS)`` wraps the whole thing
so a stuck yt-dlp can't pin the worker forever.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import SecretStr

from pipeline.schema import BrandDetection, ObjectDetection

from . import cache
from . import reference_gen
from . import video_download
from .reference_gen import (
    BACKEND_LOCAL,
    UnsupportedKeyPrefix,
    VlmJsonError,
    backend_for_key,
)


JOB_TIMEOUT_SECONDS = 30 * 60
_job_lock = asyncio.Semaphore(1)


class JobBusy(Exception):
    pass


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
    """VLM -> reference.json, then SigLIP NaFlex -> logo_embedding.npy.

    Sequential because both stages compete for MPS / the GPTQ CPU kernels.
    Returns ``(backend_tag, reference_dict)``. The NaFlex model is NOT
    held resident here -- Phase 2 brand scoring reloads it; the warm-cache
    optimization is deferred until profiling shows it matters.
    """
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
    """Load NaFlex briefly, embed the reference image, persist as .npy, unload."""
    from pipeline.brand_detector import embed_logo, load_brand_models

    _, naflex_model, naflex_proc = load_brand_models()
    emb = embed_logo(naflex_model, naflex_proc, image_path)
    np.save(out_path, emb.detach().cpu().numpy())
    del naflex_model, naflex_proc
    _free_memory()


# ---- Phase 2 stages ---------------------------------------------------------

def _free_memory():
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _stage_sponsors(transcript: dict, job_dir: Path) -> list[dict]:
    out_path = job_dir / "sponsors.json"
    if cache.has_json(out_path):
        return cache.read_json(out_path)

    from pipeline import sponsor_detect

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
    from pipeline import frame_sampler

    if sponsors:
        return frame_sampler.sample_frames(video_path, sponsors)
    return frame_sampler.sample_frames_whole_video(video_path)


def _stage_owlv2(frames, reference: dict, job_dir: Path) -> list[ObjectDetection]:
    out_path = job_dir / "detections.jsonl"
    if cache.has_jsonl(out_path):
        return [
            ObjectDetection(
                timestamp=r["timestamp"],
                region_index=r["region_index"],
                box=tuple(r["box"]),
                owlv2_score=r["owlv2_score"],
            )
            for r in cache.read_jsonl(out_path)
        ]

    from pipeline import object_detector

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
        return [_brand_det_from_dict(r) for r in cache.read_jsonl(out_path)]

    import torch

    from pipeline import brand_detector

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
    from pipeline.interval_builder import build_intervals

    intervals, _ = build_intervals(
        brand_dets,
        score_attr="naflex_score",
        score_threshold=score_threshold,
        max_gap_seconds=max_gap_seconds,
        min_frames=min_frames,
        include_text=False,
    )
    return intervals


# ---- Public entry point -----------------------------------------------------

async def run_job(
    *,
    video_url: str,
    reference_image_path: Path,
    api_key: SecretStr | None,
    product_name_hint: str | None,
    score_threshold: float,
    max_gap_seconds: float,
    min_frames: int,
) -> dict:
    """Run one full job and return the response dict.

    Raises ``JobBusy`` if another job is already in flight,
    ``UnsupportedKeyPrefix`` if api_key prefix is unknown,
    ``VlmJsonError`` if the VLM output cannot be parsed/validated,
    ``video_download.LiveStreamRejected`` / ``video_download.VideoUnavailable``
    for download failures, ``asyncio.TimeoutError`` on overall job timeout.
    """
    image_bytes = reference_image_path.read_bytes()
    image_sha = cache.hash_image_bytes(image_bytes)

    if api_key is None:
        backend_tag = BACKEND_LOCAL
    else:
        backend_tag = backend_for_key(api_key)  # raises UnsupportedKeyPrefix early

    job_id = cache.job_id_for(video_url, image_sha, backend_tag)
    job_dir = cache.cache_dir(job_id)
    video_path = _video_temp_path(job_id)

    if _job_lock.locked():
        raise JobBusy("another job is in flight")

    async with _job_lock:
        return await asyncio.wait_for(
            _run_job_inner(
                video_url=video_url,
                reference_image_path=reference_image_path,
                api_key=api_key,
                product_name_hint=product_name_hint,
                score_threshold=score_threshold,
                max_gap_seconds=max_gap_seconds,
                min_frames=min_frames,
                job_id=job_id,
                job_dir=job_dir,
                video_path=video_path,
                backend_tag=backend_tag,
            ),
            timeout=JOB_TIMEOUT_SECONDS,
        )


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
    io = _io_task(video_url, job_dir, video_path)
    mps = _mps_task(reference_image_path, api_key, product_name_hint, job_dir)
    (resolved_video, transcript, metadata), (_backend_tag, reference) = await asyncio.gather(io, mps)

    if transcript is None:
        sponsors: list[dict] = []
    else:
        sponsors = await asyncio.to_thread(_stage_sponsors, transcript, job_dir)

    frames = await asyncio.to_thread(_stage_frames, resolved_video, sponsors)
    dets = await asyncio.to_thread(_stage_owlv2, frames, reference, job_dir)
    logo_emb_path = job_dir / "logo_embedding.npy"
    brand_dets = await asyncio.to_thread(
        _stage_brand_scoring, dets, frames, reference, logo_emb_path, job_dir
    )
    intervals = _stage_intervals(brand_dets, score_threshold, max_gap_seconds, min_frames)

    # The temp video can go now that brand_detections.jsonl is persisted;
    # the cache dir keeps video_path.txt as the pointer, and resume will
    # re-download if needed.
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
