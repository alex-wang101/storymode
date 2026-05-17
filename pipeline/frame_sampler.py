"""Stage 2 frame sampler.

Given a video and sponsor intervals from stage 1, return decoded RGB frames
sampled at ``fps`` within each interval expanded by ``buffer_seconds`` on
both sides. Overlapping buffered windows are merged so the same frame is
never emitted twice. Library function only -- no disk writes, no CLI.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

# SampledFrame lives in the shared schema module; re-export so existing
# callers can keep importing it from here.
from pipeline.schema import SampledFrame  # noqa: F401


logger = logging.getLogger(__name__)


def _buffered_windows(regions, buffer_seconds, duration):
    """Clamp and buffer each input region; drop any that collapse to non-positive length."""
    windows = []
    for i, r in enumerate(regions):
        start = max(0.0, float(r["start"]) - buffer_seconds)
        end = min(duration, float(r["end"]) + buffer_seconds)
        if end > start:
            windows.append((start, end, i))
    return windows


def _merge_windows(windows):
    """Merge overlapping windows; the merged window inherits the first contributor's index."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w[0])
    merged = [windows[0]]
    for w_start, w_end, idx in windows[1:]:
        last_start, last_end, last_idx = merged[-1]
        if w_start <= last_end:
            merged[-1] = (last_start, max(last_end, w_end), last_idx)
        else:
            merged.append((w_start, w_end, idx))
    return merged


def _sample_window(cap, w_start, w_end, region_index, fps):
    """Seek once to ``w_start``, then grab forward emitting frames at 1/fps spacing."""
    targets = np.arange(w_start, w_end, 1.0 / fps).tolist()
    if not targets:
        return []
    cap.set(cv2.CAP_PROP_POS_MSEC, w_start * 1000.0)
    out = []
    i = 0
    while i < len(targets):
        if not cap.grab():
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if pos >= targets[i]:
            ok, frame_bgr = cap.retrieve()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out.append(SampledFrame(timestamp=pos, image=frame_rgb, region_index=region_index))
            logger.debug("region %d: requested %.3fs actual %.3fs", region_index, targets[i], pos)
            while i < len(targets) and targets[i] <= pos:
                i += 1
    return out


def sample_frames(video_path, sponsor_regions, fps=1.0, buffer_seconds=5.0):
    """Decode RGB frames from ``video_path`` at ``fps`` within each buffered sponsor region.

    Returns a list of ``SampledFrame`` sorted by timestamp ascending. Empty
    if ``sponsor_regions`` is empty.
    """
    if not sponsor_regions:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if video_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"video has zero duration or unreadable metadata: {video_path}")
    duration = frame_count / video_fps

    windows = _merge_windows(_buffered_windows(sponsor_regions, buffer_seconds, duration))

    frames = []
    for idx, (w_start, w_end, region_index) in enumerate(windows):
        t0 = time.perf_counter()
        emitted = _sample_window(cap, w_start, w_end, region_index, fps)
        frames.extend(emitted)
        logger.info(
            "window %d [%.2fs, %.2fs] region_index=%d frames=%d elapsed=%.2fs",
            idx, w_start, w_end, region_index, len(emitted), time.perf_counter() - t0,
        )

    cap.release()
    frames.sort(key=lambda f: f.timestamp)
    return frames


def sample_frames_whole_video(video_path, fps=1.0):
    """Decode RGB frames from ``video_path`` at ``fps`` across the entire video.

    Used by the API path when no sponsor regions are available (no transcript,
    or transcript with zero detected sponsors). Returns ``SampledFrame``s
    with ``region_index=0`` so the rest of the pipeline treats them as a
    single contiguous region.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if video_fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f"video has zero duration or unreadable metadata: {video_path}")
    duration = frame_count / video_fps

    frames = _sample_window(cap, 0.0, duration, region_index=0, fps=fps)
    cap.release()
    frames.sort(key=lambda f: f.timestamp)
    return frames


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    video = Path("/Users/alexwang/Documents/GitHub/storymode/data/videos/zbiotics-bacon.mp4")
    regions = [{"start": 60.0, "end": 120.0}]

    frames = sample_frames(video, regions)
    print(f"got {len(frames)} frames")
    if frames:
        print(f"first timestamp: {frames[0].timestamp:.3f}s")
        print(f"last timestamp:  {frames[-1].timestamp:.3f}s")
        print(f"shape: {frames[0].image.shape}, dtype: {frames[0].image.dtype}")
