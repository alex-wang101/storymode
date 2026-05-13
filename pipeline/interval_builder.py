"""Group score-positive frames into time intervals via temporal consistency.

Used by both ``detect-text`` (OCR signal) and ``detect-object`` (NaFlex
signal) in main.py. The two commands share this aggregation logic; they
differ only in which signal field they pass in.

Rule: a frame is "positive" if at least one of its OWLv2 boxes scores
above the threshold on the chosen signal. Consecutive positive frames
whose timestamps are within ``max_gap_seconds`` of each other belong to
the same interval. Intervals with fewer than ``min_frames`` frames are
dropped as single-frame flukes.
"""

from __future__ import annotations

from pipeline.schema import BrandDetection


def _per_frame_best(
    brand_dets: list[BrandDetection],
    score_attr: str,
    score_threshold: float,
) -> dict:
    """Return ``{(region_index, timestamp): BrandDetection}`` keeping the
    box with the highest signal score per frame, filtered to scores at or
    above ``score_threshold``. Frames with no qualifying box drop out."""
    bests = {}
    for d in brand_dets:
        score = getattr(d, score_attr)
        if score < score_threshold:
            continue
        key = (d.region_index, d.timestamp)
        cur = bests.get(key)
        if cur is None or score > getattr(cur, score_attr):
            bests[key] = d
    return bests


def build_intervals(
    brand_dets: list[BrandDetection],
    *,
    score_attr: str,
    score_threshold: float,
    max_gap_seconds: float = 2.0,
    min_frames: int = 2,
    include_text: bool = False,
) -> tuple[list[dict], dict]:
    """Group score-positive frames into time intervals.

    Returns ``(intervals, frame_bests)`` where ``frame_bests`` is the dict
    of per-frame winning detections (useful for downstream crop-saving).
    """
    bests = _per_frame_best(brand_dets, score_attr, score_threshold)
    positives = sorted(bests.values(), key=lambda d: d.timestamp)
    if not positives:
        return [], bests

    groups = [[positives[0]]]
    for d in positives[1:]:
        if d.timestamp - groups[-1][-1].timestamp <= max_gap_seconds:
            groups[-1].append(d)
        else:
            groups.append([d])

    intervals = []
    for group in groups:
        if len(group) < min_frames:
            continue
        best = max(group, key=lambda d: getattr(d, score_attr))
        iv = {
            "start": float(group[0].timestamp),
            "end": float(group[-1].timestamp),
            "duration_seconds": float(group[-1].timestamp - group[0].timestamp),
            "frame_count": len(group),
            "region_index": int(group[0].region_index),
            f"max_{score_attr}": float(getattr(best, score_attr)),
            "best_box": [float(v) for v in best.box],
        }
        if include_text:
            iv["best_text"] = " ".join(best.ocr_texts) if best.ocr_texts else ""
        intervals.append(iv)

    return intervals, bests
