"""Shared dataclasses used across pipeline stages.

Single source of truth for cross-stage data shapes. Each stage produces or
consumes one of these types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledFrame:
    """A decoded RGB frame at a known timestamp, tagged with its source sponsor region."""
    timestamp: float
    image: np.ndarray            # RGB uint8, shape (H, W, 3)
    region_index: int


@dataclass
class ObjectDetection:
    """A single OWLv2 bounding box on a single sampled frame."""
    timestamp: float
    region_index: int
    box: tuple[float, float, float, float]   # (x0, y0, x1, y1) in source-frame coords
    owlv2_score: float


@dataclass
class BrandDetection(ObjectDetection):
    """ObjectDetection plus two independent brand verification scores.

    ``ocr_score`` answers "is the brand NAME visible on this box?"
    ``naflex_score`` answers "is the brand OBJECT visible on this box?"
    The two signals are reported separately; downstream stages aggregate
    on whichever signal they care about (see ``pipeline/interval_builder.py``).
    """
    ocr_score: float                          # max rapidfuzz ratio vs brand keywords [0, 1]
    ocr_texts: list[str]                      # raw OCR strings detected on the crop
    naflex_score: float                       # cosine sim vs logo embedding [0, 1]
