"""OWLv2 text-guided object detection on sampled frames.

Library function. ``load_owlv2()`` loads the model once;
``detect_objects(frames, *, model, processor, text_queries)`` runs
frame-by-frame inference and returns a flat list of ObjectDetection
records carrying ``timestamp`` and ``region_index`` through unchanged.
"""

from __future__ import annotations

import logging
import time

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from pipeline.schema import ObjectDetection, SampledFrame


logger = logging.getLogger(__name__)

MODEL_ID = "google/owlv2-base-patch16-ensemble"


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_owlv2():
    """Load OWLv2 text-guided once. Returns ``(model, processor)``."""
    processor = Owlv2Processor.from_pretrained(MODEL_ID)
    model = (
        Owlv2ForObjectDetection.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        .to(_device())
        .eval()
    )
    return model, processor


def detect_objects(
    frames: list[SampledFrame],
    *,
    model,
    processor,
    text_queries: list[str],
    threshold: float = 0.10,
) -> list[ObjectDetection]:
    """Run OWLv2 text-guided on each frame, emit one ObjectDetection per box.

    Same invocation pattern as the OVOD_eval notebook's OWLv2-text cell:
    ``padding="max_length", truncation=True`` so the multi-prompt list
    tokenizes to a fixed shape.
    """
    detections: list[ObjectDetection] = []
    for sf in frames:
        t0 = time.perf_counter()
        pil = Image.fromarray(sf.image)
        inputs = processor(
            text=[text_queries],
            images=pil,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil.size[::-1]]).to(model.device)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()

        for box, score in zip(boxes, scores):
            detections.append(
                ObjectDetection(
                    timestamp=sf.timestamp,
                    region_index=sf.region_index,
                    box=tuple(float(v) for v in box),
                    owlv2_score=float(score),
                )
            )

        logger.debug(
            "owlv2 t=%.2fs region=%d boxes=%d elapsed=%.2fs",
            sf.timestamp, sf.region_index, len(boxes), time.perf_counter() - t0,
        )

    detections.sort(key=lambda d: d.timestamp)
    return detections
