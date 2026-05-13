"""Brand verification: OCR (brand text) and NaFlex (brand object) as two
independent signals.

For each ObjectDetection from the OWLv2 stage, crops the source frame at
the detection box and computes:

  - ``ocr_score``     -- EasyOCR + rapidfuzz against brand keywords;
                         answers "is the brand NAME visible?"
  - ``naflex_score``  -- SigLIP 2 NaFlex image-image against ``logo.png``;
                         answers "is the brand OBJECT visible?"

Both scores live on every BrandDetection. Downstream code (see
``pipeline/interval_builder.py``) aggregates whichever signal it cares
about into time intervals.

Library functions only. ``load_brand_models()`` loads EasyOCR + NaFlex
once; ``embed_logo()`` pre-computes the reference embedding;
``score_detections()`` does the per-detection work.
"""

from __future__ import annotations

import logging

import easyocr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rapidfuzz import fuzz
from transformers import AutoModel, AutoProcessor

from pipeline.schema import BrandDetection, ObjectDetection, SampledFrame


logger = logging.getLogger(__name__)

NAFLEX_ID = "google/siglip2-base-patch16-naflex"


def _device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_brand_models():
    """Load EasyOCR + NaFlex once.

    Returns ``(ocr_reader, naflex_model, naflex_processor)``. EasyOCR runs
    on CPU (its ``gpu=True`` flag is CUDA-specific and doesn't help on MPS).
    """
    ocr_reader = easyocr.Reader(["en"], gpu=False)
    processor = AutoProcessor.from_pretrained(NAFLEX_ID)
    model = (
        AutoModel.from_pretrained(NAFLEX_ID, torch_dtype=torch.float32)
        .to(_device())
        .eval()
    )
    return ocr_reader, model, processor


@torch.no_grad()
def _naflex_image_embed(image_pil, model, processor):
    """Return L2-normalized NaFlex image embedding for a PIL image."""
    inputs = processor(images=image_pil, return_tensors="pt").to(model.device)
    out = model.get_image_features(**inputs)
    if not isinstance(out, torch.Tensor):
        out = out.pooler_output if hasattr(out, "pooler_output") else out
    return F.normalize(out, dim=-1)


def embed_logo(naflex_model, naflex_processor, logo_path):
    """Embed ``logo.png`` as the brand reference. Returns an L2-normalized tensor."""
    logo = Image.open(logo_path).convert("RGB")
    return _naflex_image_embed(logo, naflex_model, naflex_processor)


def _ocr_score(crop_pil, ocr_reader, brand_keywords):
    """Run OCR on a crop and fuzzy-match against brand keywords.

    Returns ``(max_match_ratio in [0, 1], list of detected text strings)``.
    Empty list of strings means OCR found no text at all -- the fallback
    trigger.
    """
    arr = np.array(crop_pil)
    detected = ocr_reader.readtext(arr, detail=0)
    if not detected:
        return 0.0, []
    best = 0.0
    for text in detected:
        for kw in brand_keywords:
            ratio = fuzz.partial_ratio(text.lower(), kw.lower()) / 100.0
            if ratio > best:
                best = ratio
    return best, detected


def score_detections(
    detections: list[ObjectDetection],
    frames: list[SampledFrame],
    *,
    ocr_reader,
    naflex_model,
    naflex_processor,
    logo_emb,
    brand_keywords: list[str],
) -> list[BrandDetection]:
    """Compute OCR (brand-text) and NaFlex (brand-object) scores per box.

    Looks up each detection's source frame by ``(timestamp, region_index)``,
    crops at the box, runs OCR + NaFlex, and emits a BrandDetection that
    carries every ObjectDetection field forward plus the two independent
    score fields. The two signals are NOT combined here; downstream code
    decides which one to aggregate into intervals.
    """
    frame_lookup = {(round(sf.timestamp, 6), sf.region_index): sf for sf in frames}

    brand_dets: list[BrandDetection] = []
    for d in detections:
        sf = frame_lookup.get((round(d.timestamp, 6), d.region_index))
        if sf is None:
            logger.warning(
                "no source frame for detection t=%.3fs region=%d",
                d.timestamp, d.region_index,
            )
            continue

        x0, y0, x1, y1 = (int(v) for v in d.box)
        x0 = max(0, x0)
        y0 = max(0, y0)
        h, w = sf.image.shape[:2]
        x1 = min(x1, w)
        y1 = min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue

        crop_arr = sf.image[y0:y1, x0:x1]
        crop_pil = Image.fromarray(crop_arr)

        ocr_score, ocr_texts = _ocr_score(crop_pil, ocr_reader, brand_keywords)

        crop_emb = _naflex_image_embed(crop_pil, naflex_model, naflex_processor)
        naflex_score = float((crop_emb @ logo_emb.T).item())

        brand_dets.append(
            BrandDetection(
                timestamp=d.timestamp,
                region_index=d.region_index,
                box=d.box,
                owlv2_score=d.owlv2_score,
                ocr_score=float(ocr_score),
                ocr_texts=ocr_texts,
                naflex_score=naflex_score,
            )
        )

    return brand_dets
