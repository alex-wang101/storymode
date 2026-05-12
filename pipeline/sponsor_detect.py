"""Sponsor segment detection via Xenova/sponsorblock-small (T5 seq2seq).

Library module. ``load_models()`` returns the (model, tokenizer, classifier)
triple once; ``find_sponsor_intervals(transcript, *, model, tokenizer,
classifier=None)`` returns sorted intervals.

Plain PyTorch CPU. No subprocess, no ONNX, no CLI.
"""

import re
import string
from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


MODEL_NAME = "Xenova/sponsorblock-small"
CLASSIFIER_NAME = "Xenova/sponsorblock-classifier"
PREFIX = "EXTRACT_SEGMENTS: "
NO_SEGMENT_TOKEN = "NO_SEGMENT_TOKEN"
TOKEN_PATTERN = re.compile(
    r"START_(?P<category>\w+)_TOKEN\s*(?P<text>.*?)\s*(?:END_\w+_TOKEN|$)",
    re.DOTALL,
)

MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
MAX_MODEL_TOKENS = 512
MERGE_GAP_SECONDS = 8.0
SNAP_TO_ZERO_THRESHOLD = 0.08
ALIGN_WORDS = 20


def load_models():
    """Load the seq2seq model, tokenizer, and optional classifier once.

    Returns ``(model, tokenizer, classifier_or_none)``. The classifier is
    optional and loads silently to ``None`` if ``Xenova/sponsorblock-classifier``
    is unavailable.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    try:
        classifier = pipeline("text-classification", model=CLASSIFIER_NAME)
    except Exception:
        classifier = None
    return model, tokenizer, classifier


@dataclass
class _Chunk:
    """A token-bounded slice of transcript items, retaining the source items for time recovery."""
    text: str
    items: list


def _count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _norm_word(w):
    return w.lower().strip(string.punctuation)


def _build_chunks(items, tokenizer):
    """Greedy token-bounded chunker with overlap.

    Each chunk's text is the space-joined item texts; each chunk keeps the
    source items so timestamps can be recovered after the model returns a
    matched span.
    """
    if not items:
        return []

    item_token_counts = [_count_tokens(tokenizer, it["text"]) for it in items]
    n = len(items)

    chunks = []
    i = 0
    while i < n:
        j = i
        token_sum = 0
        while j < n and token_sum + item_token_counts[j] <= MAX_CHUNK_TOKENS:
            token_sum += item_token_counts[j]
            j += 1
        if j == i:
            # Single item exceeds MAX_CHUNK_TOKENS; force it into its own chunk
            # (the inference loop will warn on truncation).
            j = i + 1

        chunk_items = items[i:j]
        text = " ".join(it["text"] for it in chunk_items)
        chunks.append(_Chunk(text=text, items=list(chunk_items)))

        if j >= n:
            break

        overlap_count = 0
        next_start = j
        for k in range(j - 1, i - 1, -1):
            if overlap_count + item_token_counts[k] > CHUNK_OVERLAP_TOKENS:
                break
            overlap_count += item_token_counts[k]
            next_start = k
        if next_start == i:
            next_start = i + 1  # guarantee progress
        i = next_start
    return chunks


def _find_best_alignment(needle, hay, start=0):
    """Best-fit position of needle in hay (>= start) by exact-match count.

    Accepts a match only if at least half of needle's tokens align exactly.
    Returns -1 if no match passes the threshold.
    """
    nl = len(needle)
    if nl == 0 or len(hay) < start + nl:
        return -1
    threshold = max(1, nl // 2)
    best_score = 0
    best_pos = -1
    for s in range(start, len(hay) - nl + 1):
        score = sum(1 for k in range(nl) if hay[s + k][0] == needle[k])
        if score > best_score and score >= threshold:
            best_score = score
            best_pos = s
    return best_pos


def _align_to_items(matched_text, items):
    """Recover (start, end) seconds for matched_text within the chunk's items.

    Greedy sublist match: align the first ALIGN_WORDS and last ALIGN_WORDS of
    matched_text against the chunk's word stream. Returns ``None`` if the
    alignment fails (model hallucinated text not in the source).
    """
    matched_words = matched_text.split()
    if not matched_words:
        return None

    word_stream = []  # list of (norm_word, item_idx)
    for idx, item in enumerate(items):
        for w in item["text"].split():
            word_stream.append((_norm_word(w), idx))
    if not word_stream:
        return None

    n_first = min(ALIGN_WORDS, len(matched_words))
    n_last = min(ALIGN_WORDS, len(matched_words))
    first = [_norm_word(w) for w in matched_words[:n_first]]
    last = [_norm_word(w) for w in matched_words[-n_last:]]

    start_pos = _find_best_alignment(first, word_stream)
    if start_pos < 0:
        return None
    end_pos = _find_best_alignment(last, word_stream, start=start_pos)
    if end_pos < 0:
        end_pos = start_pos + len(first) - 1
    else:
        end_pos += len(last) - 1
    end_pos = min(end_pos, len(word_stream) - 1)

    start_item_idx = word_stream[start_pos][1]
    end_item_idx = word_stream[end_pos][1]
    return float(items[start_item_idx]["start"]), float(items[end_item_idx]["end"])


def _merge_intervals(predictions):
    """Merge overlapping or same-category-within-MERGE_GAP_SECONDS predictions."""
    if not predictions:
        return []
    preds = sorted(predictions, key=lambda p: p["start"])
    merged = [dict(preds[0])]
    for p in preds[1:]:
        last = merged[-1]
        overlaps = p["start"] <= last["end"]
        same_cat_close = (
            p["category"] == last["category"]
            and p["start"] - last["end"] <= MERGE_GAP_SECONDS
        )
        if overlaps or same_cat_close:
            last["end"] = max(last["end"], p["end"])
            last["text"] = last["text"] + " " + p["text"]
        else:
            merged.append(dict(p))
    return merged


def find_sponsor_intervals(transcript, *, model, tokenizer, classifier=None):
    """Detect sponsor / selfpromo / interaction intervals in a transcript dict.

    Args:
        transcript: canonical transcript dict (see transcript_ingest.py).
        model: T5 seq2seq model loaded via ``load_models()``.
        tokenizer: matching tokenizer.
        classifier: optional text-classification pipeline. When provided,
            refines per-prediction category and probability; drops predictions
            classified as ``"none"`` with score > 0.5.

    Returns:
        List of dicts sorted by ``start``, each with keys
        ``start``, ``end``, ``category``, ``probability`` (or ``None``), ``text``.
    """
    items = transcript.get("items", [])
    if not items:
        return []

    chunks = _build_chunks(items, tokenizer)

    predictions = []
    for chunk in chunks:
        input_text = PREFIX + chunk.text
        encoded = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_TOKENS,
        )
        input_ids = encoded.input_ids
        actual_len = int(input_ids.shape[1])
        if actual_len >= MAX_MODEL_TOKENS:
            print(
                f"warning: sponsor_detect chunk truncated at {MAX_MODEL_TOKENS} tokens "
                f"({actual_len} after prefix); chunking should have prevented this"
            )

        with torch.no_grad():
            generated = model.generate(input_ids, max_length=actual_len + 50)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

        if NO_SEGMENT_TOKEN in decoded:
            continue

        for m in TOKEN_PATTERN.finditer(decoded):
            category = m.group("category").lower()
            matched_text = (m.group("text") or "").strip()
            if not matched_text:
                continue
            span = _align_to_items(matched_text, chunk.items)
            if span is None:
                continue
            start, end = span
            predictions.append({
                "start": start,
                "end": end,
                "category": category,
                "text": matched_text,
                "probability": None,
            })

    merged = _merge_intervals(predictions)

    for p in merged:
        if p["start"] < SNAP_TO_ZERO_THRESHOLD:
            p["start"] = 0.0

    if classifier is not None:
        refined = []
        for p in merged:
            try:
                result = classifier(p["text"], truncation=True)
                top = result[0] if isinstance(result, list) else result
                label = top["label"].lower()
                score = float(top["score"])
            except Exception:
                refined.append(p)
                continue
            if label == "none" and score > 0.5:
                continue
            p["category"] = label
            p["probability"] = score
            refined.append(p)
        merged = refined

    return sorted(merged, key=lambda p: p["start"])
