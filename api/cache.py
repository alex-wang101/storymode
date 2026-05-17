"""Per-stage cache layout and atomic-write helpers.

Job id = sha256(video_url || ref_image_sha || backend_tag || PIPELINE_VERSION)[:16].
Threshold params live outside the key -- they only affect the millisecond
interval-build step, which is rebuilt from brand_detections.jsonl every
request. See plan: data/cache/jobs/<job_id>/ for the artifact layout.

Atomicity: JSON files write to ``*.tmp`` then ``os.replace()``. JSONL files
get a sibling ``.done`` marker written after the writer closes; resume
treats missing ``.done`` as "must re-run".
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

from . import PIPELINE_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = REPO_ROOT / "data" / "cache" / "jobs"


def hash_image_bytes(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def job_id_for(video_url: str, ref_image_sha: str, backend_tag: str) -> str:
    """Stable 16-char id for the (video, reference, backend, pipeline) tuple."""
    h = hashlib.sha256()
    h.update(video_url.encode("utf-8"))
    h.update(b"\x00")
    h.update(ref_image_sha.encode("ascii"))
    h.update(b"\x00")
    h.update(backend_tag.encode("ascii"))
    h.update(b"\x00")
    h.update(PIPELINE_VERSION.encode("ascii"))
    return h.hexdigest()[:16]


def cache_dir(job_id: str) -> Path:
    d = JOBS_ROOT / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def has_json(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def has_jsonl(path: Path) -> bool:
    """JSONL artifacts only count as present if their .done sentinel exists."""
    done = path.with_suffix(path.suffix + ".done")
    return path.is_file() and done.is_file()


@contextmanager
def open_jsonl_writer(path: Path) -> Iterator[TextIO]:
    """Stream JSONL writer that drops a ``.done`` sibling on clean exit.

    A crash mid-write leaves no ``.done`` so the resume check forces a re-run.
    """
    done = path.with_suffix(path.suffix + ".done")
    if done.exists():
        done.unlink()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yield f
    os.replace(tmp, path)
    done.touch()


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
