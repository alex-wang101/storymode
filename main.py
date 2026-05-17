"""Single entry point for the videofind pipeline.

Subcommands:
    python main.py detect-sponsors                    # stage 2 only (sponsor regions)
    python main.py detect-text                        # "where is the brand NAME visible?"
    python main.py detect-object                      # "where is the brand OBJECT visible?"

All command flags:
    --url <youtube>     fetch transcript via yt-dlp (default: bundled transcript)
    --output FILE       write the JSON result to FILE (default: a sensible path
                        under experiments/stage2/)
    --test              (detect-text/detect-object only) also dump cropped PNGs
                        of every positive frame's winning box for visual review

On first run, if any core dependency is missing, the script auto-installs
from requirements.txt before continuing.
"""

import os

# Set BEFORE importing torch so unsupported MPS ops fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Bundled-asset pairs. Each pair is (transcript JSON, video file). Same
# brand and logo for both -- only the source video differs.
BACON_TRANSCRIPT_PATH = REPO_ROOT / "data" / "references" / "transcript_bacon.json"
BACON_VIDEO_PATH = REPO_ROOT / "data" / "videos" / "zbiotics-bacon.mp4"
PICKLES_TRANSCRIPT_PATH = REPO_ROOT / "data" / "references" / "transcript_pickles.json"
PICKLES_VIDEO_PATH = REPO_ROOT / "data" / "videos" / "zbiotics-pickles.mp4"

LOGO_PATH = REPO_ROOT / "data" / "references" / "logo.png"
BRAND_JSON_PATH = REPO_ROOT / "data" / "references" / "zbiotics.json"
VIDEO_CACHE_DIR = REPO_ROOT / "data" / "cache" / "videos"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# Import names (not pip names) for the critical CLI dependencies.
# Used by ensure_dependencies() to decide whether to auto-install.
_CORE_IMPORTS = ["torch", "transformers", "yt_dlp", "cv2", "easyocr", "rapidfuzz", "PIL"]

# Make sibling packages importable when running this file directly
sys.path.insert(0, str(REPO_ROOT))


def ensure_dependencies():
    """Auto-install requirements.txt if any core dependency is missing.

    Runs at startup so first-time users don't have to manually pip-install
    before invoking the CLI. After install, the script's own imports proceed
    normally inside the command function.
    """
    missing = [m for m in _CORE_IMPORTS if importlib.util.find_spec(m) is None]
    if not missing:
        return
    if not REQUIREMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing imports {missing} and {REQUIREMENTS_PATH} not found"
        )
    print(
        f"Missing imports {missing}. Installing from {REQUIREMENTS_PATH.name}...",
        file=sys.stderr,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)]
    )


def cmd_detect_sponsors(args):
    """Detect sponsor segments in a transcript (bundled file or YouTube URL)."""
    # Lazy imports so `--help` doesn't pay for transformers/torch startup.
    from pipeline.sponsor_detect import find_sponsor_intervals, load_models
    from pipeline.transcript_ingest import fetch_from_youtube, load_from_file

    if args.url:
        print(f"Fetching transcript via yt-dlp: {args.url}", file=sys.stderr)
        transcript = fetch_from_youtube(args.url)
    else:
        transcript_path, _ = _resolve_inputs(args)
        print(f"Loading bundled transcript: {transcript_path}", file=sys.stderr)
        transcript = load_from_file(transcript_path)

    n_items = len(transcript.get("items", []))
    duration = transcript.get("duration_seconds") or 0.0
    title = transcript.get("title", "") or "(no title)"
    print(f"  {n_items} transcript items, {duration:.1f}s, {title!r}", file=sys.stderr)

    print("Loading sponsorblock model...", file=sys.stderr)
    model, tokenizer, classifier = load_models()
    if classifier is None:
        print("  classifier unavailable; raw extractor output only", file=sys.stderr)

    print("Detecting sponsor segments...", file=sys.stderr)
    intervals = find_sponsor_intervals(
        transcript, model=model, tokenizer=tokenizer, classifier=classifier
    )

    print()
    print(f"Sponsored segments ({len(intervals)} found):")
    if not intervals:
        print("  (none)")
    for itv in intervals:
        print(f"  {itv['start']:>7.1f}  -  {itv['end']:>7.1f} sec    ({itv['category']})")

    if args.output:
        payload = {
            "video_id": transcript.get("video_id"),
            "url": transcript.get("url"),
            "title": transcript.get("title"),
            "duration_seconds": transcript.get("duration_seconds"),
            "sponsored_intervals": [
                {"start": itv["start"], "end": itv["end"], "category": itv["category"]}
                for itv in intervals
            ],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(intervals)} intervals to {args.output}", file=sys.stderr)


def _free_memory():
    """Release tensors and clear the MPS cache between stages."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _load_brand_metadata():
    """Read ``zbiotics.json``. Returns ``(text_queries, brand_keywords)``."""
    with open(BRAND_JSON_PATH) as f:
        meta = json.load(f)
    text_queries = [meta["primary_detection_prompt"]] + list(meta.get("alternate_prompts", []))
    brand_keywords = list(meta.get("brand_text_keywords", []))
    return text_queries, brand_keywords


def _resolve_inputs(args):
    """Pick the (transcript_path_or_None, video_path) pair from the source flags.

    Returns ``(transcript_path, video_path)`` where ``transcript_path`` is
    ``None`` for URL mode (caller fetches via yt-dlp instead). Bacon is the
    default if no source flag is given.
    """
    if getattr(args, "url", None):
        return None, None  # caller handles URL mode separately
    if getattr(args, "pickles", False):
        return PICKLES_TRANSCRIPT_PATH, PICKLES_VIDEO_PATH
    return BACON_TRANSCRIPT_PATH, BACON_VIDEO_PATH


def _resolve_video_path(args, transcript):
    """Return the local video file path for the current run mode."""
    if getattr(args, "url", None):
        video_id = transcript.get("video_id")
        if not video_id:
            sys.exit("URL mode requires a video_id in the transcript; yt-dlp should have provided it.")
        return VIDEO_CACHE_DIR / f"{video_id}.mp4"
    if getattr(args, "pickles", False):
        return PICKLES_VIDEO_PATH
    return BACON_VIDEO_PATH


def _run_pipeline_up_to_brand(args):
    """Run transcript -> sponsor -> frame sample -> OWLv2 -> brand scoring.

    Shared upstream pipeline for ``detect-text`` and ``detect-object``.
    Returns ``(transcript, sponsors, frames, brand_dets)`` or ``None`` if
    no sponsor regions were detected.
    """
    from pipeline import brand_detector, frame_sampler, object_detector, sponsor_detect
    from pipeline.transcript_ingest import fetch_from_youtube, load_from_file

    # 1. Transcript
    if getattr(args, "url", None):
        print(f"Fetching transcript via yt-dlp: {args.url}", file=sys.stderr)
        transcript = fetch_from_youtube(args.url)
    else:
        transcript_path, _ = _resolve_inputs(args)
        print(f"Loading bundled transcript: {transcript_path}", file=sys.stderr)
        transcript = load_from_file(transcript_path)
    n_items = len(transcript.get("items", []))
    duration = transcript.get("duration_seconds") or 0.0
    title = transcript.get("title", "") or "(no title)"
    print(f"  {n_items} items, {duration:.1f}s, {title!r}", file=sys.stderr)

    # 2. Video file (must exist locally)
    video_path = _resolve_video_path(args, transcript)
    if not video_path.exists():
        sys.exit(
            f"video file not found: {video_path}\n"
            f"place the source video at this path and re-run."
        )
    print(f"  video: {video_path}", file=sys.stderr)

    # 3. Brand metadata (prompts + OCR keywords)
    text_queries, brand_keywords = _load_brand_metadata()

    # 4. Sponsor detection (T5)
    print("Detecting sponsor regions...", file=sys.stderr)
    sponsor_model, sponsor_tok, sponsor_classifier = sponsor_detect.load_models()
    all_regions = sponsor_detect.find_sponsor_intervals(
        transcript,
        model=sponsor_model,
        tokenizer=sponsor_tok,
        classifier=sponsor_classifier,
    )
    del sponsor_model, sponsor_tok, sponsor_classifier
    _free_memory()

    # Filter to category == "sponsor" only. SponsorBlock-ML also emits
    # `selfpromo`, `interaction`, and other categories, but the production
    # pipeline only cares about paid sponsor reads.
    sponsors = [r for r in all_regions if r.get("category") == "sponsor"]
    dropped = len(all_regions) - len(sponsors)
    if dropped:
        dropped_cats = sorted({r.get("category") for r in all_regions
                               if r.get("category") != "sponsor"})
        print(
            f"  filtered out {dropped} non-sponsor region(s) "
            f"({', '.join(dropped_cats)})",
            file=sys.stderr,
        )

    if not sponsors:
        print("No sponsor regions detected; nothing to analyze.")
        return None
    print(f"  {len(sponsors)} sponsor region(s)", file=sys.stderr)

    # 5. Frame sampling
    print("Sampling frames within sponsor regions...", file=sys.stderr)
    frames = frame_sampler.sample_frames(video_path, sponsors)
    print(f"  {len(frames)} frames sampled", file=sys.stderr)

    # 6. Object detection (OWLv2)
    print("Running OWLv2 text-guided detection...", file=sys.stderr)
    owl_model, owl_proc = object_detector.load_owlv2()
    objects = object_detector.detect_objects(
        frames, model=owl_model, processor=owl_proc, text_queries=text_queries,
    )
    del owl_model, owl_proc
    _free_memory()
    print(f"  {len(objects)} object detection(s)", file=sys.stderr)

    # 7. Brand scoring (OCR + NaFlex, two independent signals)
    print("Scoring detections (OCR + NaFlex)...", file=sys.stderr)
    ocr_reader, naflex_model, naflex_proc = brand_detector.load_brand_models()
    logo_emb = brand_detector.embed_logo(naflex_model, naflex_proc, LOGO_PATH)
    brand_dets = brand_detector.score_detections(
        objects, frames,
        ocr_reader=ocr_reader,
        naflex_model=naflex_model,
        naflex_processor=naflex_proc,
        logo_emb=logo_emb,
        brand_keywords=brand_keywords,
    )
    del ocr_reader, naflex_model, naflex_proc, logo_emb
    _free_memory()

    return transcript, sponsors, frames, brand_dets


def _save_crops(frame_bests, frames, output_dir, score_attr):
    """Save the cropped PNG of every positive frame's winning box.

    Filenames sort chronologically per region and include the score so a
    folder listing tells the story at a glance:
    ``region0_t00389.20_0.92.png``.
    """
    from PIL import Image

    frame_lookup = {(round(sf.timestamp, 6), sf.region_index): sf for sf in frames}
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale PNGs from previous runs so the folder reflects only the
    # current run. Only touches *.png so unrelated files (e.g. README) stay.
    for stale in output_dir.glob("*.png"):
        stale.unlink()

    saved = 0
    for (region_idx, ts), d in frame_bests.items():
        sf = frame_lookup.get((round(ts, 6), region_idx))
        if sf is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in d.box)
        x0 = max(0, x0)
        y0 = max(0, y0)
        h, w = sf.image.shape[:2]
        x1 = min(x1, w)
        y1 = min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = sf.image[y0:y1, x0:x1]
        score = getattr(d, score_attr)
        name = f"region{region_idx}_t{ts:08.2f}_{score:.2f}.png"
        Image.fromarray(crop).save(output_dir / name)
        saved += 1
    return saved


def _print_intervals(label, intervals):
    """Print a compact interval summary."""
    print()
    print(f"{label}: {len(intervals)} interval(s) found")
    for iv in intervals:
        score_key = next(k for k in iv if k.startswith("max_"))
        line = (
            f"  [{iv['start']:7.1f}s, {iv['end']:7.1f}s]  "
            f"duration={iv['duration_seconds']:5.1f}s  "
            f"frames={iv['frame_count']:3d}  "
            f"{score_key}={iv[score_key]:.2f}  "
            f"region={iv['region_index']}"
        )
        if "best_text" in iv and iv["best_text"]:
            line += f"  text={iv['best_text']!r}"
        print(line)


def _default_output_path(transcript, suffix):
    """Build the default output JSON path under experiments/stage2/."""
    video_id = transcript.get("video_id") or "run"
    return REPO_ROOT / "experiments" / "stage2" / f"{video_id}_{suffix}.json"


def _detect_signal(args, *, score_attr, label, default_threshold, suffix):
    """Shared body of cmd_detect_text and cmd_detect_object.

    Runs the upstream pipeline, builds intervals from the chosen signal,
    prints them, writes JSON, and optionally saves crops.
    """
    from pipeline.interval_builder import build_intervals

    result = _run_pipeline_up_to_brand(args)
    if result is None:
        return
    transcript, sponsors, frames, brand_dets = result

    threshold = args.score_threshold if args.score_threshold is not None else default_threshold
    intervals, frame_bests = build_intervals(
        brand_dets,
        score_attr=score_attr,
        score_threshold=threshold,
        max_gap_seconds=args.max_gap,
        min_frames=args.min_frames,
        include_text=(score_attr == "ocr_score"),
    )

    _print_intervals(label, intervals)

    payload = {
        "video": {
            "id": transcript.get("video_id"),
            "url": transcript.get("url"),
            "title": transcript.get("title"),
            "duration_seconds": transcript.get("duration_seconds"),
        },
        "params": {
            "score_attr": score_attr,
            "score_threshold": threshold,
            "max_gap_seconds": args.max_gap,
            "min_frames": args.min_frames,
        },
        "intervals": intervals,
    }
    output_path = args.output or _default_output_path(transcript, suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(intervals)} interval(s) to {output_path}", file=sys.stderr)

    if args.test:
        video_id = transcript.get("video_id") or "run"
        crops_dir = REPO_ROOT / "experiments" / "stage2" / f"{video_id}_{suffix}_crops"
        saved = _save_crops(frame_bests, frames, crops_dir, score_attr)
        print(f"Saved {saved} crop(s) to {crops_dir}", file=sys.stderr)

    # Drop the big in-memory buffers explicitly so this function is safe to
    # import and call from a longer-running script.
    del transcript, sponsors, frames, brand_dets, frame_bests
    _free_memory()


def cmd_detect_text(args):
    """OCR-only: time intervals where the brand NAME was consistently visible."""
    _detect_signal(
        args,
        score_attr="ocr_score",
        label="Brand-text intervals",
        default_threshold=0.7,
        suffix="text_intervals",
    )


def cmd_detect_object(args):
    """NaFlex-only: time intervals where the brand OBJECT was consistently visible."""
    _detect_signal(
        args,
        score_attr="naflex_score",
        label="Brand-object intervals",
        default_threshold=0.6,
        suffix="object_intervals",
    )


def build_parser():
    """Build the top-level argparse parser with one subparser per command."""
    parser = argparse.ArgumentParser(
        prog="videofind",
        description="Open-source brand-detection pipeline for sponsored creator videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_detect_args(p, default_threshold):
        src = p.add_mutually_exclusive_group()
        src.add_argument(
            "--url",
            default=None,
            help="YouTube URL to fetch the transcript via yt-dlp.",
        )
        src.add_argument(
            "--bacon",
            action="store_true",
            help="Use the bundled bacon-curing video and transcript (default).",
        )
        src.add_argument(
            "--pickles",
            action="store_true",
            help="Use the bundled pickleback video and transcript.",
        )
        p.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Write the intervals JSON to this path (default: experiments/stage2/<id>_<suffix>.json).",
        )
        p.add_argument(
            "--test",
            action="store_true",
            help=(
                "Save cropped PNGs of every positive frame's winning box to "
                "experiments/stage2/<id>_<suffix>_crops/ for visual review."
            ),
        )
        p.add_argument(
            "--score-threshold",
            dest="score_threshold",
            type=float,
            default=None,
            help=f"Minimum signal score for a frame to count as positive (default: {default_threshold}).",
        )
        p.add_argument(
            "--max-gap",
            dest="max_gap",
            type=float,
            default=2.0,
            help="Max seconds between adjacent positive frames in one interval (default: 2.0).",
        )
        p.add_argument(
            "--min-frames",
            dest="min_frames",
            type=int,
            default=2,
            help="Minimum frames required per interval; smaller groups are dropped (default: 2).",
        )

    p_text = subparsers.add_parser(
        "detect-text",
        help="OCR-only pipeline: time intervals where the brand NAME was consistently visible.",
    )
    _add_detect_args(p_text, 0.7)
    p_text.set_defaults(func=cmd_detect_text)

    p_object = subparsers.add_parser(
        "detect-object",
        help="NaFlex-only pipeline: time intervals where the brand OBJECT was consistently visible.",
    )
    _add_detect_args(p_object, 0.6)
    p_object.set_defaults(func=cmd_detect_object)

    p_sponsors = subparsers.add_parser(
        "detect-sponsors",
        help="Stage 2 only: detect sponsor segments from a transcript.",
    )
    src_sp = p_sponsors.add_mutually_exclusive_group()
    src_sp.add_argument(
        "--url",
        default=None,
        help="YouTube URL to fetch the transcript via yt-dlp.",
    )
    src_sp.add_argument(
        "--bacon",
        action="store_true",
        help="Use the bundled bacon-curing transcript (default).",
    )
    src_sp.add_argument(
        "--pickles",
        action="store_true",
        help="Use the bundled pickleback transcript.",
    )
    p_sponsors.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a simplified JSON to this path (default: stdout-only).",
    )
    p_sponsors.set_defaults(func=cmd_detect_sponsors)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    ensure_dependencies()
    args.func(args)


if __name__ == "__main__":
    main()
