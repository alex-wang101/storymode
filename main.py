"""Single entry point for the videofind pipeline.

Usage:
    python main.py detect-sponsors                    # research reproduction (bundled transcript)
    python main.py detect-sponsors --url <youtube>    # run on your own YouTube video
    python main.py detect-sponsors --output FILE      # also write JSON to FILE

On first run, if any core dependency is missing, the script auto-installs
from requirements.txt before continuing.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPT_PATH = REPO_ROOT / "data" / "references" / "transcript.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# Import names (not pip names) for the critical CLI dependencies.
# Used by ensure_dependencies() to decide whether to auto-install.
_CORE_IMPORTS = ["torch", "transformers", "yt_dlp"]

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
        print(f"Loading bundled transcript: {DEFAULT_TRANSCRIPT_PATH}", file=sys.stderr)
        transcript = load_from_file(DEFAULT_TRANSCRIPT_PATH)

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


def build_parser():
    """Build the top-level argparse parser with one subparser per command."""
    parser = argparse.ArgumentParser(
        prog="videofind",
        description="Open-source brand-detection pipeline for sponsored creator videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sponsors = subparsers.add_parser(
        "detect-sponsors",
        help="Detect sponsor segments from a transcript (file or YouTube URL).",
    )
    p_sponsors.add_argument(
        "--url",
        default=None,
        help="YouTube URL to fetch the transcript via yt-dlp (default: use bundled transcript).",
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
