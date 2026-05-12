"""Detect sponsor segments in a transcript.

Two paths:
    (no flag)              load data/references/transcript.json (research reproduction).
    --url <youtube_url>    fetch transcript via yt-dlp (run on your own video).

Optional --output writes a JSON file alongside the stdout summary.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.sponsor_detect import find_sponsor_intervals, load_models
from pipeline.transcript_ingest import fetch_from_youtube, load_from_file


DEFAULT_TRANSCRIPT_PATH = REPO_ROOT / "data" / "references" / "transcript.json"


def main():
    parser = argparse.ArgumentParser(description="Detect sponsor segments in a transcript")
    parser.add_argument(
        "--url",
        default=None,
        help="YouTube URL to fetch the transcript via yt-dlp (overrides the bundled transcript)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write a simplified JSON to this path (default: stdout-only)",
    )
    args = parser.parse_args()

    if args.url:
        print(f"Fetching transcript via yt-dlp: {args.url}", file=sys.stderr)
        transcript = fetch_from_youtube(args.url)
    else:
        print(f"Loading bundled transcript: {DEFAULT_TRANSCRIPT_PATH}", file=sys.stderr)
        transcript = load_from_file(DEFAULT_TRANSCRIPT_PATH)

    n_items = len(transcript.get("items", []))
    duration = transcript.get("duration_seconds") or 0.0
    title = transcript.get("title", "") or "(no title)"
    print(
        f"  {n_items} transcript items, {duration:.1f}s, {title!r}",
        file=sys.stderr,
    )

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
        print(
            f"  {itv['start']:>7.1f}  -  {itv['end']:>7.1f} sec    ({itv['category']})"
        )

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


if __name__ == "__main__":
    main()
