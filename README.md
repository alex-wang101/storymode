# videofind

Visual object localization in video. Given a video and one or more reference
images of an object, return time intervals when the object appears on screen,
with per-interval evidence.

This is a personal learning project. The artifacts are this repository and an
accompanying blog post — not a production system, not a hosted service. The
intended audience is technical readers familiar with computer vision and ML
engineering.

## Status

- **Stage 1 — model comparison.** Complete. See `notebooks/OVOD_eval.ipynb`
  (open-vocabulary object detectors) and `notebooks/BRAND_eval.ipynb`
  (OCR primary + SigLIP 2 NaFlex fallback against a logo reference). The
  earlier image-image SSL eval (`notebooks/SSL_EVAL.ipynb`) is kept as a
  documented baseline; image-image similarity proved less reliable on
  brand-specific discrimination than the OCR + logo-similarity combo.
- **Stage 2 — sponsor region detection.** In progress. `main.py
  detect-sponsors` uses `Xenova/sponsorblock-small` to identify sponsor
  segments in a YouTube transcript. See the "Replicate the pipeline"
  section below.
- `CLAUDE.md` has the full stage map and what's planned next.

## Problem

Production content-compliance systems verify creator videos against brand-brief
requirements. Spoken requirements ("must say the brand name once") are checkable
from a transcript. Visual requirements ("show the product for at least 10
seconds within the first 5 minutes") are not — they get kicked to human review.

This repo implements the visual half: a pipeline that takes a video plus
reference image(s) of an object and returns time intervals when the object is on
screen, plus the per-interval signals that justify each interval.

## Relationship to prior work

After designing the approach independently, this turned out to be the EGO4D
Visual Queries 2D Localization (VQ2D) task. The accompanying blog post positions
this work explicitly against that literature; there is no novelty claim. Domain
differences from EGO4D:

|              | EGO4D                       | This project                       |
| ------------ | --------------------------- | ---------------------------------- |
| Viewpoint    | Egocentric (first-person)   | Third-person creator content       |
| Cuts         | None                        | Frequent, often sub-second         |
| Transcript   | Usually absent              | Usually present (yt-dlp captions)  |
| Output unit  | Per-frame bounding boxes    | Time intervals (start, end)        |

Two pipeline choices follow from those differences and have no analogue in the
egocentric VQ2D literature:

- **Scene-detection-aware sampling** (Stage 2). Cuts are exploited rather than
  ignored; samples are placed mid-shot and at a within-shot rate.
- **Transcript-priority regions** (Stage 1). Transcript-text similarity to a
  reference descriptor raises the sampling rate in likely regions.

## Hardware and runtime expectations

- Single machine: Apple Silicon (M-series) MacBook with 16 GB unified memory.
- Backend: PyTorch with the MPS device, or CPU fallback. **No CUDA path.**
- Disk: ~20 GB headroom for fetched videos and intermediate caches.
- Indicative runtime on M4 Pro (to be measured): ~`TODO` minutes per
  10-minute video for the full pipeline.

## Pipeline

Five stages. Each writes a serialized intermediate to disk, so any stage can be
re-run, swapped, or ablated without re-running upstream stages.

1. **Stage 1 — Transcript-priority regions.** Sentence-embedding similarity
   between transcript spans and a text descriptor of the reference. Spans above
   threshold are flagged as priority.
2. **Stage 2 — Frame sampling.** Inside priority regions: ~2 fps. Outside:
   PySceneDetect shot boundaries plus mid-shot frames plus a periodic safety-net
   rate floored by the minimum object-appearance duration of interest.
3. **Stage 3 — Detection.** OWLv2 image-guided detection on each sampled frame
   with the reference image(s) as the query. Per frame: timestamp, bounding
   box, OWLv2 confidence.
4. **Stage 4 — Re-ranking.** DINOv2 embedding of each detected crop, cosine
   similarity to the reference embedding. Adds `reference_similarity` to each
   detection.
5. **Stage 5 — Temporal aggregation.** Hysteresis-style grouping of frame-level
   detections into intervals, with admission and commit thresholds, gap closure,
   and a duration floor. Produces the final intervals.

Output schema, one record per interval:

```json
{
  "start": 12.40,
  "end": 18.20,
  "peak_owl": 0.71,
  "peak_sim": 0.83,
  "best_frame_timestamp": 14.80,
  "best_box": [320, 180, 540, 360],
  "transcript_evidence": "..."
}
```

Signals are reported separately. There is no combined "final confidence" score —
this is a deliberate choice for honesty and debuggability.

## Replicate the pipeline

Stage 2 (sponsor region detection) has a working end-to-end CLI. The other
stages live in notebooks under `notebooks/` and are explored interactively.

### Install

```bash
# Python 3.10+ recommended.
pip install torch transformers yt-dlp easyocr rapidfuzz pillow
```

That's the minimal set for `main.py detect-sponsors` to run. For the
Stage 1 notebooks (OVOD, SSL, BRAND evals) also install `ultralytics`,
`open_clip_torch`, `matplotlib`, and `numpy<2` — there's a `%pip install`
cell at the top of each notebook with the exact list.

### Run sponsor detection on the bundled research transcript

The repo ships with the transcript used during research (a 21-minute
bacon-curing YouTube video) at `data/references/transcript.json`. Run
sponsor detection against it with no arguments:

```bash
python main.py detect-sponsors
```

Expected output (timestamps will vary based on what the model finds):

```
Loading bundled transcript: .../data/references/transcript.json
  N transcript items, 1250.0s, 'Possibly The BEST Bacon EVER!...'
Loading sponsorblock model...
Detecting sponsor segments...

Sponsored segments (N found):
   <start> -  <end> sec    (sponsor)
   ...
```

To eyeball whether the predictions are right, cross-reference the
timestamps against the transcript items in `data/references/transcript.json`
— the text near a predicted segment should read like a sponsor read.

### Run on your own YouTube video

Pass any YouTube URL with `--url`. `yt-dlp` fetches the auto-caption track
and formats it into the same dict shape as the bundled transcript:

```bash
python main.py detect-sponsors --url https://www.youtube.com/watch?v=<id>
```

### Save results to JSON

```bash
python main.py detect-sponsors --output results.json
python main.py detect-sponsors --url <url> --output results.json
```

The JSON contains only the timestamps + categories of detected sponsor
segments, plus the video metadata. The full per-segment text and
classifier confidence are returned by the Python API
(`pipeline.sponsor_detect.find_sponsor_intervals`) but trimmed from the
CLI output for simplicity.

### Notebooks (Stage 1 — model comparison)

The Stage 1 work lives in three Jupyter notebooks under `notebooks/`:

- `OVOD_eval.ipynb` — open-vocabulary object detectors compared on hand-
  picked positive / negative frames. Outputs `experiments/stage1/ovod_detections.jsonl`.
- `SSL_EVAL.ipynb` — image-image similarity (DINOv2 / SigLIP 2 / EVA-02)
  re-ranking OVOD output. Kept as a documented baseline; not used in
  production.
- `BRAND_eval.ipynb` — OCR (EasyOCR + rapidfuzz against brand keywords)
  primary, NaFlex image-image vs `logo.png` fallback. The production
  brand-verification path.

Open in Jupyter or VSCode, run the install cell once, then run the
remaining cells top-to-bottom.

## Evaluation methodology

- **Eval set.** 15–20 hand-labeled YouTube videos, mixing short (<2 min),
  medium (2–15 min), and long (>20 min). Annotations live in `eval/labels.json`.
- **Splits.** Tuning and reporting use disjoint video-level splits. Thresholds
  are not selected on the videos used to report final metrics.
- **EGO4D subset.** A small subset of the EGO4D VQ2D validation set
  (10–20 clips) is used for **qualitative** domain-shift comparison only. This
  project does not compete with published baselines on their leaderboard
  metrics.
- **Metrics.** Recall, precision, mean temporal IoU on matched intervals,
  wall-clock per video.
- **Threshold handling.** Frame-level admission thresholds are fixed loose;
  `gap_max` and `d_min` are fixed by motivated defaults; the two
  interval-level thresholds are swept jointly and reported as a Pareto curve
  rather than a single chosen point.
- **Ablations.** With/without transcript priority (Stage 1); with/without
  DINOv2 re-ranking (Stage 4); scene-detection sampling vs fixed-rate.

## Repository layout

```
videofind/
├── README.md                       # this file
├── CLAUDE.md                       # stage map and current progress
├── main.py                         # single CLI entry point with subcommands
├── pipeline/
│   ├── __init__.py
│   ├── transcript_ingest.py        # load_from_file + fetch_from_youtube (yt-dlp)
│   └── sponsor_detect.py           # load_models + find_sponsor_intervals
├── data/
│   ├── references/
│   │   ├── zbiotics.json           # brand metadata: prompts, keywords, features
│   │   ├── zbiotics.png            # full product reference photo
│   │   ├── logo.png                # brand-logo-only reference (used by NaFlex)
│   │   └── transcript.json         # bundled research transcript (bacon video)
│   └── frames/                     # hand-picked positive/negative frames for Stage 1
│       ├── positive_easy/          # bottle clearly visible
│       ├── positive_hard/          # bottle partial/occluded
│       ├── negative_easy/          # no bottles in frame
│       └── negative_hard/          # bottles that aren't ZBiotics
├── notebooks/
│   ├── OVOD_eval.ipynb             # Stage 1a: detector comparison
│   ├── SSL_EVAL.ipynb              # Stage 1b: image-image similarity baseline (deprecated)
│   └── BRAND_eval.ipynb            # Stage 1c: OCR + NaFlex brand verification
├── experiments/stage1/             # eval outputs from the notebooks (JSONL + summaries + figures)
└── scripts/                        # ad-hoc helpers (currently empty)
```

### Conventions

- **Bundled vs fetched data.** `data/references/` and `data/samples/` are checked
  in so the quickstart works without network. `data/videos/` and `data/ego4d/`
  are gitignored and reconstructed from `scripts/download_data.py` and the
  EGO4D clip manifest.
- **Stage outputs are files.** Each stage reads and writes JSONL on disk, keyed
  by video id. This decouples stages, makes ablations cheap (don't re-run
  Stage 3 to swap Stage 4), and makes failures inspectable.
- **Frozen experiments.** Anything reported in the blog post lives under
  `experiments/<run_id>/` with its config and metrics. The run id is referenced
  explicitly in the post.
- **Configs are the source of truth.** No magic numbers in code; thresholds and
  sampling parameters live in YAML and are version-controlled with the run.
- **Determinism.** Seeds set at process start. Model revisions pinned by
  Hugging Face revision SHA in `configs/`, not by tag.

## Models

| Stage                        | Purpose                                    | HF id                                   | License        |
| ---------------------------- | ------------------------------------------ | --------------------------------------- | -------------- |
| OVOD                         | Open-vocab detection (text-guided)         | `google/owlv2-base-patch16-ensemble`    | Apache 2.0     |
| Brand text                   | OCR (EasyOCR; ResNet/CRNN under the hood)  | bundled with `easyocr` pip package      | Apache 2.0     |
| Brand logo                   | Image-image similarity (aspect-preserving) | `google/siglip2-base-patch16-naflex`    | Apache 2.0     |
| Sponsor region (Stage 2)     | Sponsor-segment extraction from transcript | `Xenova/sponsorblock-small`             | CC BY-NC-SA 4.0 (training data) |
| Sponsor region (Stage 2)     | Optional segment-category refinement       | `Xenova/sponsorblock-classifier`        | CC BY-NC-SA 4.0 |

Earlier eval (`SSL_EVAL.ipynb`) compared `facebook/dinov2-base`,
`facebook/dinov3-vitb16-pretrain-lvd1689m`, `google/siglip2-base-patch16-256`,
and an EVA-02 CLIP variant via `open_clip` for image-image re-ranking. The
SigLIP 2 NaFlex (logo-only) approach replaced these in the production
brand-verification path.

## What's deliberately out of scope

- LLM agents for sampling decisions. Deterministic similarity is sufficient
  here; agents add nondeterminism without adding capability.
- A combined evidence score across signals. Reporting separately is more
  honest and easier to debug.
- Bounding-box-area filtering for "clearly displayed" semantics. The task
  is "where does the object appear", not "is it prominently displayed".
- Web app / hosted demo. The artifacts are the blog post and this repo.
- SAM-based region matching as the primary detector. Considered, but too
  expensive for 16 GB unified memory.
- Local Whisper transcription. yt-dlp auto-captions are used to keep one
  fewer model in memory.
- Tracking. The literature uses detect-then-track; this project is
  sampling-based because the output is intervals, not per-frame boxes.

## License

This project is **non-commercial only**. The Stage 2 sponsor-region
detection uses `Xenova/sponsorblock-small`, a T5 model trained on the
SponsorBlock community database, which is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
That non-commercial restriction is inherited by anything that depends
on those weights. If you fork this repo and want to commercialize, you
would need to re-train the sponsor-region detector on permissively
licensed data.

The repository's own code is otherwise unencumbered (project-license
TBD; intended permissive for the code itself, with the non-commercial
restriction applying only to the SponsorBlock-dependent path).

Attribution: SponsorBlock by Ajay Ramachandran
(https://sponsor.ajay.app/); model by Xenova
(https://github.com/xenova/sponsorblock-ml).

## Citation

If this project's pipeline or evaluation is referenced, a link to the blog
post and to this repository is sufficient. The accompanying literature this
work is positioned against:

```
@inproceedings{ego4d_vq2d,
  title  = {Ego4D: Around the World in 3,000 Hours of Egocentric Video},
  author = {Grauman, Kristen and others},
  year   = {2022}
}
```

`TODO` — also cite OWLv2, DINOv2, PySceneDetect, sentence-transformers.
