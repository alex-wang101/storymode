# videofind

Visual object localization in video. Given a video and one or more reference
images of an object, return time intervals when the object appears on screen,
with per-interval evidence.

This is a personal learning project. The artifacts are this repository and an
accompanying blog post — not a production system, not a hosted service. The
intended audience is technical readers familiar with computer vision and ML
engineering.

## Status

Stage 1 — model comparison. Stage 0 (eval-set labeling) is running in
parallel. See [`CLAUDE.md`](./CLAUDE.md) for the canonical stage map and
current progress; that file is updated as stages complete.

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

## Quickstart

Reproduces the pipeline on a 30-second clip bundled in `data/samples/`.

```bash
# 1. Install dependencies into a uv-managed virtualenv
uv sync

# 2. Run the pipeline on the bundled sample
uv run python eval/run_pipeline.py \
  --video data/samples/sample_video.mp4 \
  --reference data/samples/sample_reference.jpg \
  --output results/sample.json
```

Expected runtime on M4 Pro: ~`TODO` seconds. First run will download model
weights from Hugging Face into the local cache.

## Reproducing the reported numbers

Ground-truth interval annotations are checked in at `eval/labels.json`. Eval
videos are not bundled (size and licensing); a downloader fetches them by URL.

```bash
# 1. Fetch the YouTube eval videos (yt-dlp; requires ffmpeg on PATH)
uv run python scripts/download_data.py

# 2. (Optional) place EGO4D clips listed in data/ego4d_clip_ids.json
#    into data/ego4d/. EGO4D access requires accepting their license;
#    see data/README.md.

# 3. Run the pipeline across the eval set
uv run python eval/run_pipeline.py --eval --config configs/<run_id>.yaml

# 4. Compute and print the reported metrics
uv run python eval/reproduce_eval.py --run experiments/<run_id>
```

Every reported number in the blog post corresponds to a frozen run under
`experiments/<run_id>/` containing the exact `config.yaml` used. Changing any
threshold, sampling parameter, or model revision moves the numbers; the config
is the source of truth.

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
├── README.md                    # this file
├── pyproject.toml               # pinned dependencies (uv-managed)
├── uv.lock                      # locked resolution; checked in for reproducibility
├── data/
│   ├── README.md                # what's bundled, what's downloaded, licensing
│   ├── references/              # CHECKED IN — reference object images
│   │   └── <object_name>/
│   │       ├── front.jpg
│   │       └── README.md        # source, license, notes for this object
│   ├── samples/                 # CHECKED IN — quickstart demo clip + reference
│   │   ├── sample_video.mp4
│   │   └── sample_reference.jpg
│   ├── videos/                  # GITIGNORED — fetched by scripts/download_data.py
│   ├── ego4d/                   # GITIGNORED — user-provided, license-gated
│   └── ego4d_clip_ids.json      # CHECKED IN — manifest of EGO4D clips used
├── eval/
│   ├── labels.json              # CHECKED IN — ground-truth interval annotations
│   ├── run_pipeline.py          # CLI: pipeline on one video or the eval set
│   └── reproduce_eval.py        # CLI: regenerates the reported metrics
├── configs/                     # YAML configs per experiment
├── src/videofind/
│   ├── stage1_transcript/
│   ├── stage2_sampling/
│   ├── stage3_detection/
│   ├── stage4_rerank/
│   ├── stage5_aggregation/
│   ├── eval/                    # metric implementations, sweep harness
│   └── utils/
├── scripts/
│   └── download_data.py
├── notebooks/                   # exploratory work; not part of the reproduction path
├── experiments/                 # one directory per recorded run
│   └── <run_id>/
│       ├── config.yaml          # frozen config
│       ├── metrics.json         # reported numbers
│       └── README.md            # short description of the run
├── tests/
└── results/                     # GITIGNORED — ad-hoc local outputs
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

| Stage | Model | HF id | Pinned revision |
| ----- | ----- | ----- | --------------- |
| 1     | sentence-transformers (small embedder) | `TODO` | `TODO` |
| 3     | OWLv2 (image-guided detector)          | `TODO` | `TODO` |
| 4     | DINOv2 (visual encoder)                | `TODO` | `TODO` |

Revisions are filled in once Phase 1 (hello-world end-to-end) selects them.

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

`TODO` — pick before publishing.

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
