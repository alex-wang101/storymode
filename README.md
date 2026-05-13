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

`python main.py run` is the headline command — it chains transcript
ingest → sponsor detection → frame sampling → OWLv2 text-guided
detection → OCR-primary / NaFlex-fallback brand verification, end to
end, and prints (or writes) a per-region summary.

`python main.py detect-sponsors` runs only the sponsor-detection stage,
useful for inspecting what SponsorBlock-ML finds before any video frame
work happens.

### Install

**You don't need to install anything manually.** The first time you run
either subcommand, `main.py` checks the core imports (`torch`,
`transformers`, `yt_dlp`, `cv2`, `easyocr`, `rapidfuzz`, `PIL`) and, if
any are missing, auto-runs `pip install -r requirements.txt` before
proceeding. Subsequent runs skip the check (sub-millisecond) and start
immediately.

If you'd rather install manually first:

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. `requirements.txt` covers both the CLI and
the Stage 1 notebooks; the notebooks also have their own `%pip install`
cell at the top in case you want to run them in a different environment.

### Run the full pipeline on bundled assets

The repo ships with the transcript used during research (a 21-minute
bacon-curing YouTube video) at `data/references/transcript.json`. You
also need the source video file at `data/videos/zbiotics-bacon.mp4` —
not bundled in git (size / licensing), place it there yourself.

```bash
python main.py run
```

Stages in order:

1. Load the bundled transcript.
2. Load `Xenova/sponsorblock-small` and predict sponsor regions.
3. **Filter to `category == "sponsor"`** — `selfpromo` / `interaction` /
   etc. regions are dropped before any video work.
4. Decode and sample frames inside each sponsor region (OpenCV, ~2 fps,
   ±5 s buffer).
5. Run OWLv2 text-guided detection per frame using the prompts in
   `data/references/zbiotics.json`.
6. For each detection, OCR-primary scoring (EasyOCR + rapidfuzz vs
   `brand_text_keywords`) with NaFlex image-image fallback against
   `data/references/logo.png` when OCR returned no text.
7. Aggregate per region; print summary; optionally write JSON.

Each model is loaded and freed in sequence; peak RAM stays around 5–6 GB
on a 16 GB Mac.

### Run on your own YouTube video

```bash
python main.py run --url https://www.youtube.com/watch?v=<id>
```

`yt-dlp` fetches the auto-caption track. The video file itself must be
present at `data/cache/videos/<id>.mp4` — the current pipeline does not
auto-download the video (only captions). Until that's added, you can
download manually:

```bash
yt-dlp -f mp4 -o "data/cache/videos/%(id)s.%(ext)s" https://www.youtube.com/watch?v=<id>
python main.py run --url https://www.youtube.com/watch?v=<id>
```

### Save the per-region summary to JSON

```bash
python main.py run --output results.json
python main.py run --url <url> --output results.json
```

### Output format

**Stdout** (always printed):

```
Run summary
  3 sponsor region(s) detected
  612 frames sampled
  2,841 OWLv2 detection(s)
  OCR primary, NaFlex fallback

  Region 0  [   87.5s,   173.2s]    123 frames     412 dets
            ocr_hits= 18   max_ocr=0.92   max_naflex=0.78   max_combined=0.92
  Region 1  [  445.0s,   502.8s]     87 frames     215 dets
            ocr_hits=  2   max_ocr=0.31   max_naflex=0.32   max_combined=0.32

  Highest combined brand score: Region 0 (combined=0.92, signal=ocr)
```

**`results.json`** (only with `--output`) — same per-region aggregates,
machine-readable:

```json
{
  "video": {
    "id": "4GBf9ZO2UN8",
    "url": "https://www.youtube.com/watch?v=4GBf9ZO2UN8",
    "title": "Possibly The BEST Bacon EVER!...",
    "duration_seconds": 1250.0
  },
  "sponsor_regions": [
    {
      "region_index": 0,
      "start": 87.5,
      "end": 173.2,
      "category": "sponsor",
      "frames_sampled": 123,
      "owlv2_detections": 412,
      "ocr_hits": 18,
      "max_ocr_score": 0.92,
      "max_naflex_score": 0.78,
      "max_combined_score": 0.92,
      "best_signal": "ocr"
    }
  ]
}
```

Per-detection raw records (every box with its scores) are intentionally
not in the JSON — for the headline answer the per-region aggregate is
enough. The Python API
(`pipeline.brand_detector.score_detections`) returns the full
`BrandDetection` list if a downstream caller needs that detail.

### Stage 2 only (sponsor detection inspection)

`python main.py detect-sponsors` runs just the SponsorBlock-ML pass.
Useful when you want to see *all* categories the model emits (`sponsor`,
`selfpromo`, `interaction`, etc.) before the `run` command's filter
drops the non-`sponsor` ones.

```bash
python main.py detect-sponsors                   # bundled transcript
python main.py detect-sponsors --url <url>       # any YouTube URL
python main.py detect-sponsors --output FILE     # also write JSON
```

Stdout looks like:

```
Sponsored segments (3 found):
   87.5 - 173.2 sec    (sponsor)
  430.0 - 445.0 sec    (selfpromo)
  912.5 - 945.0 sec    (sponsor)
```

This command makes no `category` filter — it shows everything. The
`run` command's filter to `sponsor` only happens downstream of it.

### Notebooks (Stage 1 — model comparison)

The Stage 1 work lives in three Jupyter notebooks under `notebooks/`:

- `OVOD_eval.ipynb` — open-vocabulary object detectors compared on
  hand-picked positive / negative frames. Outputs
  `experiments/stage1/ovod_detections.jsonl`.
- `SSL_EVAL.ipynb` — image-image similarity (DINOv2 / SigLIP 2 / EVA-02)
  re-ranking OVOD output. Kept as a documented baseline; not used in
  production.
- `BRAND_eval.ipynb` — OCR primary, NaFlex image-image vs `logo.png`
  fallback. The production brand-verification recipe.

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
