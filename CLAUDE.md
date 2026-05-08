# CLAUDE.md

Project-level context for Claude Code sessions. This file is the canonical
record of where the project is in its plan and what's been decided. Update
the "Current stage" block and the per-stage status markers as work progresses.

## Project at a glance

- **What.** `videofind` — given a video and reference image(s) of an object,
  return time intervals when the object appears.
- **Why.** Personal learning project. Recreates the visual half of a
  content-compliance system that's transcript-only in production. Artifact is
  a blog post + this repo, not a service.
- **Audience.** Technical readers familiar with computer vision and ML
  engineering, plus recruiters reading the blog.
- **Positioning.** Essentially the EGO4D VQ2D task; positioned explicitly
  against that literature, no novelty claim.

## Hard constraints (don't relitigate)

- 16 GB Apple Silicon M4 Pro. PyTorch with MPS backend or CPU. **No CUDA.**
- Single machine, open-source models only.
- Output is interval-level (`start`, `end`, signals) — not per-frame boxes.
- No combined "final confidence" score; signals are reported separately.
- No tracking. Sampling density substitutes for tracking-based propagation.
- No web app or hosted demo; the repo and post are the artifacts.

## Current stage

**Stage 1 — Model comparison.** Just starting.

**Stage 0 is happening asynchronously** (user is sourcing and labeling 15–20
YouTube videos in parallel). Stage 1 work that doesn't depend on the full
labeled set can proceed using a handful of frames the user provides directly.

What "done" looks like for Stage 1:

- Each candidate model has been run in a notebook on the same 5–10 positive
  and 5–10 negative frames.
- Side-by-side outputs (boxes, scores, crops) are saved so the comparison is
  reviewable later.
- A short written decision recording which models go into the pipeline and
  *why* — based on observed behavior on this content, not on reputation.

Don't start writing the model wrappers for production use yet; this stage is
exploratory. Keep it in `notebooks/` until a model is chosen.

## Stage map

Status markers: `[ ]` not started, `[~]` in progress, `[x]` done,
`[async]` user is handling outside the main thread.

### `[async]` Stage 0 — Collect and label eval videos
Source 15–20 videos (mix of short / medium / long), pick a target product per
video, hand-label intervals where the product appears. Annotations go in
`eval/labels.json`. Ground truth that every later stage measures against.

### `[~]` Stage 1 — Model comparison
Pick 5–10 positive and 5–10 negative frames from labeled videos. Run candidate
models in a notebook. Compare outputs side by side. Decide which models go in
the pipeline based on what was actually observed.

Split into two sub-notebooks:

**1a — Object detection** (`notebooks/stage1_model_comparison.py`, written):
- *Image-guided lane:* OWLv2 image-guided, YOLOE visual-prompt, SAM 3 visual.
- *Text-guided lane:* OWLv2 text-guided, YOLO-World v2, YOLOE text, SAM 3
  text, Grounding DINO, OmDet-Turbo, Florence-2.
- Skipped: DetCLIP (no clean pip path), T-Rex2 (gated weights).

**1b — Visual embedding for re-ranking** (separate notebook, not yet written):
- DINOv2 (baseline), DINOv3 (newer; restricted license — flag before using),
  SigLIP 2 (real alternative, may outperform DINOv2 on retrieval-flavored
  tasks), EVA-CLIP / EVA-02. CLIP variants and FashionCLIP are dominated and
  can be skipped.

Deliverable: a short comparison note in `experiments/stage1/` recording which
models are kept, which are dropped, and the reason. No production wrappers
yet — these notebooks are exploratory.

### `[ ]` Stage 2 — Frame extraction
Build the component that takes a video and a list of timestamps and returns
RGB frame arrays. Handles seek-and-decode, color conversion, timestamps past
end-of-video. Foundation for every later stage.

### `[ ]` Stage 3 — Visual detection layer
Frame + reference image -> list of scored detections. Image-guided detection
plus a re-ranking step on the candidate boxes. Per-frame only, no temporal
logic. Wrap as a clean function with explicit signal fields per detection.

### `[ ]` First eval checkpoint
Wire Stage 2 + Stage 3 into a minimal end-to-end pipeline: fixed-rate sampling,
detection on every sampled frame, trivial grouping of consecutive
high-confidence frames into intervals. Run on the full eval set. Record
baseline metrics (precision, recall, mean IoU, runtime per video). This is the
starting line for everything that follows.

### `[ ]` Stage 4 — Smart frame sampling
Replace fixed-rate sampling with scene-detection-based sampling
(PySceneDetect): one frame per shot midpoint, additional samples within long
shots, periodic safety-net rate floored by the minimum appearance duration of
interest.

### `[ ]` Stage 5 — Temporal aggregation with hysteresis
Replace trivial grouping with proper interval logic: admission threshold to
start an interval, commit threshold for the interval to count, gap-based
merging of nearby detections, minimum duration filter, singleton survival for
high-confidence isolated detections. Output structured intervals.

### `[ ]` Second eval checkpoint
Run the full pipeline (Stages 2–5) on the eval set. Compare to the first
checkpoint baseline. Sweep `tau_frame_*`, `tau_interval_*`, `gap_max`,
`d_min`. Document precision-recall curves and the chosen operating point(s).

### `[ ]` Stage 6 — Transcript priority layer (optional)
Ingest auto-caption transcript via yt-dlp. Fuzzy-match a user-provided product
name / brand against transcript spans, tolerating caption errors. Mark matches
(plus a forward extension to absorb pronoun references) as priority. Modify
Stage 4 sampling to be dense in priority spans, sparse elsewhere. If no
transcript is available, the layer is a no-op.

### `[ ]` Third eval checkpoint (ablation)
Run with and without the transcript layer on the same eval set. Document the
recall / runtime tradeoff. Becomes the transcript-ablation section of the
post.

### `[ ]` Stage 7 — EGO4D qualitative run
Run the pipeline on 10–20 clips from EGO4D VQ2D validation. Document where
failure modes shift versus creator content. Qualitative comparison only — not
competing on benchmark metrics.

### `[ ]` Stage 8 — Failure mode catalog
Run final pipeline on full eval set, examine every failure, categorize by
type (tiny products, occlusion, similar competitors, motion blur, etc.),
capture screenshots with brief captions. The section readers screenshot.

### `[ ]` Stage 9 — Repo polish
README quickstart verified on a fresh clone. Sample video and reference image
checked in. `download_data.py` and `reproduce_eval.py` working. Final results
JSON committed under `experiments/`.

### `[ ]` Stage 10 — Blog post
Draft, plots, embedded clips, failure catalog from Stage 8.

## Working conventions

- **One concern per stage.** Don't bundle Stage 4's sampling change with
  Stage 5's aggregation logic in the same change set; the eval checkpoints
  exist to attribute deltas to specific changes.
- **Stage outputs are files.** Each stage reads and writes JSONL keyed by
  video id under `data/cache/<video_id>/`. Decouples stages, makes
  ablations cheap, makes failures inspectable.
- **Configs are the source of truth.** No magic numbers in code. Thresholds
  and sampling parameters live in YAML and are versioned alongside the
  experiment results that used them.
- **Frozen experiments.** Anything reported in the blog post lives in
  `experiments/<run_id>/` with `config.yaml` + `metrics.json`. The post
  references run ids explicitly.
- **Determinism.** Seeds set at process start. Hugging Face model revisions
  pinned by SHA in configs, not by tag.
- **Ablation hygiene.** Tuning and reporting use disjoint video-level
  splits. Don't tune thresholds on the same videos used to report final
  metrics.

## Open questions / unresolved decisions

- **Stage 5 input signals.** User has used phrasing that suggests Stage 5
  aggregates DINOv2 similarity only, but the design doc has it aggregating
  both OWLv2 confidence and DINOv2 similarity. Resolve before writing the
  hysteresis logic — the answer changes what the metric and thresholds
  even mean.
- **Image- vs text-guided detection.** A text descriptor is already being
  generated for Stage 6. Stage 1's comparison should answer whether it's
  worth running a text-conditioned detector channel in parallel with the
  image-guided one (recall safety net), or whether image-guided alone is
  sufficient on this content.
- **Model revision pins.** TBD until Stage 1 selects the detectors. Pin
  HF revisions in `configs/` once chosen.
- **Train/test video split.** Will be specified in `eval/labels.json`
  (per-video `split` field) once Stage 0 has enough labeled videos to
  split meaningfully.

## How to update this file

- When a stage starts, change its `[ ]` to `[~]` and update **Current stage**.
- When a stage completes, change to `[x]` and move open questions / decisions
  it answered into a brief note under the stage.
- Don't rewrite history; future-Claude needs the trail of what was decided
  when. Append, don't overwrite.
