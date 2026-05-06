# data/

What's checked into the repo, what's fetched on first run, and what the user
has to provide themselves.

## Layout

```
data/
├── references/         CHECKED IN  — reference object images
├── samples/            CHECKED IN  — small clip used by the quickstart
├── videos/             GITIGNORED — fetched by scripts/download_data.py
├── ego4d/              GITIGNORED — user-provided (license-gated)
└── ego4d_clip_ids.json CHECKED IN  — manifest of which EGO4D clips are used
```

## `references/`

One subdirectory per object, named with a stable slug:

```
references/
└── <object_slug>/
    ├── front.jpg            # canonical reference view
    ├── alt_*.jpg            # optional additional views
    └── README.md            # source URL, license, notes
```

Each `<object_slug>/README.md` records:

- where the image came from (URL or capture method),
- its license / usage terms,
- any cropping or preprocessing applied,
- notes a downstream reader needs (e.g. "logo region only", "matte background").

The pipeline reads every `*.jpg` / `*.png` under an object's folder as a
reference; multi-reference detection keeps the max confidence across queries.

## `samples/`

A short clip and reference image bundled so the quickstart in the root README
runs without network. Keep this small (single-digit MB) and use only content
the project has clear rights to redistribute.

## `videos/`

Fetched from a YouTube URL list by `scripts/download_data.py`. Gitignored
because of size and licensing. The downloader is the single source of truth for
which videos are part of the eval set; `eval/labels.json` keys annotations to
the same video ids the downloader writes.

## `ego4d/` and `ego4d_clip_ids.json`

EGO4D access requires accepting their license at
https://ego4d-data.org/, after which clips can be downloaded via the EGO4D
CLI. `ego4d_clip_ids.json` records which clip ids this project uses; the
clips themselves are not redistributed here.

If `data/ego4d/` is empty, EGO4D-related evaluation steps are skipped rather
than failed.

## Cache

Pipeline-derived intermediates (sampled frames, per-stage JSONL outputs,
detection crops) are written under `data/cache/<video_id>/`. This directory is
gitignored and safe to delete; rerunning the pipeline regenerates it from
`data/videos/` and `data/references/`.
