"""FastAPI layer that wraps the videofind pipeline for arbitrary videos.

Exposes two endpoints (cloud / local) that run the full pipeline against a
user-supplied video URL + reference image, with per-stage caching so an
interrupted job resumes cheaply. The CLI in ``main.py`` is unaffected.
"""

# Bump when any cached-stage producer changes (model revision, prompt,
# extraction logic). Old cache dirs become unreachable on bump, which is
# the point -- silent staleness is worse than re-running a stage.
PIPELINE_VERSION = "1"
