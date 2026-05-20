"""FastAPI layer that wraps the videofind pipeline for arbitrary videos.

Exposes an async brand-detection API: submit a job, watch progress over SSE,
fetch the result. The CLI in ``main.py`` is unaffected.
"""

import os as _os

# Point urllib/ssl at certifi's CA bundle. Python.org's macOS build ships
# without bundled root certs, so the first time easyocr / huggingface tries
# to download a model over HTTPS we hit SSL_CERTIFICATE_VERIFY_FAILED.
# Set before any HTTPS call so subprocesses inherit it too.
try:
    import certifi as _certifi

    _ca = _certifi.where()
    _os.environ.setdefault("SSL_CERT_FILE", _ca)
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
except ImportError:
    pass
