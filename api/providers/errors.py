"""Sanitized provider error.

Wraps any SDK/transport exception from a cloud VLM so that nothing the
underlying provider returned (which can echo the API key in some 401
payloads) propagates to the FastAPI response.
"""


class ProviderError(Exception):
    pass
