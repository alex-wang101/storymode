"""Sanitized provider error — prevents upstream error bodies (which can echo the API key) from propagating."""


class ProviderError(Exception):
    pass
