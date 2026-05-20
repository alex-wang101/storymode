"""Routes VLM reference-JSON generation to local Qwen or a cloud provider.

Both paths return a validated Reference dict. Raises VlmJsonError on parse/validation
failure; the api_key is forwarded as SecretStr and unwrapped only at the SDK call site.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from pydantic import SecretStr, ValidationError

from .providers import anthropic as anthropic_adapter
from .providers import google as google_adapter
from .providers import openai as openai_adapter
from .providers.errors import ProviderError
from .reference_schema import Reference


BACKEND_LOCAL = "local-qwen2vl2b-gptq-int8"
BACKEND_ANTHROPIC = "cloud-anthropic"
BACKEND_OPENAI = "cloud-openai"
BACKEND_GOOGLE = "cloud-google"

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class VlmJsonError(Exception):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        # ``raw`` is the model output, never the api_key. Callers may surface it.
        self.raw = raw


class UnsupportedKeyPrefix(Exception):
    pass


def backend_for_key(api_key: SecretStr) -> str:
    key = api_key.get_secret_value()
    # Order matters: ``sk-ant-`` also matches ``sk-``.
    if key.startswith("sk-ant-"):
        return BACKEND_ANTHROPIC
    if key.startswith("AIza"):
        return BACKEND_GOOGLE
    if key.startswith("sk-"):
        return BACKEND_OPENAI
    raise UnsupportedKeyPrefix(
        "api_key must start with sk-ant- (Anthropic), AIza (Google), or sk- (OpenAI)"
    )


def _strip_and_parse(raw: str) -> dict:
    stripped = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise VlmJsonError(f"VLM output is not valid JSON: {e}", raw=stripped) from None


def _validate(parsed: dict, raw: str) -> dict:
    try:
        return Reference.model_validate(parsed).model_dump()
    except ValidationError as e:
        raise VlmJsonError(f"VLM output failed schema validation: {e}", raw=raw) from None


async def generate_reference_local(
    image_path: Path,
    hint: str | None,
    raw_output_path: Path,
) -> dict:
    """Run Qwen in a child process; parse + validate its output.

    If a sidecar ``<image>.json`` exists next to the reference image and parses
    against the Reference schema, use it directly. The local Qwen2-VL-2B model
    is too small to reliably emit the nested schema; curated sidecars are the
    intended local-mode workflow for known products.
    """
    sidecar = image_path.with_suffix(".json")
    if sidecar.is_file():
        raw = sidecar.read_text()
        parsed = _strip_and_parse(raw)
        return _validate(parsed, raw)

    args = [
        sys.executable, "-m", "api.qwen_subprocess",
        "--image", str(image_path),
        "--output", str(raw_output_path),
    ]
    if hint:
        args += ["--hint", hint]
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise VlmJsonError(
            f"qwen subprocess exited {proc.returncode}: {stderr.decode(errors='replace')}",
            raw="",
        )
    raw = raw_output_path.read_text()
    parsed = _strip_and_parse(raw)
    return _validate(parsed, raw)


async def generate_reference_cloud(
    image_path: Path,
    api_key: SecretStr,
    hint: str | None,
) -> tuple[str, dict]:
    """Returns (backend_tag, validated_reference_dict)."""
    backend = backend_for_key(api_key)
    if backend == BACKEND_ANTHROPIC:
        raw = await anthropic_adapter.generate(image_path, api_key, hint)
    elif backend == BACKEND_OPENAI:
        raw = await openai_adapter.generate(image_path, api_key, hint)
    elif backend == BACKEND_GOOGLE:
        raw = await google_adapter.generate(image_path, api_key, hint)
    else:
        raise UnsupportedKeyPrefix(backend)
    parsed = _strip_and_parse(raw)
    return backend, _validate(parsed, raw)
