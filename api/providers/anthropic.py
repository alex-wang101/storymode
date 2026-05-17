"""Anthropic Claude adapter. Schema enforcement via forced tool-use.

Anthropic has no native JSON-mode; the schema-enforcing path is to define
a single tool whose ``input_schema`` is the target schema, then force
``tool_choice`` to that tool. The model's tool-call ``input`` is the
guaranteed-valid JSON object.

The ``api_key`` ``SecretStr`` is unwrapped exactly once, at the SDK
construction site. Provider 4xx/5xx are caught and re-raised as a
generic ``ProviderError`` so the upstream error body -- which can echo
the key in some 401 payloads -- never leaves this function.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

from pydantic import SecretStr

from ..reference_prompt import SYSTEM_PROMPT, user_prompt
from ..reference_schema import reference_json_schema
from .errors import ProviderError

MODEL = "claude-sonnet-4-6"


def _media_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def _call_sync(image_b64: str, media_type: str, hint: str | None, api_key: SecretStr) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key.get_secret_value())
    tool = {
        "name": "emit_reference",
        "description": "Emit the reference JSON describing the product in the image.",
        "input_schema": reference_json_schema(),
    }
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_reference"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt(hint)},
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise ProviderError(f"anthropic upstream error: {type(e).__name__}") from None

    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            return json.dumps(block.input)
    raise ProviderError("anthropic returned no tool_use block")


async def generate(image_path: Path, api_key: SecretStr, hint: str | None) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return await asyncio.to_thread(
        _call_sync, image_b64, _media_type(image_path), hint, api_key
    )
