"""Google Gemini adapter. Schema enforcement via response_schema.

The ``api_key`` ``SecretStr`` is unwrapped exactly once, at the SDK
construction site. Provider 4xx/5xx are caught and re-raised as
``ProviderError`` with no key echo.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import SecretStr

from ..reference_prompt import SYSTEM_PROMPT, user_prompt
from ..reference_schema import Reference
from .errors import ProviderError

MODEL = "gemini-2.5-flash"


def _media_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def _call_sync(image_bytes: bytes, media_type: str, hint: str | None, api_key: SecretStr) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key.get_secret_value())
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=media_type)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[image_part, user_prompt(hint)],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=Reference,
            ),
        )
    except Exception as e:
        raise ProviderError(f"google upstream error: {type(e).__name__}") from None

    if not resp.text:
        raise ProviderError("google returned empty content")
    return resp.text


async def generate(image_path: Path, api_key: SecretStr, hint: str | None) -> str:
    image_bytes = image_path.read_bytes()
    return await asyncio.to_thread(
        _call_sync, image_bytes, _media_type(image_path), hint, api_key
    )
