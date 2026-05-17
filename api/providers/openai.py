"""OpenAI GPT-4o adapter. Schema enforcement via response_format json_schema.

The ``api_key`` ``SecretStr`` is unwrapped exactly once, at the SDK
construction site. Provider 4xx/5xx are caught and re-raised as
``ProviderError`` with no key echo.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from pydantic import SecretStr

from ..reference_prompt import SYSTEM_PROMPT, user_prompt
from ..reference_schema import reference_json_schema
from .errors import ProviderError

MODEL = "gpt-4o-2024-11-20"


def _media_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def _call_sync(image_b64: str, media_type: str, hint: str | None, api_key: SecretStr) -> str:
    import openai

    client = openai.OpenAI(api_key=api_key.get_secret_value())
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt(hint)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "reference",
                    "schema": reference_json_schema(),
                    "strict": False,
                },
            },
        )
    except openai.APIError as e:
        raise ProviderError(f"openai upstream error: {type(e).__name__}") from None

    content = resp.choices[0].message.content
    if not content:
        raise ProviderError("openai returned empty content")
    return content


async def generate(image_path: Path, api_key: SecretStr, hint: str | None) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return await asyncio.to_thread(
        _call_sync, image_b64, _media_type(image_path), hint, api_key
    )
