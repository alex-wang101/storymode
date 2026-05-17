"""Shared prompt + example used by all four VLM backends.

Same system message and same user template for Qwen, Anthropic, OpenAI,
and Gemini. Backends that support server-side JSON enforcement plug the
schema in via their SDK; backends that don't (Qwen) rely on the prompt's
schema description plus a parse-and-repair pass in ``reference_gen``.
"""

from __future__ import annotations

import json

from .reference_schema import reference_json_schema


SYSTEM_PROMPT = (
    "You are a visual-recognition prompt engineer. Given a reference image of "
    "a single physical product, output a JSON object that describes the "
    "product in the exact schema requested. Output ONLY the JSON object -- "
    "no prose, no markdown fences, no commentary. Be concrete and visual: "
    "describe shape, color, label layout, and size. Do not invent text that "
    "is not legible on the product. Use the product's actual brand name if "
    "it is visible; otherwise use a short descriptive name."
)


def user_prompt(product_name_hint: str | None) -> str:
    schema = json.dumps(reference_json_schema(), indent=2)
    hint = (
        f"\n\nThe user has indicated the product is: {product_name_hint!r}. "
        "Use this as a hint, but only include text on the product that is "
        "actually visible in the image."
        if product_name_hint
        else ""
    )
    return (
        "Look at the attached reference image and emit a JSON object that "
        "conforms to the following schema. Every array field must contain at "
        "least one element. Do not include any field that is not in the schema."
        f"{hint}\n\n"
        f"Schema:\n{schema}"
    )
