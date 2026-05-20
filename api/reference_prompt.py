"""Shared prompt + example used by all four VLM backends.

Same system message and same user template for Qwen, Anthropic, OpenAI,
and Gemini. Backends that support server-side JSON enforcement plug the
schema in via their SDK; backends that don't (Qwen) rely on the prompt's
schema description plus a parse-and-repair pass in ``reference_gen``.

The prompt leads with a concrete one-shot example because the raw JSON
Schema (with ``$defs`` / ``$ref``) confuses small models like Qwen2-VL-2B,
which latch onto the first nested definition and emit only that block.
A worked example is far more reliable than abstract type definitions.
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


# One-shot example for a *different* product (Liquid Death water can) so the
# model copies the structure, not the contents. Must validate against the
# Reference schema.
_EXAMPLE = {
    "product_name": "Liquid Death Mountain Water tallboy can",
    "primary_detection_prompt": "tall slim matte black aluminum beverage can with white skull artwork and large white Liquid Death wordmark",
    "alternate_prompts": [
        "Liquid Death can",
        "matte black aluminum water can with white skull illustration",
        "tall 16 fl oz black beverage can with white gothic logo",
        "black tallboy can with white death-metal style branding",
    ],
    "visual_attributes": {
        "shape": "tall narrow cylindrical aluminum tallboy can, ~16 fl oz proportions",
        "cap": "standard silver aluminum pop-tab top, matte finish",
        "body": "matte black aluminum body with high-contrast white illustration",
        "label": "front-printed wraparound design with a white stylized skull and large white Liquid Death wordmark",
        "text": "large white text reading 'LIQUID DEATH' across the upper body; smaller text 'MOUNTAIN WATER' below",
        "color_palette": ["matte black", "bright white", "silver top"],
        "scale": "tall slim beverage can, roughly 16 fl oz",
    },
    "hard_negative_objects": [
        "regular silver soda can",
        "white-bodied energy drink can",
        "short stubby beer can",
        "glass water bottle",
        "black energy drink can without white skull artwork",
    ],
    "most_important_matching_features": [
        "matte black aluminum body",
        "large white Liquid Death wordmark",
        "white skull illustration",
        "tall slim tallboy proportions",
    ],
    "brand_text_keywords": [
        "liquid death",
        "mountain water",
        "murder your thirst",
    ],
}


def user_prompt(product_name_hint: str | None) -> str:
    schema = json.dumps(reference_json_schema(), indent=2)
    example = json.dumps(_EXAMPLE, indent=2)
    hint = (
        f"\n\nThe user has indicated the product is: {product_name_hint!r}. "
        "Use this as a hint, but only include text on the product that is "
        "actually visible in the image."
        if product_name_hint
        else ""
    )
    return (
        "Look at the attached reference image and emit a single JSON object "
        "describing that product. The OUTER object must have exactly these "
        "seven top-level keys: product_name, primary_detection_prompt, "
        "alternate_prompts, visual_attributes, hard_negative_objects, "
        "most_important_matching_features, brand_text_keywords. "
        "`visual_attributes` is itself an object with the keys shape, cap, "
        "body, label, text, color_palette, scale. Every array must have at "
        "least one element. Do NOT emit just the visual_attributes block on "
        "its own -- always wrap it inside the full outer object.\n\n"
        "Here is a complete worked example for a DIFFERENT product (a "
        "Liquid Death water can). Copy this structure exactly, but replace "
        "the contents to describe the product in the attached image:\n\n"
        f"{example}\n\n"
        f"{hint}\n\n"
        "Reference schema (for validation; the example above shows the "
        f"required structure):\n{schema}"
    )
