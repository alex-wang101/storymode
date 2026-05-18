"""Pydantic schema for the VLM-generated reference JSON.

min_length=1 on arrays prevents empty alternate_prompts / brand_text_keywords,
which would silently produce zero OWLv2 detections downstream.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VisualAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: str = Field(min_length=1)
    cap: str = Field(min_length=1)
    body: str = Field(min_length=1)
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    color_palette: list[str] = Field(min_length=1)
    scale: str = Field(min_length=1)


class Reference(BaseModel):
    """The reference JSON the VLM is required to emit."""

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(min_length=1)
    primary_detection_prompt: str = Field(min_length=1)
    alternate_prompts: list[str] = Field(min_length=1)
    visual_attributes: VisualAttributes
    hard_negative_objects: list[str] = Field(min_length=1)
    most_important_matching_features: list[str] = Field(min_length=1)
    brand_text_keywords: list[str] = Field(min_length=1)


def reference_json_schema() -> dict:
    """The JSON Schema dict for ``Reference``, suitable for server-side enforcement."""
    return Reference.model_json_schema()
