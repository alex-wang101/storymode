"""Qwen2-VL-2B-Instruct-GPTQ-Int8 inference as a one-shot subprocess.

Run as a subprocess so the MPS allocator is fully released on exit
(torch.mps.empty_cache() alone isn't reliable enough for GPTQ models).
"""

from __future__ import annotations

# Force CPU fallback for any MPS-unsupported ops *before* importing torch.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
from pathlib import Path

from .reference_prompt import SYSTEM_PROMPT, user_prompt


MODEL_REVISION = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8"


def main() -> int:
    parser = argparse.ArgumentParser(prog="qwen-subprocess")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hint", default="", help="Optional product name hint.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    # Heavy imports are inside main so --help is cheap and so import errors
    # surface as a clean non-zero exit rather than at module load.
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_REVISION,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_REVISION)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(args.image)},
                {"type": "text", "text": user_prompt(args.hint or None)},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(output_text)
    os.replace(tmp, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
