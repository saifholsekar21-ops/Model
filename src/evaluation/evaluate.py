#!/usr/bin/env python3
"""Small generation smoke evaluation for a trained INFINITY AI adapter."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompt", default="Write a Python function that checks whether a number is prime. Explain edge cases.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for this evaluation script.")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["model"]["name_or_path"]
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(processor.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
