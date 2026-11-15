#!/usr/bin/env python3
"""Interactive text chat using the INFINITY AI Thinker adapter."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. For CPU inference, use a separately exported compatible build.")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.base)
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.base, quantization_config=quant, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()
    history = []

    print("INFINITY AI — type /exit to quit")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        inputs = processor.apply_chat_template(
            history, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        answer = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print("\nINFINITY:", answer)
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
