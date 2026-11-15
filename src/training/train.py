#!/usr/bin/env python3
"""Experimental QLoRA SFT for Qwen2.5-Omni Thinker text conversations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("messages"), list):
                    raise ValueError(f"{path}:{line_no}: expected messages list")
                rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def build_dataset(rows: list[dict], processor, max_length: int) -> Dataset:
    samples = []
    for row in rows:
        messages = row["messages"]
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("This first milestone accepts text-only conversations.")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = processor.tokenizer(
            text, truncation=True, max_length=max_length
        )
        samples.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        })
    return Dataset.from_list(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_id = cfg["model"]["name_or_path"]
    train_path = Path(cfg["data"]["train_file"])
    val_path = Path(cfg["data"]["validation_file"])

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. Enable a Kaggle GPU accelerator.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    print("NOTE: This is an experimental 7B multimodal Thinker QLoRA path.")

    if args.preflight:
        print("Preflight passed: CUDA is available. This does not prove model loading or training will fit.")
        return

    if not train_path.exists() or not val_path.exists():
        raise SystemExit("Run src/data/prepare.py first to create train and validation JSONL.")

    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["model"]["load_in_4bit"]),
        bnb_4bit_quant_type=cfg["model"]["quant_type"],
        bnb_4bit_use_double_quant=bool(cfg["model"]["double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(cfg["training"]["gradient_checkpointing"])
    )

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    train_ds = build_dataset(train_rows, processor, int(cfg["data"]["max_seq_length"]))
    val_ds = build_dataset(val_rows, processor, int(cfg["data"]["max_seq_length"]))

    t = cfg["training"]
    output_dir = Path(t["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(t["num_train_epochs"]),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        warmup_ratio=float(t["warmup_ratio"]),
        logging_steps=int(t["logging_steps"]),
        save_steps=int(t["save_steps"]),
        eval_steps=int(t["eval_steps"]),
        save_total_limit=int(t["save_total_limit"]),
        max_grad_norm=float(t["max_grad_norm"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        fp16=bool(t["fp16"]),
        bf16=bool(t["bf16"]),
        optim=t["optim"],
        dataloader_num_workers=int(t["dataloader_num_workers"]),
        report_to=[],
        eval_strategy="steps",
        remove_unused_columns=False,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=processor.tokenizer, mlm=False
        ),
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "processor"))
    print("Training completed. Adapter saved to:", output_dir / "adapter")


if __name__ == "__main__":
    main()
