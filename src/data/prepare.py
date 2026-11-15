#!/usr/bin/env python3
"""Normalize text instruction data into chat JSONL for INFINITY AI."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

DOMAINS = {
    "coding", "general", "cybersecurity_defensive",
    "research", "mathematics", "multimodal_text",
}


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    domain = str(row.get("domain", "general")).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")

    if isinstance(row.get("messages"), list):
        messages = row["messages"]
    else:
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("Record needs messages[] or string prompt/response.")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    cleaned = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be an object.")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Initial text-only pipeline requires non-empty text content.")
        cleaned.append({"role": role, "content": content.strip()})

    if not any(m["role"] == "user" for m in cleaned):
        raise ValueError("Conversation must contain a user message.")
    if cleaned[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant response.")

    return {"messages": cleaned, "domain": domain}


def fingerprint(record: dict[str, Any]) -> str:
    payload = json.dumps(record["messages"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("Each line must be a JSON object.")
                rows.append(normalize_record(raw))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--validation-split", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    if not 0.0 < args.validation_split < 0.5:
        raise SystemExit("--validation-split must be between 0 and 0.5")

    records = load_jsonl(args.input)
    if len(records) < 2:
        raise SystemExit("Provide at least two valid examples.")

    unique = {}
    for record in records:
        unique.setdefault(fingerprint(record), record)
    records = list(unique.values())
    if len(records) < 2:
        raise SystemExit("At least two unique examples are required.")

    random.Random(args.seed).shuffle(records)
    val_count = max(1, round(len(records) * args.validation_split))
    validation = records[:val_count]
    train = records[val_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in (("train", train), ("validation", validation)):
        target = args.output_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "input_examples": len(load_jsonl(args.input)),
        "unique_examples": len(records),
        "train_examples": len(train),
        "validation_examples": len(validation),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
