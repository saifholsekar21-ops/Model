# INFINITY AI 1.0

A research and engineering project to fine-tune a unified multimodal foundation model for coding, general assistance, defensive cybersecurity, research, mathematics, and multimodal understanding.

## Important scope

The initial foundation uses Qwen/Qwen2.5-Omni-7B. It is a pretrained multimodal model family supporting text, image, audio, and video. This repository does **not** train a foundation model from scratch.

The first training milestone is text-only supervised fine-tuning of the model's Thinker component with QLoRA. The pretrained multimodal architecture is retained, but this first run does not teach new image/audio skills. Multimodal fine-tuning, speech generation, and full-model deployment are separate milestones and must be validated independently.

A 7B multimodal model is demanding. A Kaggle T4 (commonly 16 GB VRAM) is not guaranteed to fit this training workload. Run the preflight and one-step smoke test first. Reduce sequence length or LoRA scope only through explicit configuration changes. Do not assume a completed training run until the adapter, logs, and evaluation output exist.

## Repository layout

- configs/train.yaml — reproducible training configuration
- data/raw/ — user-supplied source datasets (not committed)
- data/processed/ — normalized JSONL datasets (not committed)
- src/data/prepare.py — validate, normalize, deduplicate, and split examples
- src/training/train.py — QLoRA supervised fine-tuning
- src/evaluation/evaluate.py — smoke evaluation and generation
- src/inference/chat.py — load the base Thinker and trained adapter
- tests/ — unit tests for data preparation

## Dataset format

Each JSONL record may be either:

    {"prompt":"Explain a Python generator.","response":"A generator ...","domain":"coding"}

or:

    {"messages":[{"role":"user","content":"..."},
                 {"role":"assistant","content":"..."}],
     "domain":"reasoning"}

Accepted domains: coding, general, cybersecurity_defensive, research, mathematics, multimodal_text. For the initial text-only training run, content must be text. Do not put secrets, credentials, private source code, personal data, or unlicensed datasets into training data.

## Kaggle quick start

1. Create a Kaggle notebook with GPU enabled.
2. Clone this repository and change into its directory.
3. Install dependencies:

       pip install -r requirements.txt

4. Run the data pipeline:

       python src/data/prepare.py --input data/raw/examples.jsonl --output-dir data/processed

5. Run preflight:

       python src/training/train.py --config configs/train.yaml --preflight

6. Run a one-step smoke test:

       python src/training/train.py --config configs/train.yaml --max-steps 1

7. Only after the smoke test succeeds, launch the configured run:

       python src/training/train.py --config configs/train.yaml

8. Evaluate the adapter:

       python src/evaluation/evaluate.py --config configs/train.yaml --adapter outputs/infinity-ai-7b

The model weights and datasets are intentionally excluded from Git. Save checkpoints to Kaggle working storage and export only the adapter to a controlled Hugging Face repository or private storage.

## Quality gates

- Validate dataset schema and duplicate rate.
- Confirm model class and PEFT target modules on the actual runtime.
- Complete a one-step training smoke test without out-of-memory errors.
- Compare base and adapter outputs on held-out examples.
- Run domain-specific coding, math, safety, and multimodal tests.
- Publish benchmark results, limitations, base model attribution, and dataset licenses.

## License and attribution

The base model's license and use conditions apply to any derivative. Review the Qwen model card and every dataset license before training or redistribution. This repository's source-code license does not override upstream model or dataset terms.
