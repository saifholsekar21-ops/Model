# Dataset handling

Place source JSONL files in this folder and run the preparation script.

Each line must be a JSON object with either prompt/response/domain or messages/domain fields. The checked-in raw/examples.jsonl contains six illustrative examples only. It validates the pipeline; it is not a useful training corpus and is not evidence of model quality.

Before using a real dataset:
- Verify the license permits model training and redistribution.
- Keep a source manifest with dataset name, URL, version, license, and collection date.
- Remove secrets, personal information, and confidential code.
- Deduplicate against validation and test sets to reduce leakage.
- Keep evaluation prompts out of training data.
- For cybersecurity, use authorized defensive examples only.
- For multimodal data, preserve modality files and metadata; the current preparation script intentionally accepts text-only records.
