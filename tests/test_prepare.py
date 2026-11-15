import pytest

from src.data.prepare import normalize_record, fingerprint


def test_prompt_response_normalization():
    row = normalize_record({
        "prompt": "Explain recursion",
        "response": "A function calling itself.",
        "domain": "coding",
    })
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][-1]["role"] == "assistant"
    assert row["domain"] == "coding"


def test_reject_unknown_domain():
    with pytest.raises(ValueError):
        normalize_record({
            "prompt": "x", "response": "y", "domain": "unknown"
        })


def test_fingerprint_is_stable():
    row = normalize_record({"prompt": "x", "response": "y"})
    assert fingerprint(row) == fingerprint(row)
