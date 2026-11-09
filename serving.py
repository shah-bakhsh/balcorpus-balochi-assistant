"""Validated inference for the commercial experimental adapter."""
import hashlib
import json
import math
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SYSTEM = "You are an experimental Balochi writing assistant. Follow the requested language and script. If uncertain, say so. Do not invent facts."


def build_messages(message, history):
    if not isinstance(message, str) or not message.strip() or len(message) > 4000:
        raise ValueError("Enter a message of 1 to 4000 characters.")
    if history is None:
        history = []
    if not isinstance(history, list):
        raise ValueError("Invalid conversation history.")
    turns, pending = [], None
    # Bound parsing work even for a history supplied directly through the API.
    for row in history[-64:]:
        if not isinstance(row, dict) or row.get("role") not in {"user", "assistant"}:
            continue
        text = row.get("content")
        if isinstance(text, list) and len(text) <= 32:
            parts = [part.get("text", "") for part in text if isinstance(part, dict) and part.get("type") == "text"]
            if any(not isinstance(part, str) or len(part) > 4000 for part in parts):
                continue
            text = "\n".join(parts)
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            continue
        item = {"role": row["role"], "content": text}
        if row["role"] == "user":
            pending = item
        elif pending is not None:
            turns.append([pending, item])
            pending = None
    return [{"role": "system", "content": SYSTEM}] + [
        item for turn in turns[-3:] for item in turn
    ] + [{"role": "user", "content": message.strip()}]

def prune_session_history(history):
    """Leave two old turns; Gradio appends the current pair after the callback."""
    if isinstance(history, list):
        history[:] = build_messages("pending", history)[1:-1][-4:]


def generate_chat_reply(generate, message, history, response_length):
    """Validate before GPU allocation and update state in Gradio's process."""
    limit = validate_generation_length(response_length)
    messages = build_messages(message, history)
    response = generate(messages[-1]["content"], messages[1:-1], limit)
    prune_session_history(history)
    return response


def fit_messages(messages, count_tokens, budget=1024):
    result = list(messages)
    while count_tokens(result) > budget and len(result) > 2:
        del result[1:3]
    if count_tokens(result) > budget:
        raise ValueError("Please shorten your message.")
    return result

def validate_generation_length(value):
    if type(value) not in {int, float} or not math.isfinite(value) or not float(value).is_integer() or not 16 <= value <= 128:
        raise ValueError("Response length must be an integer between 16 and 128 tokens.")
    return int(value)

def validate_release(folder, expected_adapter_sha256):
    root = Path(folder).resolve()
    metadata = json.loads((root / "release-metadata.json").read_text(encoding="utf-8"))
    if metadata.get("purpose") != "commercial":
        raise ValueError("The public demo requires the commercial adapter.")
    if metadata.get("base_model_id") != BASE_MODEL or metadata.get("base_revision") != BASE_REVISION:
        raise ValueError("Base model or revision differs from this release.")
    files = metadata.get("files", {})
    if not {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}.issubset(files):
        raise ValueError("Incomplete adapter file manifest.")
    if files["adapter_model.safetensors"] != expected_adapter_sha256:
        raise ValueError("Adapter hash differs from the pinned public release.")
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Invalid adapter file path.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Adapter file hash mismatch: " + name)
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != BASE_MODEL or config.get("revision") != BASE_REVISION:
        raise ValueError("Adapter configuration does not pin the expected base.")
    return metadata
