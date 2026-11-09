"""Load only the pinned public adapter. No remote Python code is executed."""
import json
from pathlib import Path
import threading

from serving import BASE_MODEL, BASE_REVISION, build_messages, fit_messages, validate_generation_length, validate_release


def download_release():
    from huggingface_hub import snapshot_download

    config = json.loads(Path(__file__).with_name("release.json").read_text(encoding="utf-8"))
    # Hub's normal snapshot directory contains links to sibling blobs. local_dir
    # materializes regular files so the strict release path check stays valid.
    folder = snapshot_download(
        config["model_id"], revision=config["revision"], token=False,
        local_dir=Path.home() / ".cache" / "balcorpus" / config["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt"],
    )
    validate_release(folder, config["adapter_sha256"])
    return folder


class InferenceEngine:
    def __init__(self, device="cpu"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        folder = download_release()
        self.tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=False)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False,
            torch_dtype=torch.float32, low_cpu_mem_usage=True,
            attn_implementation="sdpa", token=False,
        )
        self.model = PeftModel.from_pretrained(base, folder, is_trainable=False).to(device).eval()
        self.device = device
        self.lock = threading.Lock()

    def render(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def reply(self, message, history=None, max_new_tokens=64):
        import torch

        limit = validate_generation_length(max_new_tokens)
        messages = fit_messages(
            build_messages(message, history),
            lambda rows: len(self.tokenizer.encode(self.render(rows), add_special_tokens=False)),
        )
        inputs = self.tokenizer(self.render(messages), add_special_tokens=False, return_tensors="pt").to(self.device)
        with self.lock, torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=limit, do_sample=False, repetition_penalty=1.1,
                max_time=20, pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("The model returned an empty response. Please try another short prompt.")
        return text
