"""Optional CUDA LoRA continued-pretraining experiment, not chat SFT."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

from language_model import chunk_tokens, train_adapter, training_precision
from prepare import ROOT, load_prepared, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purpose", choices=("commercial", "noncommercial"), required=True)
    parser.add_argument("--model", choices=("pilot", "small"), default="small")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--experimental", action="store_true", help="Run an experiment on explicitly unreviewed text candidates")
    args = parser.parse_args()
    if not args.experimental:
        parser.error("Native review is pending. Use --experimental only for an unreviewed research experiment.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty --output directory")
    receipt = read_json(ROOT / "config/corpus.json")
    prepared = args.prepared or ROOT / "prepared" / args.purpose
    rows = load_prepared(prepared, args.purpose, receipt)
    metadata = read_json(ROOT / "config/models.json")[args.model]
    options = read_json(ROOT / "config/training.json")
    options["base_revision"] = metadata["revision"]
    if args.max_steps is not None:
        options["max_steps"] = args.max_steps

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        parser.error("CUDA GPU is unavailable. No pretrained weights have been requested.")
    if torch.cuda.device_count() != 1:
        parser.error("This starter expects exactly one visible CUDA GPU. Set CUDA_VISIBLE_DEVICES before running.")
    precision = training_precision(torch.cuda.get_device_capability(0)[0])
    if precision == "float32" and args.model == "pilot":
        parser.error("Use --model small for the Pascal FP32 starter. The 4B configuration is not validated on this host.")
    set_seed(options["seed"])
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_id"], revision=metadata["revision"], trust_remote_code=False)
    if tokenizer.eos_token_id != metadata["eos_token_id"]:
        raise ValueError("Pinned tokenizer EOS mismatch")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Test rows are validated by the loader but never tokenized, evaluated or trained here.
    chunks = {}
    for split in ("train", "validation"):
        chunks[split] = []
        for row in rows[split]:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            chunks[split].extend(chunk_tokens(ids, tokenizer.eos_token_id, options["block_size"]))
    dtype = getattr(torch, precision)
    model = AutoModelForCausalLM.from_pretrained(
        metadata["model_id"], revision=metadata["revision"], trust_remote_code=False,
        torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa",
    )
    _, metrics = train_adapter(model, chunks["train"], chunks["validation"], tokenizer.pad_token_id, args.output, options)
    tokenizer.save_pretrained(args.output / "adapter")
    run = {
        "status": "UNVALIDATED_experimental_continued_pretraining_NOT_instruction_tuned",
        "purpose": args.purpose, "model": metadata, "options": options, "precision": precision,
        "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
        "corpus_receipt": receipt, "metrics": metrics,
        "source_licenses": read_json(prepared / "manifest.json")["source_licenses"],
        "test_used_for_optimization_or_evaluation": False,
        "native_reviewed_training_records": sum(r["human_reviewed"] is True for r in rows["train"]),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory},
        "environment": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate"]}},
        "publication": "NOT_uploaded_source_terms_and_native_evaluation_remain_applicable",
    }
    write_json(args.output / "run.json", run)
    # Purpose/provenance follows the adapter when it is copied separately.
    write_json(args.output / "adapter" / "experiment_provenance.json", run)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
