"""Build explicit model-only exports, never the surrounding private corpus."""
import hashlib
import json
from pathlib import Path
import shutil

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = (
    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "merges.txt", "vocab.json",
)


def stage_model(source, output, purpose, visibility):
    if purpose not in {"commercial", "noncommercial"} or visibility not in {"public", "private"}:
        raise ValueError("Unknown purpose or visibility.")
    if purpose == "noncommercial" and visibility != "private":
        raise ValueError("The research adapter must remain private.")
    source, output = Path(source), Path(output)
    adapter = source / "models" / (purpose + "-v1") / "adapter"
    provenance = json.loads((adapter / "experiment_provenance.json").read_text(encoding="utf-8"))
    if provenance["purpose"] != purpose:
        raise ValueError("Source purpose mismatch.")
    licenses = set(provenance["source_licenses"])
    if purpose == "commercial" and licenses != {"CC-BY-4.0"}:
        raise ValueError("Unapproved source license in public commercial release.")
    model = provenance["model"]
    if model["model_id"] != BASE_MODEL or model["revision"] != BASE_REVISION:
        raise ValueError("Unexpected base model.")
    chosen = [name for name in FILES if (adapter / name).is_file()]
    if not set(FILES[:4]).issubset(chosen):
        raise ValueError("Incomplete trained adapter.")
    if (source / "manifest.json").exists():
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        hashes = {r["path"]: r["sha256"] for r in manifest["files"]}
        for name in chosen + ["experiment_provenance.json"]:
            path = adapter / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.relative_to(source).as_posix()]:
                raise ValueError("Source artifact hash mismatch.")
    output.mkdir(parents=True, exist_ok=False)
    for name in chosen:
        shutil.copyfile(adapter / name, output / name)
    metadata = {
        "status": "experimental_continued_pretraining_NOT_native_verified",
        "purpose": purpose, "visibility": visibility,
        "base_model_id": BASE_MODEL, "base_revision": BASE_REVISION,
        "source_licenses": sorted(licenses),
        "native_reviewed_training_records": provenance.get("native_reviewed_training_records", 0),
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in chosen},
    }
    if (source / "verification.json").exists():
        measured = json.loads((source / "verification.json").read_text(encoding="utf-8"))["purposes"][purpose]
        metadata["training_words"] = measured["train_word_occurrences"]
        metadata["optimizer_steps"] = measured["training"]["optimizer_steps"]
        metadata["validation"] = {key: measured["training"][key] for key in ("before", "after")}
        metadata["reload_verified"] = measured["reload_matches_saved_metrics"]
    (output / "release-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
