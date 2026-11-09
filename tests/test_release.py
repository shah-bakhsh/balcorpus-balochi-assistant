import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_release import stage_model


def fixture(tmp_path):
    source = tmp_path / "private-results"
    for purpose in ["commercial", "noncommercial"]:
        folder = source / "models" / (purpose + "-v1") / "adapter"
        folder.mkdir(parents=True)
        for name in ["adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json"]:
            (folder / name).write_bytes(b"fixture:" + name.encode())
        provenance = {"purpose": purpose, "model": {"model_id": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}, "source_licenses": ["CC-BY-4.0"] if purpose == "commercial" else ["CC-BY-4.0", "CC-BY-NC-SA-4.0"]}
        (folder / "experiment_provenance.json").write_text(json.dumps(provenance))
        (folder / "secret.env").write_text("fixture secret must never be exported")
        (folder / "training_corpus.jsonl").write_text("fixture raw corpus must never be exported")
    return source


def test_nc_publication_fails_before_creating_output(tmp_path):
    source = fixture(tmp_path)
    out = tmp_path / "public"
    with pytest.raises(ValueError, match="private"):
        stage_model(source, out, "noncommercial", "public")
    assert not out.exists()


def test_public_model_export_is_an_explicit_allowlist(tmp_path):
    source = fixture(tmp_path)
    out = tmp_path / "public"
    result = stage_model(source, out, "commercial", "public")
    original = source / "models/commercial-v1/adapter/adapter_model.safetensors"
    assert (out / "adapter_model.safetensors").read_bytes() == original.read_bytes()
    assert result["files"]["adapter_model.safetensors"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert not (out / "secret.env").exists()
    assert not (out / "training_corpus.jsonl").exists()
    assert not (out / "experiment_provenance.json").exists()


def test_source_permission_mismatch_is_rejected(tmp_path):
    source = fixture(tmp_path)
    p = source / "models/commercial-v1/adapter/experiment_provenance.json"
    data = json.loads(p.read_text())
    data["source_licenses"] = ["CC-BY-NC-SA-4.0"]
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="license"):
        stage_model(source, tmp_path / "public", "commercial", "public")
