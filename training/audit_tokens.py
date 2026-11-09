"""Count actual tokenizer tokens; downloads tokenizer JSON only, never weights."""
import argparse
import json
from pathlib import Path
import urllib.request

from language_model import chunk_tokens
from prepare import ROOT, SPLITS, load_prepared, read_json, sha256, write_json


def audit(model_alias, cache):
    from tokenizers import Tokenizer
    model = read_json(ROOT / "config/models.json")[model_alias]
    path = Path(cache) / model["revision"] / "tokenizer.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{model['model_id']}/resolve/{model['revision']}/tokenizer.json"
        request = urllib.request.Request(url, headers={"User-Agent": "BalAssistant-tokenizer-audit/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError("Unexpectedly large tokenizer file")
        Tokenizer.from_str(payload.decode("utf-8"))
        path.write_bytes(payload)
    tokenizer = Tokenizer.from_file(str(path))
    receipt = read_json(ROOT / "config/corpus.json")
    block = read_json(ROOT / "config/training.json")["block_size"]
    result = {"model": model, "tokenizer_sha256": sha256(path), "block_size": block,
              "token_count_definition": "original text encoded without special tokens; EOS and boundary overlap reported separately",
              "balochi_fluency": "NOT_EVALUATED_tokenization_is_not_language_quality", "purposes": {}}
    for purpose in ("commercial", "noncommercial"):
        rows = load_prepared(ROOT / "prepared" / purpose, purpose, receipt)
        result["purposes"][purpose] = {}
        for split in SPLITS:
            totals = {"records": len(rows[split]), "text_tokens": 0, "document_eos_tokens": 0,
                      "chunk_input_tokens_with_overlap": 0, "predicted_next_token_targets": 0, "chunks": 0}
            for row in rows[split]:
                ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                if not ids:
                    raise ValueError("Tokenizer produced an empty document")
                chunks = chunk_tokens(ids, model["eos_token_id"], block)
                totals["text_tokens"] += len(ids)
                totals["document_eos_tokens"] += 1
                totals["chunks"] += len(chunks)
                totals["chunk_input_tokens_with_overlap"] += sum(len(c) for c in chunks)
                totals["predicted_next_token_targets"] += sum(len(c) - 1 for c in chunks)
            result["purposes"][purpose][split] = totals
    report = ROOT / "reports" / f"tokens-{model_alias}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pilot", "small"), default="pilot")
    parser.add_argument("--cache", type=Path, default=ROOT.parent / ".cache/adaptation-tokenizers")
    args = parser.parse_args()
    print(json.dumps(audit(args.model, args.cache), indent=2))
