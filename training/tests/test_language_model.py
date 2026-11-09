"""Random weights and integer/English fixtures test engineering, not Balochi."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from language_model import chunk_tokens, collate_chunks, validation_loss, train_adapter


@pytest.mark.parametrize("major,want", [(None, "float32"), (6, "float32"), (7, "float16"), (8, "bfloat16"), (9, "bfloat16")])
def test_precision_avoids_emulated_bf16_and_slow_pascal_fp16(major, want):
    from language_model import training_precision
    assert training_precision(major) == want


@pytest.mark.parametrize("tokens,block,want", [
    ([10, 11, 12, 13, 14, 15], 4, [[10, 11, 12, 13], [13, 14, 15, 99]]),
    ([10, 11, 12, 13], 4, [[10, 11, 12, 13], [13, 99]]),
    ([10], 4, [[10, 99]]),
])
def test_boundary_context_and_tail_targets_are_not_dropped(tokens, block, want):
    result = chunk_tokens(tokens, 99, block)
    assert result == want
    assert [t for chunk in result for t in chunk[1:]] == tokens[1:] + [99]


def test_single_token_blocks_are_invalid():
    with pytest.raises(ValueError):
        chunk_tokens([1, 2], 99, 1)


def test_eos_labels_survive_when_eos_is_also_padding():
    batch = collate_chunks([[10, 11, 99], [20, 99]], 99)
    assert batch["input_ids"].tolist() == [[10, 11, 99], [20, 99, 99]]
    assert batch["labels"].tolist() == [[10, 11, 99], [20, 99, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_local_qwen3_lora_can_optimize_save_and_reload(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import PeftModel

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64, eos_token_id=63, pad_token_id=63,
    ))
    base = tmp_path / "random-engineering-base"
    model.save_pretrained(base)
    model.config._name_or_path = str(base)
    train = [[1, 2, 3, 63], [5, 6, 7, 8, 63]]
    validation = [[9, 10, 63]]
    # A different heldout token count catches accidentally evaluating train.
    before = validation_loss(model, validation, 63)
    assert before["predicted_tokens"] == 2
    adapted, metrics = train_adapter(model, train, validation, 63, tmp_path / "run", {
        "max_steps": 2, "epochs": 1, "learning_rate": 0.01,
        "gradient_accumulation_steps": 1, "rank": 2, "alpha": 4,
        "gradient_checkpointing": False,
        "base_revision": "b" * 40,
    })
    import json
    saved_config = json.loads((tmp_path / "run/adapter/adapter_config.json").read_text(encoding="utf-8"))
    assert saved_config["revision"] == "b" * 40
    assert metrics["optimizer_steps"] == 2
    assert metrics["before"]["predicted_tokens"] == 2
    assert metrics["after"]["predicted_tokens"] == 2
    assert 0 < metrics["trainable_parameters"] < metrics["total_parameters"]
    assert any(torch.count_nonzero(p).item() for n, p in adapted.named_parameters() if "lora_B" in n)
    adapted.eval()
    batch = collate_chunks(validation, 63)
    with torch.no_grad():
        expected = adapted(**batch).logits
    reloaded = PeftModel.from_pretrained(Qwen3ForCausalLM.from_pretrained(base), tmp_path / "run/adapter")
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(**batch).logits
    assert torch.allclose(actual, expected, atol=1e-6)
