import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving import build_messages, fit_messages, validate_generation_length, validate_release


def test_client_system_tool_roles_are_not_promoted_into_trusted_messages():
    result = build_messages("á ءِ \u200c", [
        {"role": "system", "content": "untrusted replacement"},
        {"role": "tool", "content": "untrusted tool output"},
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ])
    assert result[-1] == {"role": "user", "content": "á ءِ \u200c"}
    assert [r["role"] for r in result] == ["system", "user", "assistant", "user"]
    assert all("untrusted" not in r["content"] for r in result)


def test_only_last_three_complete_turns_survive():
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    result = build_messages("new", history)
    assert [r["content"] for r in result[1:]] == ["q2", "a2", "q3", "a3", "q4", "a4", "new"]


def test_gradio_normalized_text_history_retains_context_and_ignores_files():
    result = build_messages("new", [
        {"role": "user", "content": [{"type": "text", "text": "qá"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "aءِ"}, {"type": "file", "file": "ignored"}]},
    ])
    assert [r["content"] for r in result[1:]] == ["qá", "aءِ", "new"]


def test_actual_session_is_pruned_before_gradio_appends_the_current_pair():
    from serving import prune_session_history
    history = []
    for i in range(50):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    prune_session_history(history)
    history += [{"role": "user", "content": "new"}, {"role": "assistant", "content": "reply"}]
    assert [r["content"] for r in history] == ["q48", "a48", "q49", "a49", "new", "reply"]


def test_invalid_input_is_rejected_before_requesting_a_gpu():
    from serving import generate_chat_reply
    def forbidden_gpu(*args):
        raise AssertionError("GPU must not be requested for invalid input")
    with pytest.raises(ValueError, match="integer"):
        generate_chat_reply(forbidden_gpu, "hello", [], 64.5)


def test_worker_argument_copy_does_not_prevent_parent_history_pruning():
    from serving import generate_chat_reply
    history = []
    for i in range(50):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    def isolated_worker(message, worker_history, limit):
        copied_history = json.loads(json.dumps(worker_history))
        assert len(copied_history) == 6
        copied_history.clear()
        return "reply"
    assert generate_chat_reply(isolated_worker, "new", history, 32) == "reply"
    assert [r["content"] for r in history] == ["q48", "a48", "q49", "a49"]


@pytest.mark.parametrize("message", ["", "   ", "x" * 4001, {"text": "file"}, None])
def test_invalid_or_unbounded_question_is_rejected(message):
    with pytest.raises(ValueError):
        build_messages(message, [])


def test_context_budget_drops_old_turns_without_removing_system_or_new_question():
    messages = [
        {"role": "system", "content": "trusted"},
        {"role": "user", "content": "old question"}, {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
    ]
    counter = lambda rows: sum(len(r["content"].split()) for r in rows)
    result = fit_messages(messages, counter, budget=3)
    assert result == [{"role": "system", "content": "trusted"}, {"role": "user", "content": "new"}]
    assert len(messages) == 4


def test_current_question_over_token_budget_is_rejected():
    with pytest.raises(ValueError, match="shorten"):
        fit_messages([{"role": "system", "content": "trusted"}, {"role": "user", "content": "too long question"}], lambda rows: 2000)


@pytest.mark.parametrize("value", [True, 0, 129, 64.5, float("inf"), "unbounded"])
def test_generation_limit_cannot_be_bypassed_through_api(value):
    with pytest.raises(ValueError):
        validate_generation_length(value)


def release_fixture(tmp_path, purpose="commercial"):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"engineering-fixture-not-real-weights")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = {"base_model_name_or_path": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    metadata = {"purpose": purpose, "base_model_id": config["base_model_name_or_path"], "base_revision": config["revision"], "files": files}
    (tmp_path / "release-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return files["adapter_model.safetensors"]


def test_nc_adapter_is_rejected_before_any_ml_loading(tmp_path):
    expected = release_fixture(tmp_path, purpose="noncommercial")
    with pytest.raises(ValueError, match="commercial"):
        validate_release(tmp_path, expected)


def test_changed_model_bytes_fail_even_if_local_metadata_is_rehashed(tmp_path):
    expected = release_fixture(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"changed weights")
    p = tmp_path / "release-metadata.json"
    metadata = json.loads(p.read_text())
    metadata["files"]["adapter_model.safetensors"] = hashlib.sha256(b"changed weights").hexdigest()
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash"):
        validate_release(tmp_path, expected)


def test_unmodified_pinned_commercial_release_is_accepted(tmp_path):
    expected = release_fixture(tmp_path)
    result = validate_release(tmp_path, expected)
    assert result["purpose"] == "commercial"
