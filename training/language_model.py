"""Engineering helpers for a causal-language-model adaptation experiment."""
import math


def training_precision(cuda_major):
    if cuda_major is None or cuda_major < 7:
        return "float32"
    return "bfloat16" if cuda_major >= 8 else "float16"


def chunk_tokens(token_ids, eos_id, block_size):
    if block_size < 2:
        raise ValueError("A causal chunk needs at least two tokens")
    if not token_ids:
        return []
    sequence = list(token_ids) + [eos_id]
    # One context token overlaps so every target after the document's initial
    # context token is predicted exactly once, including the final EOS.
    return [sequence[start:start + block_size] for start in range(0, len(sequence) - 1, block_size - 1)]

def collate_chunks(chunks, pad_id):
    import torch
    width = max(len(row) for row in chunks)
    return {
        "input_ids": torch.tensor([row + [pad_id] * (width - len(row)) for row in chunks], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in chunks], dtype=torch.long),
        "labels": torch.tensor([row + [-100] * (width - len(row)) for row in chunks], dtype=torch.long),
    }

def validation_loss(model, chunks, pad_id):
    import torch
    if not chunks:
        raise ValueError("Validation has no chunks")
    device = next(model.parameters()).device
    was_training = model.training
    total, targets = 0.0, 0
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            batch = {k: v.to(device) for k, v in collate_chunks([chunk], pad_id).items()}
            count = int((batch["labels"][:, 1:] != -100).sum())
            loss = float(model(**batch).loss)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            total += loss * count
            targets += count
    model.train(was_training)
    mean = total / targets
    return {"token_weighted_nll": mean, "perplexity": math.exp(mean) if mean < 700 else None, "predicted_tokens": targets}

def train_adapter(model, train_chunks, validation_chunks, pad_id, output, options):
    from pathlib import Path
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    if not train_chunks or not validation_chunks:
        raise ValueError("Training and validation both require nonempty chunks")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory for each experiment")
    set_seed(options.get("seed", 42))
    before = validation_loss(model, validation_chunks, pad_id)
    model.config.use_cache = False
    adapted = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=options["rank"], lora_alpha=options["alpha"], revision=options.get("base_revision"),
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none",
    ))
    checkpointing = options.get("gradient_checkpointing", False)
    if checkpointing:
        adapted.enable_input_require_grads()
    cuda = next(adapted.parameters()).device.type == "cuda"
    major = torch.cuda.get_device_capability(next(adapted.parameters()).device)[0] if cuda else None
    precision = training_precision(major)
    args = TrainingArguments(
        output_dir=str(output / "trainer"), num_train_epochs=options["epochs"],
        max_steps=options.get("max_steps", -1), learning_rate=options["learning_rate"],
        per_device_train_batch_size=1, gradient_accumulation_steps=options["gradient_accumulation_steps"],
        gradient_checkpointing=checkpointing, bf16=precision == "bfloat16", fp16=precision == "float16",
        use_cpu=not cuda, optim="adamw_torch", warmup_ratio=0.05, weight_decay=0.01,
        logging_steps=1, save_strategy="no", eval_strategy="no", report_to=[],
        push_to_hub=False, dataloader_pin_memory=cuda, seed=options.get("seed", 42),
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=adapted, args=args,
        train_dataset=[{"input_ids": chunk} for chunk in train_chunks],
        data_collator=lambda examples: collate_chunks([example["input_ids"] for example in examples], pad_id),
    )
    trainer.train()
    after = validation_loss(adapted, validation_chunks, pad_id)
    adapter_dir = output / "adapter"
    # Vocabulary is unchanged and the base embeddings stay frozen.
    adapted.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
    return adapted, {
        "before": before, "after": after, "optimizer_steps": trainer.state.global_step,
        "trainable_parameters": sum(p.numel() for p in adapted.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in adapted.parameters()),
        "native_quality_evaluation": "NOT_RUN", "chat_quality_improvement": "NOT_ESTABLISHED",
    }
