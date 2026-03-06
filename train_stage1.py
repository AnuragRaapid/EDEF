# pyright: reportMissingImports=false

"""Stage 1 EDEF Training: Projector Alignment.

Trains ONLY the EntityDistProjector + GatedFusion while keeping Qwen3 frozen.
This teaches the projector to map distribution vectors into useful embeddings.
"""

import argparse
import os
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from edef_data import EDEFDataCollator, build_edef_dataset
from edef_model import attach_edef_to_model, save_edef_checkpoint
from ner_dataset_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DIST_PATH,
    DEFAULT_PHASE1_MODEL_PATH,
    load_task_metadata_from_dist_path,
)


class EDEFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        entity_dist_vectors = inputs.pop("entity_dist_vectors", None)
        if entity_dist_vectors is not None:
            inputs["entity_dist_vectors"] = entity_dist_vectors
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


class GateLoggingCallback(TrainerCallback):
    def __init__(self, model, log_every_steps=100):
        self.model = model
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return

        gate = getattr(self.model, "fusion_gate", None)
        gate_net = getattr(gate, "gate_net", None)
        gate_bias = getattr(gate_net, "bias", None)
        if gate_bias is None:
            return

        with torch.no_grad():
            gate_sigmoid = torch.sigmoid(gate_bias.detach())
            print(
                f"Step {state.global_step}: Gate bias mean={gate_bias.mean().item():.4f}, "
                f"sigmoid mean={gate_sigmoid.mean().item():.4f}, "
                f"min={gate_sigmoid.min().item():.4f}, max={gate_sigmoid.max().item():.4f}"
            )

        return


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 EDEF training (projector + gate only).")
    parser.add_argument(
        "--phase1_model",
        type=str,
        default=DEFAULT_PHASE1_MODEL_PATH,
        help="Path to Phase 1 merged model.",
    )
    parser.add_argument(
        "--phase1_adapter",
        type=str,
        default=None,
        help="Optional Phase 1 LoRA adapter path. If provided, base model is loaded and merged.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="Base model ID when using --phase1_adapter.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Local JSON path or Hugging Face dataset repo id for training data.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Local JSON path or Hugging Face dataset repo id for validation data.",
    )
    parser.add_argument(
        "--dist_path",
        type=str,
        default=DEFAULT_DIST_PATH,
        help="Entity distribution file path.",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="saves/edef-stage1")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gate_log_steps", type=int, default=100)
    return parser.parse_args()


def load_phase1_model(args):
    """Load Phase 1 model.

    Mode 1 (default): load fully merged model from --phase1_model.
    Mode 2: load base model + merge LoRA adapter from --phase1_adapter.
    """
    if args.phase1_adapter:
        from peft import PeftModel

        model: Any = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
        model = PeftModel.from_pretrained(model, args.phase1_adapter)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.phase1_model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
        )
    return model


def print_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({pct:.2f}%)")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    task_metadata = load_task_metadata_from_dist_path(args.dist_path)
    dist_dim = int(task_metadata["dist_dim"])
    instruction = str(task_metadata["instruction"])

    tokenizer_source = args.base_model if args.phase1_adapter else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Phase 1 model...")
    model = load_phase1_model(args)

    print("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = getattr(model.config, "hidden_size", 2560)
    model = attach_edef_to_model(model, dist_dim=dist_dim, hidden_dim=hidden_dim)

    for param in model.entity_projector.parameters():
        param.requires_grad = True
    for param in model.fusion_gate.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    print_trainable_params(model)
    print(f"Loaded {len(task_metadata['entity_types'])} entity types: {task_metadata['entity_types']}")

    print("Building EDEF datasets...")
    train_dataset = build_edef_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        max_length=args.max_length,
        dist_dim=dist_dim,
        dataset_split=args.train_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    eval_dataset = build_edef_dataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        max_length=args.max_length,
        dist_dim=dist_dim,
        dataset_split=args.val_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    data_collator = EDEFDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=not args.bf16,
        tf32=True,
        optim="adamw_torch_fused",
        torch_compile=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        report_to="none",
    )

    trainer = EDEFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[GateLoggingCallback(model=model, log_every_steps=args.gate_log_steps)],
    )

    print("Starting Stage 1 EDEF training...")
    trainer.train()

    checkpoint_path = os.path.join(args.output_dir, "edef_checkpoint")
    save_edef_checkpoint(model, checkpoint_path)
    print(f"EDEF checkpoint saved to {checkpoint_path}/")

    tokenizer.save_pretrained(args.output_dir)
    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
