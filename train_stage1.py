# pyright: reportMissingImports=false

"""Stage 1 EDEF Training.

Stage 1 freezes Qwen and trains only the side-signal modules:
- distribution mode: projector + gate
- medical mode: projector + gate + medical-encoder LoRA
"""

from __future__ import annotations

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
from edef_model import (
    DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
    DEFAULT_MEDICAL_CHUNK_OVERLAP,
    DEFAULT_MEDICAL_CHUNK_SIZE,
    DEFAULT_MEDICAL_ENCODER_MODEL,
    DEFAULT_MEDICAL_LORA_TARGET_MODULES,
    DEFAULT_MEDICAL_LORA_TOP_LAYERS,
    DEFAULT_SIGNAL_SOURCE,
    apply_medical_encoder_lora,
    attach_edef_to_model,
    get_edef_config,
    save_edef_checkpoint,
)
from ner_dataset_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DIST_PATH,
    DEFAULT_PHASE1_MODEL_PATH,
    load_task_metadata_from_dist_path,
)


class EDEFTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
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
    def __init__(self, model: Any, log_every_steps: int = 100):
        self.model = model
        self.log_every_steps = log_every_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
        del args, control, kwargs
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return None

        gate = getattr(self.model, "fusion_gate", None)
        gate_net = getattr(gate, "gate_net", None)
        gate_bias = getattr(gate_net, "bias", None)
        if gate_bias is None:
            return None

        with torch.no_grad():
            gate_sigmoid = torch.sigmoid(gate_bias.detach())
            print(
                f"Step {state.global_step}: Gate bias mean={gate_bias.mean().item():.4f}, "
                f"sigmoid mean={gate_sigmoid.mean().item():.4f}, "
                f"min={gate_sigmoid.min().item():.4f}, max={gate_sigmoid.max().item():.4f}"
            )
        return None


def _parse_csv_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("At least one LoRA target module must be provided.")
    return modules


def _resolve_task_setup(args: argparse.Namespace) -> tuple[dict[str, Any], int, str]:
    if args.signal_source == "distribution" and not args.dist_path:
        raise ValueError("Distribution mode requires --dist_path.")

    metadata: dict[str, Any] = {"entity_types": []}
    dist_dim = 45
    instruction = str(args.instruction or "").strip()
    if args.dist_path:
        metadata = load_task_metadata_from_dist_path(args.dist_path)
        dist_dim = int(metadata["dist_dim"])
        if not instruction:
            instruction = str(metadata["instruction"])
    if not instruction:
        raise ValueError("Provide --instruction or --dist_path with task metadata.")
    return metadata, dist_dim, instruction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 EDEF training (distribution or semantic medical fusion)."
    )
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
        help="Entity distribution file path. Optional in medical mode if --instruction is provided.",
    )
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument(
        "--signal_source",
        type=str,
        default=DEFAULT_SIGNAL_SOURCE,
        choices=["distribution", "medical_encoder"],
    )
    parser.add_argument(
        "--medical_encoder_model",
        type=str,
        default=DEFAULT_MEDICAL_ENCODER_MODEL,
    )
    parser.add_argument(
        "--medical_chunk_size", type=int, default=DEFAULT_MEDICAL_CHUNK_SIZE
    )
    parser.add_argument(
        "--medical_chunk_overlap", type=int, default=DEFAULT_MEDICAL_CHUNK_OVERLAP
    )
    parser.add_argument(
        "--max_prompt_medical_tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
    )
    parser.add_argument("--medical_lora_r", type=int, default=8)
    parser.add_argument("--medical_lora_alpha", type=int, default=16)
    parser.add_argument(
        "--medical_lora_target_modules",
        type=str,
        default=",".join(DEFAULT_MEDICAL_LORA_TARGET_MODULES),
    )
    parser.add_argument(
        "--medical_lora_top_layers",
        type=int,
        default=DEFAULT_MEDICAL_LORA_TOP_LAYERS,
    )
    parser.add_argument("--medical_lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--medical_lora_use_dora", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="saves/edef-stage1")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gate_log_steps", type=int, default=100)
    return parser.parse_args()


def load_phase1_model(args: argparse.Namespace):
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


def print_trainable_params(model: Any):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({pct:.2f}%)")


def patch_transformers_tf32() -> None:
    if not hasattr(torch.backends, "fp32_precision"):
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is None or not hasattr(matmul_backend, "allow_tf32"):
        return

    import transformers.training_args as hf_training_args

    def _legacy_enable_tf32(enable: bool) -> None:
        matmul_backend.allow_tf32 = enable
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = enable

    hf_training_args.enable_tf32 = _legacy_enable_tf32


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    patch_transformers_tf32()

    task_metadata, dist_dim, instruction = _resolve_task_setup(args)
    tokenizer_source = args.base_model if args.phase1_adapter else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    medical_tokenizer = None
    if args.signal_source == "medical_encoder":
        medical_tokenizer = AutoTokenizer.from_pretrained(
            args.medical_encoder_model,
            trust_remote_code=True,
        )

    print("Loading Phase 1 model...")
    model = load_phase1_model(args)

    print("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = int(getattr(model.config, "hidden_size", 2560))
    model = attach_edef_to_model(
        model,
        dist_dim=dist_dim,
        hidden_dim=hidden_dim,
        signal_source=args.signal_source,
        medical_encoder_model_name=args.medical_encoder_model,
        medical_chunk_size=args.medical_chunk_size,
        medical_chunk_overlap=args.medical_chunk_overlap,
        max_prompt_medical_tokens=args.max_prompt_medical_tokens,
    )

    if args.signal_source == "medical_encoder":
        model = apply_medical_encoder_lora(
            model,
            r=args.medical_lora_r,
            alpha=args.medical_lora_alpha,
            target_modules=_parse_csv_modules(args.medical_lora_target_modules),
            top_layers=args.medical_lora_top_layers,
            dropout=args.medical_lora_dropout,
            use_dora=args.medical_lora_use_dora,
        )
        medical_encoder = getattr(model, "medical_encoder", None)
        gradient_checkpoint_fn = getattr(
            medical_encoder, "gradient_checkpointing_enable", None
        )
        if callable(gradient_checkpoint_fn):
            gradient_checkpoint_fn()

    for param in model.entity_projector.parameters():
        param.requires_grad = True
    for param in model.fusion_gate.parameters():
        param.requires_grad = True

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    print_trainable_params(model)
    print(f"Signal source: {args.signal_source}")
    if task_metadata.get("entity_types"):
        print(
            f"Loaded {len(task_metadata['entity_types'])} entity types: {task_metadata['entity_types']}"
        )
    print(f"EDEF config: {get_edef_config(model)}")

    print("Building EDEF datasets...")
    train_dataset = build_edef_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        signal_source=args.signal_source,
        medical_tokenizer=medical_tokenizer,
        max_length=args.max_length,
        dist_dim=dist_dim,
        medical_chunk_size=args.medical_chunk_size,
        medical_chunk_overlap=args.medical_chunk_overlap,
        max_prompt_medical_tokens=args.max_prompt_medical_tokens,
        dataset_split=args.train_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    eval_dataset = build_edef_dataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        signal_source=args.signal_source,
        medical_tokenizer=medical_tokenizer,
        max_length=args.max_length,
        dist_dim=dist_dim,
        medical_chunk_size=args.medical_chunk_size,
        medical_chunk_overlap=args.medical_chunk_overlap,
        max_prompt_medical_tokens=args.max_prompt_medical_tokens,
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
        callbacks=[
            GateLoggingCallback(model=model, log_every_steps=args.gate_log_steps)
        ],
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
