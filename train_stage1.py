# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedImport=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportImplicitRelativeImport=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Stage 1 late-correction EDEF training."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from edef_data import EDEFDataCollator, build_edef_dataset
from edef_model import attach_edef_to_model, get_edef_host, save_edef_checkpoint
from edef_paths import resolve_phase2_split_path
from edef_training_logging import Stage1DiagnosticsCallback as GateLoggingCallback


def _should_use_bf16(requested: bool) -> bool:
    return bool(
        requested and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )


def _resolve_model_dtype(use_bf16: bool) -> torch.dtype:
    return torch.bfloat16 if use_bf16 else torch.float32


class EDEFTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        projector_learning_rate: float,
        gate_learning_rate: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.projector_learning_rate = projector_learning_rate
        self.gate_learning_rate = gate_learning_rate

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        entity_dist_vectors = inputs.pop("entity_dist_vectors", None)
        entity_prompt_mask = inputs.pop("entity_prompt_mask", None)
        if entity_dist_vectors is not None:
            inputs["entity_dist_vectors"] = entity_dist_vectors
        if entity_prompt_mask is not None:
            inputs["entity_prompt_mask"] = entity_prompt_mask
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer

        projector_decay: list[torch.nn.Parameter] = []
        projector_no_decay: list[torch.nn.Parameter] = []
        gate_decay: list[torch.nn.Parameter] = []
        gate_no_decay: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "fusion_gate" in name:
                target_list = (
                    gate_no_decay
                    if name.endswith(".bias") or "norm" in name.lower()
                    else gate_decay
                )
            else:
                target_list = (
                    projector_no_decay
                    if name.endswith(".bias") or "norm" in name.lower()
                    else projector_decay
                )
            target_list.append(param)

        optimizer_groups: list[dict[str, Any]] = []
        if projector_decay:
            optimizer_groups.append(
                {
                    "group_name": "projector_decay",
                    "params": projector_decay,
                    "lr": self.projector_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if projector_no_decay:
            optimizer_groups.append(
                {
                    "group_name": "projector_no_decay",
                    "params": projector_no_decay,
                    "lr": self.projector_learning_rate,
                    "weight_decay": 0.0,
                }
            )
        if gate_decay:
            optimizer_groups.append(
                {
                    "group_name": "gate_decay",
                    "params": gate_decay,
                    "lr": self.gate_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if gate_no_decay:
            optimizer_groups.append(
                {
                    "group_name": "gate_no_decay",
                    "params": gate_no_decay,
                    "lr": self.gate_learning_rate,
                    "weight_decay": 0.0,
                }
            )

        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
        )
        return self.optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 late-correction EDEF training."
    )
    parser.add_argument(
        "--phase1_model",
        type=str,
        default="./qwen3-phase1-checkpoint",
        help="Path to the Phase 1 merged model.",
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
        default=resolve_phase2_split_path("train_ner_filtered.json"),
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=resolve_phase2_split_path("val_ner_filtered.json"),
    )
    parser.add_argument("--dist_path", type=str, default="./entity_distributions.json")
    parser.add_argument("--output_dir", type=str, default="saves/late-edef-stage1")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Alias for --projector_lr"
    )
    parser.add_argument(
        "--edef_lr",
        type=float,
        default=None,
        help="Deprecated alias for --projector_lr.",
    )
    parser.add_argument(
        "--projector_lr",
        type=float,
        default=None,
        help="Stage 1 learning rate for the projector.",
    )
    parser.add_argument(
        "--gate_lr",
        type=float,
        default=None,
        help="Stage 1 learning rate for the gate.",
    )
    parser.add_argument(
        "--gate_lr_scale",
        type=float,
        default=3.0,
        help="Multiplier applied to projector_lr when --gate_lr is not set.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--gate_log_steps",
        type=int,
        default=100,
        help="How often to log EDEF diagnostics. Set 0 to disable.",
    )
    parser.add_argument("--insertion_layer", type=int, default=28)
    parser.add_argument("--projector_bottleneck_dim", type=int, default=512)
    parser.add_argument(
        "--projector_use_temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fusion_projected_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--corrector_dim", type=int, default=512)
    parser.add_argument("--corrector_heads", type=int, default=8)
    return parser.parse_args()


def load_phase1_model(args: argparse.Namespace) -> Any:
    use_bf16 = _should_use_bf16(args.bf16)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_model_dtype(use_bf16),
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    if args.phase1_adapter:
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
        model = PeftModel.from_pretrained(model, args.phase1_adapter)
        model = model.merge_and_unload()
        return model

    return AutoModelForCausalLM.from_pretrained(args.phase1_model, **load_kwargs)


def print_trainable_params(model: Any) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100.0) if total > 0 else 0.0
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({pct:.2f}%)")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.projector_lr is None:
        args.projector_lr = args.edef_lr if args.edef_lr is not None else args.lr
    if args.gate_lr is None:
        args.gate_lr = args.projector_lr * args.gate_lr_scale

    tokenizer_source = args.base_model if args.phase1_adapter else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Phase 1 model...")
    model = load_phase1_model(args)

    print("Freezing backbone parameters...")
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = int(getattr(model.config, "hidden_size", 2560))
    model = attach_edef_to_model(
        model,
        dist_dim=45,
        hidden_dim=hidden_dim,
        insertion_layer=args.insertion_layer,
        projector_bottleneck_dim=args.projector_bottleneck_dim,
        projector_use_temperature=args.projector_use_temperature,
        fusion_projected_norm=args.fusion_projected_norm,
        corrector_layers=0,
        corrector_dim=args.corrector_dim,
        corrector_heads=args.corrector_heads,
    )

    host = get_edef_host(model)
    for param in host.entity_projector.parameters():
        param.requires_grad = True
    for param in host.fusion_gate.parameters():
        param.requires_grad = True

    # Keep gradient checkpointing disabled because EDEF is injected through a
    # forward hook, and checkpoint recomputation must exactly match the
    # original forward pass.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    print_trainable_params(model)
    print(
        f"Stage 1 learning rates: projector={args.projector_lr:.2e}, gate={args.gate_lr:.2e}"
    )

    print("Building datasets...")
    train_dataset = build_edef_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        max_length=args.max_length,
        dist_dim=45,
    )
    eval_dataset = build_edef_dataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        dist_path=args.dist_path,
        max_length=args.max_length,
        dist_dim=45,
    )
    data_collator = EDEFDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    use_bf16 = _should_use_bf16(args.bf16)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.projector_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=use_bf16,
        fp16=False,
        tf32=torch.cuda.is_available(),
        optim="adamw_torch",
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_num_workers=2,
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
        projector_learning_rate=args.projector_lr,
        gate_learning_rate=args.gate_lr,
    )

    print("Starting Stage 1 late-correction training...")
    trainer.train()

    checkpoint_path = os.path.join(args.output_dir, "edef_checkpoint")
    save_edef_checkpoint(model, checkpoint_path)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Stage 1 checkpoint saved to {checkpoint_path}")
    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
