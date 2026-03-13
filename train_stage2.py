# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedImport=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportImplicitRelativeImport=false, reportUnannotatedClassAttribute=false, reportUnknownArgumentType=false, reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Stage 2 late-correction EDEF training."""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from edef_paths import resolve_phase2_split_path
from edef_training_logging import Stage2DiagnosticsCallback as GateLoggingCallback

if __package__:
    _data_mod = importlib.import_module(".edef_data", package=__package__)
    _model_mod = importlib.import_module(".edef_model", package=__package__)
else:
    _data_mod = importlib.import_module("edef_data")
    _model_mod = importlib.import_module("edef_model")

build_edef_dataset = _data_mod.build_edef_dataset
EDEFDataCollator = _data_mod.EDEFDataCollator
attach_edef_to_model = _model_mod.attach_edef_to_model
get_decoder_layer_count = _model_mod.get_decoder_layer_count
get_edef_host = _model_mod.get_edef_host
load_edef_checkpoint = _model_mod.load_edef_checkpoint
load_edef_config = _model_mod.load_edef_config
save_edef_checkpoint = _model_mod.save_edef_checkpoint


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
        lora_learning_rate: float,
        edef_learning_rate: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lora_learning_rate = lora_learning_rate
        self.edef_learning_rate = edef_learning_rate

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

        edef_decay: list[torch.nn.Parameter] = []
        edef_no_decay: list[torch.nn.Parameter] = []
        lora_decay: list[torch.nn.Parameter] = []
        lora_no_decay: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_edef = any(
                token in name
                for token in ("entity_projector", "fusion_gate", "late_corrector")
            )
            target_list = None
            if is_edef:
                target_list = (
                    edef_no_decay
                    if name.endswith(".bias") or "norm" in name.lower()
                    else edef_decay
                )
            else:
                target_list = (
                    lora_no_decay
                    if name.endswith(".bias") or "norm" in name.lower()
                    else lora_decay
                )
            target_list.append(param)

        optimizer_groups: list[dict[str, Any]] = []
        if edef_decay:
            optimizer_groups.append(
                {
                    "group_name": "edef_decay",
                    "params": edef_decay,
                    "lr": self.edef_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if edef_no_decay:
            optimizer_groups.append(
                {
                    "group_name": "edef_no_decay",
                    "params": edef_no_decay,
                    "lr": self.edef_learning_rate,
                    "weight_decay": 0.0,
                }
            )
        if lora_decay:
            optimizer_groups.append(
                {
                    "group_name": "lora_decay",
                    "params": lora_decay,
                    "lr": self.lora_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
        if lora_no_decay:
            optimizer_groups.append(
                {
                    "group_name": "lora_no_decay",
                    "params": lora_no_decay,
                    "lr": self.lora_learning_rate,
                    "weight_decay": 0.0,
                }
            )

        self.optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999))
        return self.optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 late-correction EDEF training."
    )
    parser.add_argument("--phase1_model", type=str, default="./qwen3-phase1-checkpoint")
    parser.add_argument("--phase1_adapter", type=str, default=None)
    parser.add_argument(
        "--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="saves/late-edef-stage1/edef_checkpoint",
        help="Path to the Stage 1 late EDEF checkpoint.",
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
    parser.add_argument("--output_dir", type=str, default="saves/late-edef-stage2")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate.")
    parser.add_argument(
        "--edef_lr", type=float, default=1e-3, help="Late module learning rate."
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument(
        "--gate_log_steps",
        type=int,
        default=100,
        help="How often to log EDEF diagnostics. Set 0 to disable.",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--insertion_layer", type=int, default=None)
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
    parser.add_argument("--corrector_layers", type=int, default=2)
    parser.add_argument("--corrector_dim", type=int, default=512)
    parser.add_argument("--corrector_heads", type=int, default=8)
    parser.add_argument(
        "--corrector_warmup_epochs",
        type=float,
        default=1.0,
        help="Warm up the late corrector alone before joint Stage 2 training.",
    )
    parser.add_argument(
        "--corrector_warmup_lr",
        type=float,
        default=None,
        help="Learning rate used during the corrector-only warmup. Defaults to --edef_lr.",
    )
    parser.add_argument(
        "--use_dora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip loading Stage 1 checkpoint and train projector/gate from scratch.",
    )
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
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
        model = PeftModel.from_pretrained(model, args.phase1_adapter)
        model = model.merge_and_unload()
        return model
    return AutoModelForCausalLM.from_pretrained(args.phase1_model, **load_kwargs)


def _resolve_stage1_attach_args(args: argparse.Namespace) -> dict[str, Any]:
    use_stage1_config = (not args.skip_stage1) and os.path.isdir(args.stage1_checkpoint)
    config_payload = (
        load_edef_config(args.stage1_checkpoint) if use_stage1_config else {}
    )
    insertion_layer = (
        args.insertion_layer
        if args.insertion_layer is not None
        else int(config_payload.get("insertion_layer", 28))
    )
    return {
        "insertion_layer": insertion_layer,
        "projector_bottleneck_dim": config_payload.get(
            "projector_bottleneck_dim",
            args.projector_bottleneck_dim,
        ),
        "projector_use_temperature": bool(
            config_payload.get(
                "projector_use_temperature", args.projector_use_temperature
            )
        ),
        "fusion_projected_norm": bool(
            config_payload.get("fusion_projected_norm", args.fusion_projected_norm)
        ),
        "corrector_dim": int(config_payload.get("corrector_dim", args.corrector_dim)),
        "corrector_heads": int(
            config_payload.get("corrector_heads", args.corrector_heads)
        ),
    }


def _build_modules_to_save(host: Any) -> list[str]:
    modules_to_save = ["entity_projector", "fusion_gate"]
    if hasattr(host, "late_corrector"):
        modules_to_save.append("late_corrector")
    return modules_to_save


def _freeze_all_parameters(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _set_stage2_trainable_params(model: Any) -> None:
    _freeze_all_parameters(model)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    host = get_edef_host(model)
    for module_name in ("entity_projector", "fusion_gate", "late_corrector"):
        module = getattr(host, module_name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True


def _set_corrector_warmup_trainable_params(model: Any) -> bool:
    _freeze_all_parameters(model)
    host = get_edef_host(model)
    corrector = getattr(host, "late_corrector", None)
    if corrector is None:
        return False
    for param in corrector.parameters():
        param.requires_grad = True
    return True


def _build_training_args(
    args: argparse.Namespace,
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    *,
    save_strategy: str,
    eval_strategy: str,
) -> TrainingArguments:
    use_bf16 = _should_use_bf16(args.bf16)
    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "bf16": use_bf16,
        "fp16": False,
        "tf32": torch.cuda.is_available(),
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy,
        "eval_strategy": eval_strategy,
        "remove_unused_columns": False,
        "seed": args.seed,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "dataloader_num_workers": 2,
        "report_to": "none",
    }
    if save_strategy == "steps":
        training_kwargs["save_steps"] = args.save_steps
    if eval_strategy == "steps":
        training_kwargs["eval_steps"] = args.save_steps
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer_source = args.base_model if args.phase1_adapter else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Phase 1 model...")
    model = load_phase1_model(args)

    attach_args = _resolve_stage1_attach_args(args)
    hidden_dim = int(getattr(model.config, "hidden_size", 2560))
    model = attach_edef_to_model(
        model,
        dist_dim=45,
        hidden_dim=hidden_dim,
        insertion_layer=attach_args["insertion_layer"],
        projector_bottleneck_dim=attach_args["projector_bottleneck_dim"],
        projector_use_temperature=attach_args["projector_use_temperature"],
        fusion_projected_norm=attach_args["fusion_projected_norm"],
        corrector_layers=args.corrector_layers,
        corrector_dim=attach_args["corrector_dim"],
        corrector_heads=attach_args["corrector_heads"],
    )

    if not args.skip_stage1 and os.path.isdir(args.stage1_checkpoint):
        load_edef_checkpoint(model, args.stage1_checkpoint)
        print(f"Loaded Stage 1 late EDEF checkpoint from {args.stage1_checkpoint}")
    elif args.skip_stage1:
        print(
            "Skipping Stage 1 checkpoint load and starting projector/gate from scratch."
        )
    else:
        print("Starting projector/gate from scratch (no Stage 1 checkpoint found)")

    host = get_edef_host(model)
    modules_to_save = _build_modules_to_save(host)
    num_layers = get_decoder_layer_count(model)
    top_lora_layers = list(range(host.edef_insertion_layer + 1, num_layers))
    if not top_lora_layers:
        raise ValueError(
            "LoRA target layer list is empty; insertion_layer is too close to the top"
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        layers_to_transform=top_lora_layers,
        layers_pattern="layers",
        modules_to_save=modules_to_save,
        lora_dropout=0.0,
        bias="none",
        use_dora=args.use_dora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    _set_stage2_trainable_params(model)

    # Keep gradient checkpointing disabled because EDEF is injected through a
    # forward hook, and checkpoint recomputation must exactly match the
    # original forward pass.
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        setattr(config, "use_cache", False)

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

    should_run_warmup = (
        args.corrector_warmup_epochs > 0
        and not args.skip_stage1
        and os.path.isdir(args.stage1_checkpoint)
        and hasattr(get_edef_host(model), "late_corrector")
    )
    if should_run_warmup:
        warmup_lr = (
            args.corrector_warmup_lr
            if args.corrector_warmup_lr is not None
            else args.edef_lr
        )
        if _set_corrector_warmup_trainable_params(model):
            warmup_args = _build_training_args(
                args,
                output_dir=os.path.join(args.output_dir, "corrector_warmup"),
                num_train_epochs=args.corrector_warmup_epochs,
                learning_rate=warmup_lr,
                save_strategy="no",
                eval_strategy="no",
            )
            warmup_trainer = EDEFTrainer(
                model=model,
                args=warmup_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                callbacks=[
                    GateLoggingCallback(
                        model=model,
                        log_every_steps=args.gate_log_steps,
                        stage_name="stage2_corrector_warmup",
                    )
                ],
                lora_learning_rate=args.lr,
                edef_learning_rate=warmup_lr,
            )
            print(
                f"Starting corrector warmup for {args.corrector_warmup_epochs:.2f} epoch(s) "
                f"at lr={warmup_lr:.2e}..."
            )
            warmup_trainer.train()
            _set_stage2_trainable_params(model)
    elif args.corrector_warmup_epochs > 0:
        print("Skipping corrector warmup because no Stage 1 checkpoint was loaded.")

    model.print_trainable_parameters()

    training_args = _build_training_args(
        args,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_strategy="steps",
        eval_strategy="steps",
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
        lora_learning_rate=args.lr,
        edef_learning_rate=args.edef_lr,
    )

    print("Starting Stage 2 late-correction training...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_edef_checkpoint(model, os.path.join(args.output_dir, "edef_checkpoint"))

    print("Stage 2 training complete!")
    print(f"LoRA + late EDEF saved to {args.output_dir}")


if __name__ == "__main__":
    main()
