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
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments

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
    return bool(requested and torch.cuda.is_available() and torch.cuda.is_bf16_supported())


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
                target_list = edef_no_decay if name.endswith(".bias") or "norm" in name.lower() else edef_decay
            else:
                target_list = lora_no_decay if name.endswith(".bias") or "norm" in name.lower() else lora_decay
            target_list.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": edef_decay, "lr": self.edef_learning_rate, "weight_decay": self.args.weight_decay},
                {"params": edef_no_decay, "lr": self.edef_learning_rate, "weight_decay": 0.0},
                {"params": lora_decay, "lr": self.lora_learning_rate, "weight_decay": self.args.weight_decay},
                {"params": lora_no_decay, "lr": self.lora_learning_rate, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999),
        )
        return self.optimizer


class GateLoggingCallback(TrainerCallback):
    def __init__(self, model: Any, log_every_steps: int = 100):
        self.model = model
        self.log_every_steps = log_every_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return

        host = get_edef_host(self.model)
        gate_sigmoid = torch.sigmoid(host.fusion_gate.gate_net.bias.detach())
        msg = (
            f"Step {state.global_step}: late gate sigmoid mean={gate_sigmoid.mean().item():.4f}, "
            f"min={gate_sigmoid.min().item():.4f}, max={gate_sigmoid.max().item():.4f}"
        )
        if hasattr(host, "late_corrector"):
            corrector_gate = torch.sigmoid(host.late_corrector.output_gate.detach())
            msg += f", corrector gate mean={corrector_gate.mean().item():.4f}"
        print(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 late-correction EDEF training.")
    parser.add_argument("--phase1_model", type=str, default="./qwen3-phase1-checkpoint")
    parser.add_argument("--phase1_adapter", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="saves/late-edef-stage1/edef_checkpoint",
        help="Path to the Stage 1 late EDEF checkpoint.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/train_ner_filtered.json",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/val_ner_filtered.json",
    )
    parser.add_argument("--dist_path", type=str, default="./entity_distributions.json")
    parser.add_argument("--output_dir", type=str, default="saves/late-edef-stage2")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate.")
    parser.add_argument("--edef_lr", type=float, default=1e-3, help="Late module learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--gate_log_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--insertion_layer", type=int, default=None)
    parser.add_argument("--corrector_layers", type=int, default=2)
    parser.add_argument("--corrector_dim", type=int, default=512)
    parser.add_argument("--corrector_heads", type=int, default=8)
    parser.add_argument("--use_dora", action=argparse.BooleanOptionalAction, default=True)
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


def _resolve_stage1_attach_args(args: argparse.Namespace) -> dict[str, int]:
    config_payload = load_edef_config(args.stage1_checkpoint) if os.path.isdir(args.stage1_checkpoint) else {}
    insertion_layer = (
        args.insertion_layer
        if args.insertion_layer is not None
        else int(config_payload.get("insertion_layer", 28))
    )
    return {
        "insertion_layer": insertion_layer,
        "corrector_dim": int(config_payload.get("corrector_dim", args.corrector_dim)),
        "corrector_heads": int(config_payload.get("corrector_heads", args.corrector_heads)),
    }


def _build_modules_to_save(host: Any) -> list[str]:
    modules_to_save = ["entity_projector", "fusion_gate"]
    if hasattr(host, "late_corrector"):
        modules_to_save.append("late_corrector")
    return modules_to_save


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
        corrector_layers=args.corrector_layers,
        corrector_dim=args.corrector_dim,
        corrector_heads=args.corrector_heads,
        edef_dtype=torch.float32,
    )

    if not args.skip_stage1 and os.path.isdir(args.stage1_checkpoint):
        load_edef_checkpoint(model, args.stage1_checkpoint)
        print(f"Loaded Stage 1 late EDEF checkpoint from {args.stage1_checkpoint}")
    else:
        print("Starting projector/gate from scratch (no Stage 1 checkpoint found)")

    host = get_edef_host(model)
    modules_to_save = _build_modules_to_save(host)
    num_layers = get_decoder_layer_count(model)
    top_lora_layers = list(range(host.edef_insertion_layer + 1, num_layers))
    if not top_lora_layers:
        raise ValueError("LoRA target layer list is empty; insertion_layer is too close to the top")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=top_lora_layers,
        layers_pattern="layers",
        modules_to_save=modules_to_save,
        lora_dropout=0.0,
        bias="none",
        use_dora=args.use_dora,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    gradient_checkpoint_fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gradient_checkpoint_fn):
        gradient_checkpoint_fn()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        setattr(config, "use_cache", False)

    model.print_trainable_parameters()

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
        learning_rate=args.lr,
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
        callbacks=[GateLoggingCallback(model=model, log_every_steps=args.gate_log_steps)],
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
