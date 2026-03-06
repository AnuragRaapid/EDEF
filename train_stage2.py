"""Stage 2 EDEF Training: Joint LoRA + EDEF.

Trains LoRA adapters + EntityDistProjector + GatedFusion jointly.
This is the main training stage where the model learns to leverage distribution features for NER.
"""

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
load_edef_checkpoint = _model_mod.load_edef_checkpoint
save_edef_checkpoint = _model_mod.save_edef_checkpoint

if __package__:
    _dataset_utils_mod = importlib.import_module(
        ".ner_dataset_utils", package=__package__
    )
else:
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

DEFAULT_DATASET_NAME = _dataset_utils_mod.DEFAULT_DATASET_NAME
DEFAULT_DIST_PATH = _dataset_utils_mod.DEFAULT_DIST_PATH
DEFAULT_PHASE1_MODEL_PATH = _dataset_utils_mod.DEFAULT_PHASE1_MODEL_PATH
load_task_metadata_from_dist_path = _dataset_utils_mod.load_task_metadata_from_dist_path


class EDEFTrainer(Trainer):
    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ):
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def get_gate_module(model):
    if hasattr(model, "fusion_gate"):
        return model.fusion_gate
    if hasattr(model, "base_model"):
        return get_gate_module(model.base_model)
    if hasattr(model, "model"):
        return get_gate_module(model.model)
    return None


def get_model_with_edef(model):
    if hasattr(model, "entity_projector") and hasattr(model, "fusion_gate"):
        return model
    if hasattr(model, "base_model"):
        found = get_model_with_edef(model.base_model)
        if found is not None:
            return found
    if hasattr(model, "model"):
        found = get_model_with_edef(model.model)
        if found is not None:
            return found
    return None


def patch_transformers_tf32() -> None:
    if not hasattr(torch.backends, "fp32_precision"):
        return

    cuda_backend = getattr(torch.backends, "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is None or not hasattr(matmul_backend, "allow_tf32"):
        return

    import transformers.training_args as hf_training_args

    # TorchInductor still reads the legacy TF32 getter during torch.compile().
    def _legacy_enable_tf32(enable: bool) -> None:
        matmul_backend.allow_tf32 = enable
        cudnn_backend = getattr(torch.backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = enable

    hf_training_args.enable_tf32 = _legacy_enable_tf32


class GateLoggingCallback(TrainerCallback):
    def __init__(self, model: Any, log_every_steps: int = 100):
        self.model = model
        self.log_every_steps = log_every_steps

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, kwargs
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return None

        gate = get_gate_module(self.model)
        gate_net = getattr(gate, "gate_net", None)
        gate_bias = getattr(gate_net, "bias", None)
        if gate_bias is None:
            return None

        with torch.no_grad():
            gate_sigmoid = torch.sigmoid(gate_bias.detach())
            print(
                "Step "
                + str(state.global_step)
                + ": Gate bias mean="
                + f"{gate_bias.mean().item():.4f}, "
                + f"sigmoid mean={gate_sigmoid.mean().item():.4f}, "
                + f"min={gate_sigmoid.min().item():.4f}, max={gate_sigmoid.max().item():.4f}"
            )

        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2 EDEF training (LoRA + projector + gate)."
    )
    parser.add_argument(
        "--phase1_model",
        type=str,
        default=DEFAULT_PHASE1_MODEL_PATH,
        help="Path to Phase 1 model (merged weights).",
    )
    parser.add_argument(
        "--phase1_adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter path for merging.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="Base model ID if using adapter approach.",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="saves/edef-stage1/edef_checkpoint",
        help="Path to Stage 1 EDEF checkpoint.",
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
    parser.add_argument("--output_dir", type=str, default="saves/edef-stage2")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--gate_log_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip loading Stage 1 checkpoint and train EDEF modules from scratch.",
    )
    return parser.parse_args()


def load_phase1_model(args):
    if args.phase1_adapter:
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


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    patch_transformers_tf32()
    task_metadata = load_task_metadata_from_dist_path(args.dist_path)
    dist_dim = int(task_metadata["dist_dim"])
    instruction = str(task_metadata["instruction"])

    tokenizer_source = args.base_model if args.phase1_adapter else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Phase 1 model...")
    model = load_phase1_model(args)

    hidden_dim = getattr(model.config, "hidden_size", 2560)
    model = attach_edef_to_model(model, dist_dim=dist_dim, hidden_dim=hidden_dim)

    if not args.skip_stage1 and os.path.exists(args.stage1_checkpoint):
        load_edef_checkpoint(model, args.stage1_checkpoint)
        print(f"Loaded Stage 1 EDEF checkpoint from {args.stage1_checkpoint}")
    else:
        print("Starting EDEF modules from scratch (no Stage 1 checkpoint)")

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
        modules_to_save=["entity_projector", "fusion_gate"],
        lora_dropout=0,
        bias="none",
        use_dora=True,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    original_peft_forward = model.forward

    def peft_edef_forward(*forward_args, **forward_kwargs):
        return original_peft_forward(*forward_args, **forward_kwargs)

    model.forward = peft_edef_forward

    gradient_checkpoint_fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gradient_checkpoint_fn):
        gradient_checkpoint_fn()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        setattr(config, "use_cache", False)

    model.print_trainable_parameters()
    print(
        f"Loaded {len(task_metadata['entity_types'])} entity types: {task_metadata['entity_types']}"
    )

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
        optim="adamw_8bit",
        weight_decay=0.01,
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

    print("Starting Stage 2 EDEF training...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    base_with_edef = get_model_with_edef(model)
    if base_with_edef is None:
        raise RuntimeError(
            "Could not find model carrying EDEF modules for checkpoint save."
        )
    save_edef_checkpoint(
        base_with_edef, os.path.join(args.output_dir, "edef_checkpoint")
    )

    print("Stage 2 training complete!")
    print(f"LoRA + EDEF saved to {args.output_dir}")


if __name__ == "__main__":
    main()
