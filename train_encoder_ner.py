"""Train an encoder-style BIO tagger on top of the Stage 2 EDEF backbone."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import EvalLoopOutput

from encoder_ner.bio_utils import (
    build_label_mappings,
    decode_bio_labels,
    normalize_entity_tuples,
    save_label_metadata,
)
from encoder_ner.dataset import BioDataCollator, _to_offset_list, build_bio_dataset
from encoder_ner.metrics import calculate_metrics, exact_match, relaxed_match
from encoder_ner.modeling import (
    DecoderBackboneTokenClassifier,
    enable_gradient_checkpointing,
    freeze_model_parameters,
    load_stage2_backbone,
)
from ner_dataset_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DIST_PATH,
    DEFAULT_PHASE1_MODEL_PATH,
    extract_entity_types_from_samples,
    load_ner_samples,
    load_task_metadata_from_dist_path,
)


class EncoderNERTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        lora_lr: float,
        head_lr: float,
        tokenizer: Any,
        id_to_label: dict[int, str],
        max_length: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lora_lr = lora_lr
        self.head_lr = head_lr
        self.tokenizer = tokenizer
        self.id_to_label = id_to_label
        self.max_length = max_length

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model_wrapped if self.model_wrapped is not None else self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        grouped_parameters: dict[tuple[str, float], dict[str, Any]] = {}

        for name, param in opt_model.named_parameters():
            if not param.requires_grad:
                continue

            if "classifier" in name:
                group_name = "classifier"
                group_lr = self.head_lr
            else:
                group_name = "lora"
                group_lr = self.lora_lr

            weight_decay = self.args.weight_decay if name in decay_parameters else 0.0
            group_key = (group_name, weight_decay)
            group = grouped_parameters.setdefault(
                group_key,
                {
                    "params": [],
                    "lr": group_lr,
                    "weight_decay": weight_decay,
                    "name": group_name,
                    "param_count": 0,
                },
            )
            group["params"].append(param)
            group["param_count"] += param.numel()

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        optimizer_kwargs.pop("optimizer_dict", None)
        self.optimizer = optimizer_cls(
            [
                {
                    "params": group["params"],
                    "lr": group["lr"],
                    "weight_decay": group["weight_decay"],
                }
                for group in grouped_parameters.values()
            ],
            **optimizer_kwargs,
        )

        print("Optimizer groups:")
        for group in grouped_parameters.values():
            print(
                f"  {group['name']}: lr={group['lr']:.2e}, "
                + f"weight_decay={group['weight_decay']:.2e}, "
                + f"params={group['param_count']:,}"
            )
        return self.optimizer

    def _resolve_decoder(self, model: Any) -> Any:
        candidates: list[Any] = [model]
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            try:
                candidates.append(get_base_model())
            except Exception:
                pass
        candidates.extend(
            [
                getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None),
                getattr(model, "model", None),
            ]
        )
        for candidate in candidates:
            if candidate is not None and callable(
                getattr(candidate, "decode_predictions", None)
            ):
                return candidate
        raise AttributeError("Could not resolve model.decode_predictions()")

    def _build_entity_metric_inputs(
        self, eval_dataset: Any, start_idx: int, stop_idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        features = [eval_dataset[idx] for idx in range(start_idx, stop_idx)]
        batch = self.data_collator(features)
        prepared_batch = self._prepare_inputs(batch)
        return features, prepared_batch

    def _compute_entity_metrics(
        self, eval_dataset: Any, metric_key_prefix: str
    ) -> dict[str, float]:
        samples = getattr(eval_dataset, "samples", None)
        if not isinstance(samples, Sequence) or not samples:
            return {}

        model = self.model
        decoder = self._resolve_decoder(model)
        was_training = model.training
        model.eval()

        all_predictions: list[list[tuple[str, str]]] = []
        all_golds: list[list[tuple[str, str]]] = []
        batch_size = max(1, self.args.per_device_eval_batch_size)

        try:
            for start_idx in range(0, len(eval_dataset), batch_size):
                stop_idx = min(start_idx + batch_size, len(eval_dataset))
                _, prepared_batch = self._build_entity_metric_inputs(
                    eval_dataset, start_idx, stop_idx
                )

                model_inputs = {
                    key: value
                    for key, value in prepared_batch.items()
                    if key != "labels"
                }
                with torch.no_grad():
                    outputs = model(**model_inputs)

                decoded_paths = decoder.decode_predictions(
                    outputs["logits"],
                    prepared_batch["attention_mask"],
                )

                for batch_offset, pred_ids in enumerate(decoded_paths):
                    sample = samples[start_idx + batch_offset]
                    text = str(sample.get("text", sample.get("input", "")))
                    encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        return_offsets_mapping=True,
                        add_special_tokens=False,
                    )
                    offset_mapping = _to_offset_list(
                        encoding["offset_mapping"].squeeze(0)
                    )
                    predicted_entities = decode_bio_labels(
                        text, offset_mapping, pred_ids, self.id_to_label
                    )
                    normalized_preds = normalize_entity_tuples(predicted_entities)
                    gold_entities = normalize_entity_tuples(sample.get("entities", []))
                    all_predictions.append(normalized_preds)
                    all_golds.append(gold_entities)
        finally:
            if was_training:
                model.train()

        exact_tp = exact_fp = exact_fn = 0
        relaxed_tp = relaxed_fp = relaxed_fn = 0
        for pred_entities, gold_entities in zip(all_predictions, all_golds):
            tp, fp, fn = exact_match(pred_entities, gold_entities)
            exact_tp += tp
            exact_fp += fp
            exact_fn += fn

            tp, fp, fn = relaxed_match(pred_entities, gold_entities)
            relaxed_tp += tp
            relaxed_fp += fp
            relaxed_fn += fn

        exact_precision, exact_recall, exact_f1 = calculate_metrics(
            exact_tp, exact_fp, exact_fn
        )
        relaxed_precision, relaxed_recall, relaxed_f1 = calculate_metrics(
            relaxed_tp, relaxed_fp, relaxed_fn
        )
        return {
            f"{metric_key_prefix}_entity_precision": exact_precision,
            f"{metric_key_prefix}_entity_recall": exact_recall,
            f"{metric_key_prefix}_entity_f1": exact_f1,
            f"{metric_key_prefix}_relaxed_entity_precision": relaxed_precision,
            f"{metric_key_prefix}_relaxed_entity_recall": relaxed_recall,
            f"{metric_key_prefix}_relaxed_entity_f1": relaxed_f1,
        }

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        eval_dataset = getattr(dataloader, "dataset", None)
        output.metrics.update(
            self._compute_entity_metrics(eval_dataset, metric_key_prefix)
        )
        return output

    def _save_checkpoint(self, model: Any, trial: Any = None) -> None:
        # Ensure stateful_callbacks has an entry for every ExportableState callback
        # so that resuming from a checkpoint saved without EarlyStoppingCallback (or
        # with a different set of callbacks) does not raise KeyError when saving.
        for cb in self.callback_handler.callbacks + [self.control]:
            if isinstance(cb, ExportableState):
                cb_name = cb.__class__.__name__
                if cb_name not in self.state.stateful_callbacks:
                    self.state.stateful_callbacks[cb_name] = cb.state()
        super()._save_checkpoint(model, trial)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BIO token classifier on top of the Stage 2 EDEF checkpoint."
    )
    parser.add_argument(
        "--phase1_model",
        type=str,
        default=DEFAULT_PHASE1_MODEL_PATH,
        help="Path to the merged Phase 1 model.",
    )
    parser.add_argument(
        "--phase1_adapter",
        type=str,
        default=None,
        help="Optional Phase 1 LoRA adapter path to merge into the base model.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-4B-Instruct-2507",
        help="Base model repo id when using --phase1_adapter.",
    )
    parser.add_argument(
        "--stage2_adapter",
        type=str,
        default=None,
        help="Path to the trained Stage 2 LoRA adapter directory.",
    )
    parser.add_argument(
        "--stage2_edef_checkpoint",
        type=str,
        default=None,
        help="Optional explicit path to the EDEF checkpoint directory inside Stage 2 output.",
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
        help="Entity distribution file path used by the Stage 2 EDEF model.",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="saves/encoder-ner")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--head_lr",
        type=float,
        default=5e-4,
        help="Learning rate for the BIO classification head.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Checkpoint save cadence. Keep this aligned with --eval_strategy.",
    )
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="Evaluation cadence. Use `epoch` so early stopping patience maps to epochs.",
    )
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Stop after this many evaluations without improving entity F1.",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum entity F1 improvement required to reset early stopping patience.",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--use_dora", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="bilstm_crf",
        choices=["linear", "bilstm", "bilstm_crf"],
        help="`bilstm_crf` is recommended for cleaner BIO boundary transitions.",
    )
    parser.add_argument("--head_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional attention implementation override, e.g. flash_attention_2.",
    )
    parser.add_argument(
        "--disable_edef",
        action="store_true",
        help="Ignore entity distribution fusion and train on the text alone.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Keep at most this many checkpoints (best is always kept when load_best_model_at_end=True).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint dir (e.g. output_dir/checkpoint-200) containing trainer_state.json to resume.",
    )
    return parser.parse_args()


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


def resolve_entity_types(
    args: argparse.Namespace,
) -> tuple[list[str], str | None, int | None]:
    instruction: str | None = None
    dist_dim: int | None = None

    if not args.disable_edef and os.path.exists(args.dist_path):
        metadata = load_task_metadata_from_dist_path(args.dist_path)
        entity_types = [str(item) for item in metadata["entity_types"]]
        instruction = str(metadata["instruction"])
        dist_dim = int(metadata["dist_dim"])
        if entity_types:
            return entity_types, instruction, dist_dim

    samples = load_ner_samples(
        args.train_data,
        split=args.train_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )
    entity_types = extract_entity_types_from_samples(samples)
    return entity_types, instruction, dist_dim


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    patch_transformers_tf32()

    entity_types, instruction, dist_dim = resolve_entity_types(args)
    label_list, label_to_id, id_to_label = build_label_mappings(entity_types)

    tokenizer_source = args.stage2_adapter or args.phase1_model or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading Stage 2 backbone...")
    backbone = load_stage2_backbone(
        phase1_model=args.phase1_model,
        phase1_adapter=args.phase1_adapter,
        base_model=args.base_model,
        stage2_adapter=args.stage2_adapter,
        stage2_edef_checkpoint=args.stage2_edef_checkpoint,
        dist_dim=dist_dim,
        attn_implementation=args.attn_implementation,
        use_edef=not args.disable_edef,
    )
    freeze_model_parameters(backbone)
    enable_gradient_checkpointing(backbone)

    model = DecoderBackboneTokenClassifier(
        backbone=backbone,
        num_labels=len(label_list),
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        label_list=label_list,
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
        modules_to_save=["classifier"],
        lora_dropout=args.lora_dropout,
        bias="none",
        use_dora=args.use_dora,
        task_type="TOKEN_CLS",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"BIO labels: {label_list}")

    print("Building BIO datasets...")
    dist_path = None if args.disable_edef else args.dist_path
    train_dataset = build_bio_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
        dist_path=dist_path,
        dataset_split=args.train_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    eval_dataset = build_bio_dataset(
        data_path=args.val_data,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=args.max_length,
        dist_path=dist_path,
        dataset_split=args.val_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    data_collator = BioDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    save_label_metadata(
        args.output_dir,
        entity_types=entity_types,
        label_list=label_list,
        extra_config={
            "head_type": args.head_type,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
            "use_edef": not args.disable_edef,
            "dist_path": dist_path,
            "phase1_model": args.phase1_model,
            "phase1_adapter": args.phase1_adapter,
            "base_model": args.base_model,
            "stage2_adapter": args.stage2_adapter,
            "stage2_edef_checkpoint": args.stage2_edef_checkpoint,
            "max_length": args.max_length,
            "eval_strategy": args.eval_strategy,
            "save_strategy": args.save_strategy,
            "early_stopping_patience": args.early_stopping_patience,
            "metric_for_best_model": "eval_entity_f1",
        },
    )

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=not args.bf16,
        tf32=True,
        optim=optim_name,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_entity_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        report_to="none",
        save_total_limit=args.save_total_limit,
    )

    trainer = EncoderNERTrainer(
        model=model,
        args=training_args,
        lora_lr=args.lr,
        head_lr=args.head_lr,
        tokenizer=tokenizer,
        id_to_label=id_to_label,
        max_length=args.max_length,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    print("Starting encoder-style BIO training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Encoder NER adapter saved to {args.output_dir}")
    if trainer.state.best_model_checkpoint:
        print(
            f"To resume from best checkpoint later, use: --resume_from_checkpoint {trainer.state.best_model_checkpoint}"
        )
    print(
        "To resume from any saved checkpoint, use: --resume_from_checkpoint <output_dir>/checkpoint-<step>"
    )


if __name__ == "__main__":
    main()
