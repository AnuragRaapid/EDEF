"""Evaluate the encoder-style BIO tagger against entity-level NER metrics."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from encoder_ner.bio_utils import (
    LABEL_METADATA_FILENAME,
    decode_bio_labels,
    load_label_metadata,
    normalize_entity_tuples,
)
from encoder_ner.dataset import _to_offset_list
from encoder_ner.metrics import (
    calculate_metrics,
    evaluate_per_type,
    exact_match,
    relaxed_match,
)
from encoder_ner.modeling import DecoderBackboneTokenClassifier, load_stage2_backbone
from ner_dataset_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DIST_PATH,
    DEFAULT_PHASE1_MODEL_PATH,
    load_ner_samples,
)

try:
    from distribution_alignment import get_token_distributions, load_distributions
except ImportError:
    get_token_distributions = None
    load_distributions = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the encoder-style BIO tagger on entity-level NER metrics."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved encoder NER adapter directory.",
    )
    parser.add_argument(
        "--phase1_model",
        type=str,
        default=DEFAULT_PHASE1_MODEL_PATH,
        help="Path to the merged Phase 1 model.",
    )
    parser.add_argument("--phase1_adapter", type=str, default=None)
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Qwen3-4B-Instruct-2507",
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
        "--test_data",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Local JSON path or Hugging Face dataset repo id for test data.",
    )
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--dist_path",
        type=str,
        default=DEFAULT_DIST_PATH,
        help="Entity distribution file path used by the Stage 2 EDEF model.",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--show_examples",
        type=int,
        default=5,
        help="Print this many mismatched predictions for quick inspection.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Optional attention implementation override, e.g. flash_attention_2.",
    )
    parser.add_argument(
        "--disable_edef",
        action="store_true",
        help="Ignore entity distribution fusion and evaluate the tagger on text alone.",
    )
    parser.add_argument(
        "--predictions_out",
        type=str,
        default=None,
        help="Optional JSON path to save raw predictions.",
    )
    return parser.parse_args()


def resolve_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = os.path.join(args.model_path, LABEL_METADATA_FILENAME)
    if not os.path.isfile(metadata_path):
        parent = os.path.dirname(args.model_path)
        if parent and os.path.isfile(os.path.join(parent, LABEL_METADATA_FILENAME)):
            metadata_path = os.path.join(parent, LABEL_METADATA_FILENAME)
    metadata = load_label_metadata(metadata_path)

    runtime = {
        "label_list": metadata["label_list"],
        "head_type": metadata.get("head_type", "bilstm_crf"),
        "head_hidden_dim": int(metadata.get("head_hidden_dim", 512)),
        "dropout": float(metadata.get("dropout", 0.1)),
        "use_edef": bool(metadata.get("use_edef", True)),
        "phase1_model": args.phase1_model or metadata.get("phase1_model"),
        "phase1_adapter": args.phase1_adapter or metadata.get("phase1_adapter"),
        "base_model": args.base_model or metadata.get("base_model"),
        "stage2_adapter": args.stage2_adapter or metadata.get("stage2_adapter"),
        "stage2_edef_checkpoint": args.stage2_edef_checkpoint
        or metadata.get("stage2_edef_checkpoint"),
        "dist_path": args.dist_path or metadata.get("dist_path"),
        "max_length": args.max_length or int(metadata.get("max_length", 2048)),
    }
    if args.disable_edef:
        runtime["use_edef"] = False
    return runtime


def resolve_decoder(model: Any) -> Any:
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


def main() -> None:
    args = parse_args()
    runtime = resolve_runtime_config(args)
    label_list = [str(label) for label in runtime["label_list"]]
    id_to_label = {idx: label for idx, label in enumerate(label_list)}

    tokenizer_source = (
        args.model_path if os.path.exists(args.model_path) else runtime["phase1_model"]
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dist_dim: int | None = None
    word_entity_dist: dict[str, list[float]] | None = None
    default_dist: list[float] | None = None
    if runtime["use_edef"]:
        if load_distributions is None or get_token_distributions is None:
            raise ImportError(
                "distribution_alignment helpers are required when EDEF is enabled"
            )
        word_entity_dist, default_dist = load_distributions(runtime["dist_path"])
        dist_dim = len(default_dist)

    print("Loading Stage 2 backbone...")
    backbone = load_stage2_backbone(
        phase1_model=runtime["phase1_model"],
        phase1_adapter=runtime["phase1_adapter"],
        base_model=runtime["base_model"],
        stage2_adapter=runtime["stage2_adapter"],
        stage2_edef_checkpoint=runtime["stage2_edef_checkpoint"],
        dist_dim=dist_dim,
        attn_implementation=args.attn_implementation,
        use_edef=runtime["use_edef"],
    )
    model = DecoderBackboneTokenClassifier(
        backbone=backbone,
        num_labels=len(label_list),
        head_type=runtime["head_type"],
        head_hidden_dim=runtime["head_hidden_dim"],
        dropout=runtime["dropout"],
        label_list=label_list,
    )
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()
    decoder = resolve_decoder(model)

    samples = load_ner_samples(
        args.test_data,
        split=args.test_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )

    device = next(model.parameters()).device
    all_predictions: list[list[tuple[str, str]]] = []
    all_golds: list[list[tuple[str, str]]] = []
    error_examples: list[dict[str, Any]] = []
    raw_predictions: list[dict[str, Any]] = []

    print(f"Evaluating {len(samples)} samples...")
    for sample_idx, sample in enumerate(samples):
        text = str(sample.get("text", sample.get("input", "")))
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=runtime["max_length"],
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offset_mapping = _to_offset_list(encoding["offset_mapping"].squeeze(0))
        model_inputs = {
            "input_ids": encoding["input_ids"].to(device),
            "attention_mask": encoding["attention_mask"].to(device),
        }

        if runtime["use_edef"]:
            assert word_entity_dist is not None
            assert default_dist is not None
            assert dist_dim is not None
            dist_vectors = get_token_distributions(
                text,
                tokenizer,
                word_entity_dist,
                default_dist,
                dist_dim,
            )
            if len(dist_vectors) > model_inputs["input_ids"].shape[1]:
                dist_vectors = dist_vectors[: model_inputs["input_ids"].shape[1]]
            elif len(dist_vectors) < model_inputs["input_ids"].shape[1]:
                pad_rows = model_inputs["input_ids"].shape[1] - len(dist_vectors)
                padding = torch.zeros(pad_rows, dist_dim, dtype=dist_vectors.dtype)
                dist_vectors = torch.cat([dist_vectors, padding], dim=0)
            model_inputs["entity_dist_vectors"] = dist_vectors.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(**model_inputs)
        decoded_paths = decoder.decode_predictions(
            outputs["logits"],
            model_inputs["attention_mask"],
        )
        pred_ids = decoded_paths[0]
        predicted_entities = decode_bio_labels(
            text, offset_mapping, pred_ids, id_to_label
        )
        normalized_preds = normalize_entity_tuples(predicted_entities)
        gold_entities = normalize_entity_tuples(sample.get("entities", []))

        all_predictions.append(normalized_preds)
        all_golds.append(gold_entities)
        raw_predictions.append(
            {
                "text": text,
                "predicted_entities": predicted_entities,
                "gold_entities": sample.get("entities", []),
            }
        )

        exact_tp, exact_fp, exact_fn = exact_match(normalized_preds, gold_entities)
        if args.show_examples > 0 and len(error_examples) < args.show_examples:
            if exact_fp > 0 or exact_fn > 0:
                error_examples.append(
                    {
                        "sample_index": sample_idx,
                        "text": text,
                        "predicted": predicted_entities,
                        "gold": sample.get("entities", []),
                    }
                )

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
    per_type = evaluate_per_type(all_predictions, all_golds)

    print("\nEntity-level metrics")
    print(
        f"Exact   : P={exact_precision:.4f} R={exact_recall:.4f} F1={exact_f1:.4f} "
        + f"(TP={exact_tp}, FP={exact_fp}, FN={exact_fn})"
    )
    print(
        f"Relaxed : P={relaxed_precision:.4f} R={relaxed_recall:.4f} F1={relaxed_f1:.4f} "
        + f"(TP={relaxed_tp}, FP={relaxed_fp}, FN={relaxed_fn})"
    )

    if error_examples:
        print("\nBoundary / tagging misses")
        for example in error_examples:
            print(f"[sample {example['sample_index']}] {example['text']}")
            print(f"  pred: {example['predicted']}")
            print(f"  gold: {example['gold']}")

    if args.predictions_out:
        with open(args.predictions_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "exact": {
                        "precision": exact_precision,
                        "recall": exact_recall,
                        "f1": exact_f1,
                        "tp": exact_tp,
                        "fp": exact_fp,
                        "fn": exact_fn,
                    },
                    "relaxed": {
                        "precision": relaxed_precision,
                        "recall": relaxed_recall,
                        "f1": relaxed_f1,
                        "tp": relaxed_tp,
                        "fp": relaxed_fp,
                        "fn": relaxed_fn,
                    },
                    "per_type": per_type,
                    "examples": raw_predictions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nSaved predictions to {args.predictions_out}")


if __name__ == "__main__":
    main()
