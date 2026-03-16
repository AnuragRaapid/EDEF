"""Baseline NER Model Evaluation Script.

Re-evaluates the baseline model (without EDEF) on the same test set
using the same evaluation methodology for fair comparison with EDEF.
"""

# pyright: reportPrivateImportUsage=false, reportAny=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ner_dataset_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DIST_PATH,
    DEFAULT_PHASE1_MODEL_PATH,
    NER_ROOT_END,
    build_ner_instruction,
    extract_entity_types_from_samples,
    load_ner_samples,
    load_task_metadata_from_dist_path,
    parse_output_payload,
)


def parse_ner_json(
    ner_string: str,
    source_text: str | None = None,
) -> list[tuple[str, str]]:
    entities = parse_output_payload(ner_string, source_text=source_text)
    return [
        (
            str(entity.get("text", "")).lower().strip(),
            str(entity.get("type", "")).lower().strip(),
        )
        for entity in entities
        if str(entity.get("text", "")).strip() and str(entity.get("type", "")).strip()
    ]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def get_word_set(entity: str) -> set[str]:
    return set(tokenize(entity))


def exact_match(
    pred_entities: list[tuple[str, str]],
    gold_entities: list[tuple[str, str]],
) -> tuple[int, int, int]:
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def relaxed_match(
    pred_entities: list[tuple[str, str]],
    gold_entities: list[tuple[str, str]],
    word_margin: int = 2,
) -> tuple[int, int, int]:
    pred_list = list(pred_entities)
    gold_list = list(gold_entities)

    matched_preds = set()
    matched_golds = set()
    tp = 0

    for pred_idx, (pred_entity, pred_type) in enumerate(pred_list):
        if pred_idx in matched_preds:
            continue

        pred_words = get_word_set(pred_entity)

        for gold_idx, (gold_entity, gold_type) in enumerate(gold_list):
            if gold_idx in matched_golds:
                continue

            if pred_type != gold_type:
                continue

            gold_words = get_word_set(gold_entity)
            is_match = False

            if pred_entity == gold_entity:
                is_match = True
            elif pred_entity in gold_entity or gold_entity in pred_entity:
                is_match = True
            elif pred_words & gold_words:
                word_diff = abs(len(pred_words) - len(gold_words))
                if word_diff <= word_margin:
                    is_match = True
            elif pred_words.issubset(gold_words) or gold_words.issubset(pred_words):
                word_diff = abs(len(pred_words) - len(gold_words))
                if word_diff <= word_margin:
                    is_match = True
            elif len(pred_words.symmetric_difference(gold_words)) <= word_margin:
                is_match = True

            if is_match:
                tp += 1
                matched_preds.add(pred_idx)
                matched_golds.add(gold_idx)
                break

    fp = len(pred_list) - len(matched_preds)
    fn = len(gold_list) - len(matched_golds)
    return tp, fp, fn


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate_per_type(
    predictions: list[list[tuple[str, str]]],
    golds: list[list[tuple[str, str]]],
    word_margin: int = 2,
) -> dict[str, dict[str, dict[str, float | int]]]:
    all_types = set()
    for pred_list in predictions:
        for _, etype in pred_list:
            all_types.add(etype)
    for gold_list in golds:
        for _, etype in gold_list:
            all_types.add(etype)

    per_type_exact: dict[str, dict[str, float | int]] = {}
    per_type_relaxed: dict[str, dict[str, float | int]] = {}

    for etype in sorted(all_types):
        exact_tp, exact_fp, exact_fn = 0, 0, 0
        relaxed_tp, relaxed_fp, relaxed_fn = 0, 0, 0

        for pred_list, gold_list in zip(predictions, golds):
            pred_filtered = [(e, t) for e, t in pred_list if t == etype]
            gold_filtered = [(e, t) for e, t in gold_list if t == etype]

            tp, fp, fn = exact_match(pred_filtered, gold_filtered)
            exact_tp += tp
            exact_fp += fp
            exact_fn += fn

            tp, fp, fn = relaxed_match(pred_filtered, gold_filtered, word_margin)
            relaxed_tp += tp
            relaxed_fp += fp
            relaxed_fn += fn

        p, r, f1 = calculate_metrics(exact_tp, exact_fp, exact_fn)
        per_type_exact[etype] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": exact_tp,
            "fp": exact_fp,
            "fn": exact_fn,
        }

        p, r, f1 = calculate_metrics(relaxed_tp, relaxed_fp, relaxed_fn)
        per_type_relaxed[etype] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": relaxed_tp,
            "fp": relaxed_fp,
            "fn": relaxed_fn,
        }

    return {"exact": per_type_exact, "relaxed": per_type_relaxed}


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_path: str, adapter_path: str | None, base_model: str
):
    dtype = resolve_dtype()
    tokenizer_source = model_path if os.path.exists(model_path) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    if adapter_path:
        base_source = model_path if os.path.exists(model_path) else base_model
        model = AutoModelForCausalLM.from_pretrained(
            base_source,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_ner_json_string(generated_text: str) -> str:
    text = generated_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()

    if text.startswith("json"):
        text = text[4:].strip()

    if '{"ner"' in text:
        start = text.find('{"ner"')
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

    if "{'ner'" in text:
        start = text.find("{'ner'")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].replace("'", '"')

    return text


def generate_prediction(
    model,
    tokenizer,
    instruction: str,
    text: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    prompt = build_prompt(tokenizer, instruction, text)
    encoded = tokenizer(prompt, return_tensors="pt")

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded = {k: v.to(target_device) for k, v in encoded.items()}

    start = time.time()
    with torch.no_grad():
        try:
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=[NER_ROOT_END],
                tokenizer=tokenizer,
            )
        except TypeError:
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - start

    input_len = encoded["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline NER re-evaluation")
    parser.add_argument("--model_path", type=str, default=DEFAULT_PHASE1_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct")
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
        help="Optional distribution path used to load the exact label prompt.",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = parse_args()
    test_samples = load_ner_samples(
        args.test_data,
        split=args.test_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )

    if args.max_samples is not None:
        test_samples = test_samples[: args.max_samples]

    if os.path.exists(args.dist_path):
        instruction = load_task_metadata_from_dist_path(args.dist_path)["instruction"]
    else:
        instruction = build_ner_instruction(
            extract_entity_types_from_samples(test_samples)
        )

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        base_model=args.base_model,
    )

    predictions: list[list[tuple[str, str]]] = []
    golds: list[list[tuple[str, str]]] = []
    debug_predictions: list[dict[str, str | int]] = []
    inference_times: list[float] = []

    for idx, sample in enumerate(test_samples):
        input_text = sample.get("input", "")
        gold_output = sample.get("output", "")

        generated_text, elapsed = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            text=input_text,
            max_new_tokens=args.max_new_tokens,
        )
        inference_times.append(elapsed)

        pred_json_text = extract_ner_json_string(generated_text)

        pred_entities = parse_ner_json(pred_json_text, source_text=input_text)
        gold_entities = parse_ner_json(gold_output, source_text=input_text)

        predictions.append(pred_entities)
        golds.append(gold_entities)

        debug_predictions.append(
            {
                "index": idx,
                "input": input_text,
                "gold": gold_output,
                "predict": pred_json_text,
            }
        )

    word_margin = 2
    total_exact_tp, total_exact_fp, total_exact_fn = 0, 0, 0
    total_relaxed_tp, total_relaxed_fp, total_relaxed_fn = 0, 0, 0

    for pred_list, gold_list in zip(predictions, golds):
        tp, fp, fn = exact_match(pred_list, gold_list)
        total_exact_tp += tp
        total_exact_fp += fp
        total_exact_fn += fn

        tp, fp, fn = relaxed_match(pred_list, gold_list, word_margin)
        total_relaxed_tp += tp
        total_relaxed_fp += fp
        total_relaxed_fn += fn

    exact_precision, exact_recall, exact_f1 = calculate_metrics(
        total_exact_tp, total_exact_fp, total_exact_fn
    )
    relaxed_precision, relaxed_recall, relaxed_f1 = calculate_metrics(
        total_relaxed_tp, total_relaxed_fp, total_relaxed_fn
    )

    per_type_results = evaluate_per_type(predictions, golds, word_margin)

    avg_inference_time = (
        sum(inference_times) / len(inference_times) if inference_times else 0.0
    )

    result = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "test_data": args.test_data,
        "test_split": args.test_split,
        "num_samples": len(test_samples),
        "exact_match": {
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": exact_f1,
        },
        "relaxed_match": {
            "precision": relaxed_precision,
            "recall": relaxed_recall,
            "f1": relaxed_f1,
        },
        "per_type": per_type_results,
        "avg_inference_time": avg_inference_time,
        "predictions": debug_predictions,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved baseline evaluation results to: {output_path}")
    print(
        f"Exact F1: {exact_f1:.4f} | Relaxed F1: {relaxed_f1:.4f} | Samples: {len(test_samples)}"
    )


if __name__ == "__main__":
    main()
