#!/usr/bin/env python3
"""
NER Evaluation Script
Calculates exact and relaxed (with word margin) precision, recall, and F1 scores.
"""

import json
from typing import Dict, List, Set, Tuple

from ner_dataset_utils import parse_output_payload


def parse_ner_json(
    ner_string: str,
    source_text: str | None = None,
) -> List[Tuple[str, str]]:
    """Parse legacy JSON or layered inline-tag outputs."""
    entities = parse_output_payload(ner_string, source_text=source_text)
    return [
        (
            str(entity.get("text", "")).lower().strip(),
            str(entity.get("type", "")).lower().strip(),
        )
        for entity in entities
        if str(entity.get("text", "")).strip() and str(entity.get("type", "")).strip()
    ]


def tokenize(text: str) -> List[str]:
    """Simple tokenization - split on whitespace and punctuation."""
    # Split on whitespace and keep words
    return text.lower().split()


def get_word_set(entity: str) -> Set[str]:
    """Get set of words in an entity."""
    return set(tokenize(entity))


def exact_match(
    pred_entities: List[Tuple[str, str]], gold_entities: List[Tuple[str, str]]
) -> Tuple[int, int, int]:
    """
    Calculate exact match TP, FP, FN.
    Entity text and type must match exactly.
    """
    # Convert to sets for comparison (handle duplicates properly)
    pred_set = set(pred_entities)
    gold_set = set(gold_entities)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return tp, fp, fn


def relaxed_match(
    pred_entities: List[Tuple[str, str]],
    gold_entities: List[Tuple[str, str]],
    word_margin: int = 2,
) -> Tuple[int, int, int]:
    """
    Calculate relaxed match TP, FP, FN with word margin.

    A prediction is considered a TP if:
    1. The entity type matches exactly
    2. The entity text overlaps with gold OR
       the word difference is within the margin

    word_margin: Maximum allowed word difference for partial match
    """
    pred_list = list(pred_entities)
    gold_list = list(gold_entities)

    # Track which predictions and golds have been matched
    matched_preds = set()
    matched_golds = set()

    tp = 0

    # Try to match each prediction with a gold
    for pred_idx, (pred_entity, pred_type) in enumerate(pred_list):
        if pred_idx in matched_preds:
            continue

        pred_words = get_word_set(pred_entity)

        for gold_idx, (gold_entity, gold_type) in enumerate(gold_list):
            if gold_idx in matched_golds:
                continue

            # Type must match exactly
            if pred_type != gold_type:
                continue

            gold_words = get_word_set(gold_entity)

            # Check for relaxed text match
            is_match = False

            # Check 1: Exact match
            if pred_entity == gold_entity:
                is_match = True

            # Check 2: One contains the other
            elif pred_entity in gold_entity or gold_entity in pred_entity:
                is_match = True

            # Check 3: Word overlap exists and word count difference is within margin
            elif pred_words & gold_words:  # Has some overlap
                word_diff = abs(len(pred_words) - len(gold_words))
                if word_diff <= word_margin:
                    is_match = True

            # Check 4: Very similar entities (one is a subset of the other in terms of words)
            elif pred_words.issubset(gold_words) or gold_words.issubset(pred_words):
                word_diff = abs(len(pred_words) - len(gold_words))
                if word_diff <= word_margin:
                    is_match = True

            # Check 5: Entities differ by at most word_margin words total
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


def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 from TP, FP, FN."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate_per_type(
    predictions: List[List[Tuple[str, str]]],
    golds: List[List[Tuple[str, str]]],
    word_margin: int = 2,
) -> Dict:
    """Evaluate metrics per entity type."""
    # Collect all entity types
    all_types = set()
    for pred_list in predictions:
        for _, etype in pred_list:
            all_types.add(etype)
    for gold_list in golds:
        for _, etype in gold_list:
            all_types.add(etype)

    per_type_exact = {}
    per_type_relaxed = {}

    for etype in sorted(all_types):
        # Filter entities by type
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


def main():
    # Load predictions file
    input_file = (
        "/home/anurag/NER/Multi-task Finetuning/predictions_qwen_4B_full_new.json"
    )
    # input_file = "/home/anurag/NER/Multi-task Finetuning/Results/predictions_phase2_vllm_ner.json"

    print(f"Loading predictions from: {input_file}")
    print("=" * 80)

    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")
    print()

    # Parse predictions and gold labels
    predictions = []
    golds = []
    parse_errors_pred = 0
    parse_errors_gold = 0

    for item in data:
        pred_entities = parse_ner_json(item.get("predict", ""))
        gold_entities = parse_ner_json(item.get("label", ""))

        if not pred_entities and item.get("predict", "").strip():
            parse_errors_pred += 1
        if not gold_entities and item.get("label", "").strip():
            parse_errors_gold += 1

        predictions.append(pred_entities)
        golds.append(gold_entities)

    if parse_errors_pred > 0:
        print(f"Warning: {parse_errors_pred} prediction entries could not be parsed")
    if parse_errors_gold > 0:
        print(f"Warning: {parse_errors_gold} gold label entries could not be parsed")
    print()

    # Calculate overall exact match metrics
    total_exact_tp, total_exact_fp, total_exact_fn = 0, 0, 0
    total_relaxed_tp, total_relaxed_fp, total_relaxed_fn = 0, 0, 0

    word_margin = 2

    for pred_list, gold_list in zip(predictions, golds):
        tp, fp, fn = exact_match(pred_list, gold_list)
        total_exact_tp += tp
        total_exact_fp += fp
        total_exact_fn += fn

        tp, fp, fn = relaxed_match(pred_list, gold_list, word_margin)
        total_relaxed_tp += tp
        total_relaxed_fp += fp
        total_relaxed_fn += fn

    # Calculate overall metrics
    exact_precision, exact_recall, exact_f1 = calculate_metrics(
        total_exact_tp, total_exact_fp, total_exact_fn
    )
    relaxed_precision, relaxed_recall, relaxed_f1 = calculate_metrics(
        total_relaxed_tp, total_relaxed_fp, total_relaxed_fn
    )

    # Print results
    print("=" * 80)
    print("OVERALL METRICS (Micro-averaged)")
    print("=" * 80)

    print("\n--- EXACT MATCH ---")
    print(f"  True Positives:  {total_exact_tp}")
    print(f"  False Positives: {total_exact_fp}")
    print(f"  False Negatives: {total_exact_fn}")
    print(f"  Precision:       {exact_precision:.4f} ({exact_precision * 100:.2f}%)")
    print(f"  Recall:          {exact_recall:.4f} ({exact_recall * 100:.2f}%)")
    print(f"  F1 Score:        {exact_f1:.4f} ({exact_f1 * 100:.2f}%)")

    print(f"\n--- RELAXED MATCH (word margin = {word_margin}) ---")
    print(f"  True Positives:  {total_relaxed_tp}")
    print(f"  False Positives: {total_relaxed_fp}")
    print(f"  False Negatives: {total_relaxed_fn}")
    print(
        f"  Precision:       {relaxed_precision:.4f} ({relaxed_precision * 100:.2f}%)"
    )
    print(f"  Recall:          {relaxed_recall:.4f} ({relaxed_recall * 100:.2f}%)")
    print(f"  F1 Score:        {relaxed_f1:.4f} ({relaxed_f1 * 100:.2f}%)")

    # Calculate per-type metrics
    print("\n" + "=" * 80)
    print("PER-TYPE METRICS")
    print("=" * 80)

    per_type_results = evaluate_per_type(predictions, golds, word_margin)

    # Print per-type exact match results
    print("\n--- EXACT MATCH (per entity type) ---")
    print(
        f"{'Entity Type':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    )
    print("-" * 75)

    exact_types = per_type_results["exact"]
    for etype in sorted(exact_types.keys()):
        metrics = exact_types[etype]
        support = metrics["tp"] + metrics["fn"]
        print(
            f"{etype:<35} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {support:>10}"
        )

    # Print per-type relaxed match results
    print(f"\n--- RELAXED MATCH (per entity type, word margin = {word_margin}) ---")
    print(
        f"{'Entity Type':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    )
    print("-" * 75)

    relaxed_types = per_type_results["relaxed"]
    for etype in sorted(relaxed_types.keys()):
        metrics = relaxed_types[etype]
        support = metrics["tp"] + metrics["fn"]
        print(
            f"{etype:<35} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f} {support:>10}"
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal predicted entities: {sum(len(p) for p in predictions)}")
    print(f"Total gold entities:      {sum(len(g) for g in golds)}")
    print(f"Number of entity types:   {len(exact_types)}")
    print(f"\nExact Match F1:   {exact_f1:.4f} ({exact_f1 * 100:.2f}%)")
    print(f"Relaxed Match F1: {relaxed_f1:.4f} ({relaxed_f1 * 100:.2f}%)")
    print(f"\nImprovement from relaxed matching: +{(relaxed_f1 - exact_f1) * 100:.2f}%")


if __name__ == "__main__":
    main()
