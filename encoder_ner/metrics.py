from __future__ import annotations


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
                if abs(len(pred_words) - len(gold_words)) <= word_margin:
                    is_match = True
            elif pred_words.issubset(gold_words) or gold_words.issubset(pred_words):
                if abs(len(pred_words) - len(gold_words)) <= word_margin:
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
        for _, entity_type in pred_list:
            all_types.add(entity_type)
    for gold_list in golds:
        for _, entity_type in gold_list:
            all_types.add(entity_type)

    per_type_exact: dict[str, dict[str, float | int]] = {}
    per_type_relaxed: dict[str, dict[str, float | int]] = {}
    for entity_type in sorted(all_types):
        exact_tp = exact_fp = exact_fn = 0
        relaxed_tp = relaxed_fp = relaxed_fn = 0

        for pred_list, gold_list in zip(predictions, golds):
            pred_filtered = [(e, t) for e, t in pred_list if t == entity_type]
            gold_filtered = [(e, t) for e, t in gold_list if t == entity_type]

            tp, fp, fn = exact_match(pred_filtered, gold_filtered)
            exact_tp += tp
            exact_fp += fp
            exact_fn += fn

            tp, fp, fn = relaxed_match(pred_filtered, gold_filtered, word_margin)
            relaxed_tp += tp
            relaxed_fp += fp
            relaxed_fn += fn

        p, r, f1 = calculate_metrics(exact_tp, exact_fp, exact_fn)
        per_type_exact[entity_type] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": exact_tp,
            "fp": exact_fp,
            "fn": exact_fn,
        }
        p, r, f1 = calculate_metrics(relaxed_tp, relaxed_fp, relaxed_fn)
        per_type_relaxed[entity_type] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": relaxed_tp,
            "fp": relaxed_fp,
            "fn": relaxed_fn,
        }

    return {"exact": per_type_exact, "relaxed": per_type_relaxed}
