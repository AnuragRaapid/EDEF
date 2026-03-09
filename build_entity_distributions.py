#!/usr/bin/env python3
# pyright: basic
import argparse
import json
import logging
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ner_dataset_utils import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET_NAME,
    build_ner_instruction,
    extract_entity_types_from_samples,
    load_ner_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-token entity type distributions from a local JSON file or Hugging Face dataset."
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Local JSON path or Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--count_split",
        type=str,
        default="train",
        help="Split used for distribution counts when --data_source is a dataset repo.",
    )
    parser.add_argument(
        "--label_splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits scanned to discover the full entity type inventory.",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional dataset revision for Hugging Face Hub loading.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Output directory for the generated distribution files.",
    )
    return parser.parse_args()


def _normalize_token(token: str) -> str:
    normalized = token.strip().lower().strip("`'\".,;:!?()[]{}")
    if not normalized:
        return ""
    if not any(ch.isalnum() for ch in normalized):
        return ""
    return normalized


def _process_tokenized_sample(
    sample: dict[str, Any],
    type_to_idx: dict[str, int],
    o_idx: int,
    word_entity_counts: dict[str, np.ndarray],
) -> bool:
    tokens = sample.get("tokens")
    if not isinstance(tokens, list):
        return False

    normalized_tokens = [_normalize_token(str(token)) for token in tokens]
    token_labels = [o_idx] * len(normalized_tokens)

    entities = sample.get("entities", [])
    if not isinstance(entities, list):
        return False

    for entity in entities:
        if not isinstance(entity, dict):
            return False

        entity_type = str(entity.get("type", "")).strip()
        type_idx = type_to_idx.get(entity_type)
        token_start = entity.get("token_start")
        token_end = entity.get("token_end")
        if (
            type_idx is None
            or not isinstance(token_start, int)
            or not isinstance(token_end, int)
            or token_start < 0
            or token_end < token_start
            or token_end >= len(normalized_tokens)
        ):
            return False

        for pos in range(token_start, token_end + 1):
            token_labels[pos] = type_idx

    for token, label_idx in zip(normalized_tokens, token_labels):
        if token:
            word_entity_counts[token][label_idx] += 1.0

    return True


def _process_word_fallback_sample(
    sample: dict[str, Any],
    type_to_idx: dict[str, int],
    o_idx: int,
    word_entity_counts: dict[str, np.ndarray],
) -> None:
    text = str(sample.get("text", sample.get("input", "")))
    text_words = [_normalize_token(token) for token in text.split()]
    text_words = [word for word in text_words if word]
    text_counter = Counter(text_words)
    entity_word_counter: Counter[str] = Counter()

    entities = sample.get("entities", [])
    if not isinstance(entities, list):
        return

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        entity_type = str(entity.get("type", "")).strip()
        type_idx = type_to_idx.get(entity_type)
        if type_idx is None:
            continue

        entity_words = [_normalize_token(token) for token in str(entity.get("text", "")).split()]
        entity_words = [word for word in entity_words if word]
        for word in entity_words:
            word_entity_counts[word][type_idx] += 1.0
            entity_word_counter[word] += 1

    for word, total_occ in text_counter.items():
        o_count = total_occ - entity_word_counter.get(word, 0)
        if o_count > 0:
            word_entity_counts[word][o_idx] += float(o_count)


def process_samples(
    samples: list[dict[str, Any]],
    type_to_idx: dict[str, int],
    o_idx: int,
    word_entity_counts: dict[str, np.ndarray],
) -> dict[str, int]:
    processed = 0
    tokenized_samples = 0
    fallback_samples = 0

    for sample in samples:
        if _process_tokenized_sample(sample, type_to_idx, o_idx, word_entity_counts):
            tokenized_samples += 1
        else:
            _process_word_fallback_sample(sample, type_to_idx, o_idx, word_entity_counts)
            fallback_samples += 1
        processed += 1

    return {
        "processed": processed,
        "tokenized_samples": tokenized_samples,
        "fallback_samples": fallback_samples,
    }


def top_words_for_type(
    word_entity_dist: dict[str, list[float]],
    type_idx: int,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    ranked = sorted(
        ((word, probs[type_idx]) for word, probs in word_entity_dist.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"word": word, "probability": float(probability)}
        for word, probability in ranked[:top_k]
        if probability > 0.0
    ]


def build_distributions(
    word_entity_counts: dict[str, np.ndarray],
    smoothing_alpha: float = 0.1,
) -> dict[str, list[float]]:
    word_entity_dist: dict[str, list[float]] = {}
    for word, counts in word_entity_counts.items():
        total = counts.sum()
        if total > 0:
            smoothed = counts + smoothing_alpha
            word_entity_dist[word] = (smoothed / smoothed.sum()).tolist()
    return word_entity_dist


def build_ngram_distributions(
    word_entity_counts: dict[str, np.ndarray],
    dist_dim: int,
    n_range: tuple[int, int] = (3, 5),
    min_ngram_count: float = 5.0,
    smoothing_alpha: float = 0.1,
) -> dict[str, list[float]]:
    """Build character n-gram level distributions from word-level counts.

    For each unique character n-gram (length 3-5) found across the vocabulary,
    accumulates the entity type counts from all words containing that n-gram.
    Allows fallback distribution inference for unseen words by aggregating
    distributions from their constituent character n-grams.
    """
    ngram_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(dist_dim, dtype=np.float64)
    )
    for word, counts in word_entity_counts.items():
        if len(word) < n_range[0]:
            continue
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(word) - n + 1):
                ngram = word[i : i + n]
                if any(ch.isalnum() for ch in ngram):
                    ngram_counts[ngram] += counts

    ngram_dist: dict[str, list[float]] = {}
    for ngram, counts in ngram_counts.items():
        total = counts.sum()
        if total >= min_ngram_count:
            smoothed = counts + smoothing_alpha
            ngram_dist[ngram] = (smoothed / smoothed.sum()).tolist()
    return ngram_dist


def compute_global_prior(
    word_entity_counts: dict[str, np.ndarray],
    dist_dim: int,
    o_idx: int,
    entity_weight: float = 0.1,
) -> list[float]:
    """Compute a global entity type frequency prior for unknown words.

    Instead of hard [0,...,0, 1.0] ("definitely O"), returns a soft prior
    where entity types get a small probability proportional to their global
    frequency. This prevents zero-probability entity signals for unseen words.
    """
    global_counts = np.zeros(dist_dim, dtype=np.float64)
    for counts in word_entity_counts.values():
        global_counts += counts

    entity_total = global_counts.sum() - global_counts[o_idx]
    if entity_total <= 0:
        prior = np.zeros(dist_dim)
        prior[o_idx] = 1.0
        return prior.tolist()

    entity_freq = global_counts.copy()
    entity_freq[o_idx] = 0.0
    if entity_freq.sum() > 0:
        entity_freq = entity_freq / entity_freq.sum()

    prior = entity_freq * entity_weight
    prior[o_idx] = 1.0 - entity_weight
    return prior.tolist()


def print_sanity_checks(
    word_entity_dist: dict[str, list[float]],
    entity_types: list[str],
    type_to_idx: dict[str, int],
    words_with_entity_signal: int,
) -> None:
    print("\n=== Sanity Checks ===")
    for type_name in entity_types[: min(4, len(entity_types))]:
        idx = type_to_idx[type_name]
        top_words = top_words_for_type(word_entity_dist, idx, top_k=5)
        print(f"- Top 5 for {type_name}: {top_words}")

    print("\nVocabulary summary:")
    print(f"- Total unique words: {len(word_entity_dist)}")
    print(f"- Words with entity signal: {words_with_entity_signal}")


def _load_label_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if os.path.exists(args.data_source):
        logging.info("Loading local JSON samples from %s", args.data_source)
        return load_ner_samples(args.data_source)

    all_samples: list[dict[str, Any]] = []
    for split_name in args.label_splits:
        logging.info("Loading label inventory split '%s' from %s", split_name, args.data_source)
        all_samples.extend(
            load_ner_samples(
                args.data_source,
                split=split_name,
                dataset_revision=args.dataset_revision,
                cache_dir=args.cache_dir,
            )
        )
    return all_samples


def _load_count_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if os.path.exists(args.data_source):
        return load_ner_samples(args.data_source)

    logging.info("Loading count split '%s' from %s", args.count_split, args.data_source)
    return load_ner_samples(
        args.data_source,
        split=args.count_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    start_time = time.time()

    label_samples = _load_label_samples(args)
    entity_types = extract_entity_types_from_samples(label_samples)
    if not entity_types:
        raise ValueError("No entity types were found in the provided dataset.")

    type_to_idx = {name: idx for idx, name in enumerate(entity_types)}
    o_idx = len(entity_types)
    type_to_idx["O"] = o_idx
    dist_dim = o_idx + 1

    logging.info("Discovered entity types: %s", entity_types)
    logging.info("Distribution dimension: %d", dist_dim)

    count_samples = _load_count_samples(args)
    word_entity_counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(dist_dim, dtype=np.float64)
    )
    process_summary = process_samples(
        samples=count_samples,
        type_to_idx=type_to_idx,
        o_idx=o_idx,
        word_entity_counts=word_entity_counts,
    )

    word_entity_dist = build_distributions(word_entity_counts, smoothing_alpha=0.1)
    words_with_entity_signal = sum(
        1 for probs in word_entity_dist.values() if int(np.argmax(np.asarray(probs))) != o_idx
    )
    total_unique_words = len(word_entity_dist)
    vocabulary_coverage = (
        (words_with_entity_signal / total_unique_words) * 100.0
        if total_unique_words > 0
        else 0.0
    )

    ngram_dist = build_ngram_distributions(
        word_entity_counts, dist_dim=dist_dim, n_range=(3, 5),
        min_ngram_count=5.0, smoothing_alpha=0.1,
    )
    global_prior = compute_global_prior(
        word_entity_counts, dist_dim=dist_dim, o_idx=o_idx, entity_weight=0.1,
    )

    entity_counts = Counter(
        str(entity.get("type", "")).strip()
        for sample in label_samples
        for entity in sample.get("entities", [])
        if str(entity.get("type", "")).strip()
    )
    per_type_top10 = {
        type_name: top_words_for_type(word_entity_dist, type_to_idx[type_name], top_k=10)
        for type_name in entity_types
    }
    prompt_instruction = build_ner_instruction(entity_types)

    stats = {
        "data_source": args.data_source,
        "count_split": args.count_split,
        "label_splits": args.label_splits,
        "entity_types": entity_types,
        "entity_type_counts": dict(entity_counts),
        "distribution_dim": dist_dim,
        "o_index": o_idx,
        "prompt_instruction": prompt_instruction,
        "default_unknown_distribution": global_prior,
        "global_prior": global_prior,
        "ngram_count": len(ngram_dist),
        "smoothing_alpha": 0.1,
        "total_unique_words": total_unique_words,
        "words_with_entity_signal": words_with_entity_signal,
        "vocabulary_coverage": vocabulary_coverage,
        "samples_processed": process_summary["processed"],
        "tokenized_samples": process_summary["tokenized_samples"],
        "fallback_samples": process_summary["fallback_samples"],
        "per_type_top10": per_type_top10,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dist_path = args.out_dir / "entity_distributions.json"
    type_idx_path = args.out_dir / "entity_type_index.json"
    stats_path = args.out_dir / "distribution_stats.json"
    ngram_path = args.out_dir / "ngram_distributions.json"

    with dist_path.open("w", encoding="utf-8") as f:
        json.dump(word_entity_dist, f, ensure_ascii=True)
    with type_idx_path.open("w", encoding="utf-8") as f:
        json.dump(type_to_idx, f, ensure_ascii=True, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)
    with ngram_path.open("w", encoding="utf-8") as f:
        json.dump(ngram_dist, f, ensure_ascii=True)

    elapsed = time.time() - start_time
    print("\n=== Build Complete ===")
    print(f"Entity distributions written to: {dist_path}")
    print(f"N-gram distributions written to: {ngram_path} ({len(ngram_dist):,} n-grams)")
    print(f"Entity type index written to: {type_idx_path}")
    print(f"Distribution stats written to: {stats_path}")
    print(f"Total processing time: {elapsed:.2f}s")
    print(f"Samples processed for counts: {process_summary['processed']}")
    print(f"Tokenized-path samples: {process_summary['tokenized_samples']}")
    print(f"Fallback-path samples: {process_summary['fallback_samples']}")
    print(f"Entity types: {', '.join(entity_types)}")
    print(f"Global prior (default for unknown words): {[f'{v:.4f}' for v in global_prior[:5]]}...")

    print_sanity_checks(word_entity_dist, entity_types, type_to_idx, words_with_entity_signal)


if __name__ == "__main__":
    main()
