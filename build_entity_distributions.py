#!/usr/bin/env python3
# pyright: basic
import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NER_PREFIX = "You are an expert medical Named Entity Recognition (NER) assistant"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-word entity type probability distributions from NER training data."
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("/home/anurag/NER/Multi-task Finetuning/Labels_NER.txt"),
        help="Path to Labels_NER.txt",
    )
    parser.add_argument(
        "--phase1-train-path",
        type=Path,
        default=Path(
            "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase1 Dataset/train.json"
        ),
        help="Path to Phase 1 train.json",
    )
    parser.add_argument(
        "--phase2-train-path",
        type=Path,
        default=Path(
            "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/train_ner_filtered.json"
        ),
        help="Path to Phase 2 train_ner_filtered.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/anurag/NER/Soft Prompt Tuning"),
        help="Output directory for JSON files",
    )
    return parser.parse_args()


def load_entity_types(labels_path: Path) -> list[str]:
    entity_types: list[str] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if "|" in line:
                line = line.split("|", 1)[1].strip()
            entity_types.append(line)
    return entity_types


def load_json_array(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {path}, got {type(data)}")
    return data


def parse_output_payload(output_value, sample_idx: int, source_name: str):
    if isinstance(output_value, dict):
        return output_value
    if isinstance(output_value, str):
        try:
            return json.loads(output_value)
        except json.JSONDecodeError as exc:
            logging.warning(
                "Skipping sample %d from %s: output JSON parse failed: %s",
                sample_idx,
                source_name,
                exc,
            )
            return None
    logging.warning(
        "Skipping sample %d from %s: unsupported output type %s",
        sample_idx,
        source_name,
        type(output_value),
    )
    return None


def process_samples(
    samples: list[dict],
    source_name: str,
    type_to_idx: dict[str, int],
    type_to_idx_casefold: dict[str, int],
    o_idx: int,
    word_entity_counts,
    start_processed: int,
) -> tuple[int, int]:
    processed = start_processed
    valid_ner_samples = 0

    for i, sample in enumerate(samples):
        try:
            instruction = str(sample.get("instruction", ""))
            if source_name == "phase1" and not instruction.startswith(NER_PREFIX):
                continue

            text = str(sample.get("input", ""))
            output_value = sample.get("output", "")
            output_payload = parse_output_payload(output_value, i, source_name)
            if output_payload is None:
                continue

            entities = output_payload.get("ner", [])
            if not isinstance(entities, list):
                logging.warning(
                    "Skipping sample %d from %s: 'ner' is not a list",
                    i,
                    source_name,
                )
                continue

            valid_ner_samples += 1
            text_words = [w for w in text.lower().split() if w]
            text_counter = Counter(text_words)
            entity_word_counter: Counter[str] = Counter()

            text_lower = text.lower()

            for entity in entities:
                if not isinstance(entity, (list, tuple)) or len(entity) != 2:
                    logging.warning(
                        "Skipping malformed entity in sample %d from %s: %r",
                        i,
                        source_name,
                        entity,
                    )
                    continue

                entity_text, entity_type = entity
                entity_text = str(entity_text)
                entity_type = str(entity_type)

                type_idx = type_to_idx.get(entity_type)
                if type_idx is None:
                    type_idx = type_to_idx_casefold.get(entity_type.casefold())

                if type_idx is None:
                    logging.warning(
                        "Unknown entity type '%s' in sample %d from %s; skipping entity",
                        entity_type,
                        i,
                        source_name,
                    )
                    continue

                entity_words = [w for w in entity_text.lower().split() if w]
                if not entity_words:
                    continue

                if entity_text.lower() not in text_lower:
                    logging.warning(
                        "Entity text not found in input (sample %d, %s): '%s'",
                        i,
                        source_name,
                        entity_text,
                    )

                for word in entity_words:
                    word_entity_counts[word][type_idx] += 1.0
                    entity_word_counter[word] += 1

            for word, total_occ in text_counter.items():
                o_count = total_occ - entity_word_counter.get(word, 0)
                if o_count > 0:
                    word_entity_counts[word][o_idx] += float(o_count)

            processed += 1
            if processed % 10000 == 0:
                logging.info("Processed %d samples so far...", processed)

        except Exception as exc:
            logging.warning(
                "Skipping sample %d from %s due to error: %s",
                i,
                source_name,
                exc,
            )
            continue

    return processed, valid_ner_samples


def top_words_for_type(
    word_entity_dist: dict[str, list[float]], type_idx: int, top_k: int = 10
) -> list[dict]:
    ranked = sorted(
        ((word, probs[type_idx]) for word, probs in word_entity_dist.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"word": word, "probability": float(prob)}
        for word, prob in ranked[:top_k]
        if prob > 0.0
    ]


def build_distributions(word_entity_counts) -> dict[str, list[float]]:
    word_entity_dist: dict[str, list[float]] = {}
    for word, counts in word_entity_counts.items():
        total = counts.sum()
        if total > 0:
            word_entity_dist[word] = (counts / total).tolist()
    return word_entity_dist


def print_sanity_checks(
    word_entity_dist: dict[str, list[float]],
    type_to_idx: dict[str, int],
    words_with_entity_signal: int,
) -> None:
    print("\n=== Sanity Checks ===")
    for type_name in [
        "Drug",
        "Medical_Condition",
        "Sign_Symptom",
        "Anatomical_Structure",
    ]:
        if type_name not in type_to_idx:
            print(f"- {type_name}: type not found")
            continue
        idx = type_to_idx[type_name]
        ranked = sorted(
            ((word, probs[idx]) for word, probs in word_entity_dist.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        top5 = [(w, float(p)) for w, p in ranked[:5] if p > 0.0]
        print(f"- Top 5 for {type_name}: {top5}")

    print("\nCommon word distributions:")
    for word in ["the", "pain", "mg", "blood"]:
        dist = word_entity_dist.get(word)
        if dist is None:
            print(f"- {word}: not in vocabulary")
        else:
            print(f"- {word}: {dist}")

    print("\nVocabulary summary:")
    print(f"- Total unique words: {len(word_entity_dist)}")
    print(f"- Words with entity signal: {words_with_entity_signal}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    start_time = time.time()

    entity_types = load_entity_types(args.labels_path)
    if len(entity_types) != 44:
        logging.warning("Expected 44 entity types, found %d", len(entity_types))

    type_to_idx = {name: idx for idx, name in enumerate(entity_types)}
    type_to_idx_casefold = {name.casefold(): idx for name, idx in type_to_idx.items()}
    o_idx = len(entity_types)
    type_to_idx["O"] = o_idx
    dist_dim = o_idx + 1

    logging.info("Loaded %d entity types. Distribution dim=%d", len(entity_types), dist_dim)

    word_entity_counts = defaultdict(lambda: np.zeros(dist_dim, dtype=np.float64))

    logging.info("Loading Phase 1 dataset from %s", args.phase1_train_path)
    phase1_load_start = time.time()
    phase1_samples = load_json_array(args.phase1_train_path)
    logging.info(
        "Phase 1 loaded: %d samples (%.2fs)",
        len(phase1_samples),
        time.time() - phase1_load_start,
    )

    logging.info("Loading Phase 2 dataset from %s", args.phase2_train_path)
    phase2_load_start = time.time()
    phase2_samples = load_json_array(args.phase2_train_path)
    logging.info(
        "Phase 2 loaded: %d samples (%.2fs)",
        len(phase2_samples),
        time.time() - phase2_load_start,
    )

    processed = 0
    phase1_process_start = time.time()
    processed, phase1_ner_samples = process_samples(
        samples=phase1_samples,
        source_name="phase1",
        type_to_idx=type_to_idx,
        type_to_idx_casefold=type_to_idx_casefold,
        o_idx=o_idx,
        word_entity_counts=word_entity_counts,
        start_processed=processed,
    )
    logging.info(
        "Phase 1 NER samples processed: %d (%.2fs)",
        phase1_ner_samples,
        time.time() - phase1_process_start,
    )

    phase2_process_start = time.time()
    processed, phase2_ner_samples = process_samples(
        samples=phase2_samples,
        source_name="phase2",
        type_to_idx=type_to_idx,
        type_to_idx_casefold=type_to_idx_casefold,
        o_idx=o_idx,
        word_entity_counts=word_entity_counts,
        start_processed=processed,
    )
    logging.info(
        "Phase 2 NER samples processed: %d (%.2fs)",
        phase2_ner_samples,
        time.time() - phase2_process_start,
    )

    word_entity_dist = build_distributions(word_entity_counts)

    words_with_entity_signal = 0
    for probs in word_entity_dist.values():
        if int(np.argmax(np.asarray(probs))) != o_idx:
            words_with_entity_signal += 1

    total_unique_words = len(word_entity_dist)
    vocabulary_coverage = (
        (words_with_entity_signal / total_unique_words) * 100.0
        if total_unique_words > 0
        else 0.0
    )

    per_type_top10 = {
        type_name: top_words_for_type(word_entity_dist, idx, top_k=10)
        for type_name, idx in type_to_idx.items()
        if type_name != "O"
    }

    default_dist = [0.0] * len(entity_types) + [1.0]

    stats = {
        "total_unique_words": total_unique_words,
        "words_with_entity_signal": words_with_entity_signal,
        "vocabulary_coverage": vocabulary_coverage,
        "per_type_top10": per_type_top10,
        "total_samples_processed": phase1_ner_samples + phase2_ner_samples,
        "phase1_ner_samples": phase1_ner_samples,
        "phase2_ner_samples": phase2_ner_samples,
        "distribution_dim": dist_dim,
        "default_unknown_distribution": default_dist,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dist_path = args.out_dir / "entity_distributions.json"
    type_idx_path = args.out_dir / "entity_type_index.json"
    stats_path = args.out_dir / "distribution_stats.json"

    with dist_path.open("w", encoding="utf-8") as f:
        json.dump(word_entity_dist, f, ensure_ascii=True)

    with type_idx_path.open("w", encoding="utf-8") as f:
        json.dump(type_to_idx, f, ensure_ascii=True, indent=2)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    elapsed = time.time() - start_time
    print("\n=== Build Complete ===")
    print(f"Entity distributions written to: {dist_path}")
    print(f"Entity type index written to: {type_idx_path}")
    print(f"Distribution stats written to: {stats_path}")
    print(f"Total processing time: {elapsed:.2f}s")
    print(f"Total samples processed: {phase1_ner_samples + phase2_ner_samples}")
    print(f"Phase 1 NER samples: {phase1_ner_samples}")
    print(f"Phase 2 NER samples: {phase2_ner_samples}")

    print_sanity_checks(word_entity_dist, type_to_idx, words_with_entity_signal)


if __name__ == "__main__":
    main()
