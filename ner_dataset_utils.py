from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_NAME = "anurag-raapid/chia"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE1_MODEL_PATH = str(PROJECT_ROOT / "qwen3-phase1-checkpoint")
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "ncbi_disease"
DEFAULT_DIST_PATH = str(DEFAULT_ARTIFACT_DIR / "entity_distributions.json")


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _parse_json_like(value: str) -> Any:
    text = value.strip()
    if not text:
        return []

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def parse_entity_records(raw_entities: Any) -> list[dict[str, Any]]:
    if raw_entities is None:
        return []

    parsed = raw_entities
    if isinstance(raw_entities, str):
        parsed = _parse_json_like(raw_entities)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected entity list, got {type(parsed)}")

    entities: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        entity_type = str(item.get("type", item.get("entity_type", ""))).strip()
        if not text or not entity_type:
            continue

        entities.append(
            {
                "text": text,
                "type": entity_type,
                "char_start": _maybe_int(item.get("char_start")),
                "char_end": _maybe_int(item.get("char_end")),
                "token_start": _maybe_int(item.get("token_start")),
                "token_end": _maybe_int(item.get("token_end")),
            }
        )

    return entities


def parse_output_payload(output_value: Any) -> list[dict[str, Any]]:
    payload = output_value
    if isinstance(output_value, str):
        stripped = output_value.strip()
        if not stripped:
            return []
        try:
            payload = _parse_json_like(stripped)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            return []

    if not isinstance(payload, dict):
        return []

    ner_items = payload.get("ner")
    if not isinstance(ner_items, list):
        return []

    entities: list[dict[str, Any]] = []
    for item in ner_items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            entity_type = str(item.get("type", item.get("entity_type", ""))).strip()
            if text and entity_type:
                entities.append({"text": text, "type": entity_type})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[0]).strip()
            entity_type = str(item[1]).strip()
            if text and entity_type:
                entities.append({"text": text, "type": entity_type})

    return entities


def entity_records_to_output_text(entities: Iterable[dict[str, Any]]) -> str:
    payload = {
        "ner": [
            [str(entity.get("text", "")).strip(), str(entity.get("type", "")).strip()]
            for entity in entities
            if str(entity.get("text", "")).strip()
            and str(entity.get("type", "")).strip()
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def extract_entity_types_from_samples(samples: Iterable[dict[str, Any]]) -> list[str]:
    entity_types = {
        str(entity.get("type", "")).strip()
        for sample in samples
        for entity in sample.get("entities", [])
        if str(entity.get("type", "")).strip()
    }
    return sorted(entity_types, key=str.casefold)


def build_ner_instruction(entity_types: list[str]) -> str:
    label_text = ", ".join(entity_types) if entity_types else "the supported entity types"
    return (
        "You are an expert biomedical Named Entity Recognition (NER) assistant. "
        "Extract every entity mention from the provided biomedical text and classify each mention "
        f"using exactly one of the following entity types: {label_text}. "
        "Use the exact surface form from the input text, do not normalize or paraphrase entities, "
        'and return only valid JSON in the exact format {"ner": [["entity text", "EntityType"], ...]}. '
        'If there are no entities, return {"ner": []}.'
    )


def _load_entity_type_index(index_path: Path) -> dict[str, int]:
    with index_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {index_path}")

    mapping: dict[str, int] = {}
    for key, value in raw.items():
        idx = _maybe_int(value)
        if idx is None:
            continue
        mapping[str(key)] = idx
    return mapping


def load_task_metadata_from_dist_path(dist_path: str) -> dict[str, Any]:
    dist_file = Path(dist_path).resolve()
    index_path = dist_file.with_name("entity_type_index.json")
    stats_path = dist_file.with_name("distribution_stats.json")

    entity_type_to_idx: dict[str, int] = {}
    entity_types: list[str] = []
    if index_path.exists():
        entity_type_to_idx = _load_entity_type_index(index_path)
        entity_types = [
            name
            for name, _ in sorted(entity_type_to_idx.items(), key=lambda item: item[1])
            if name != "O"
        ]

    stats: dict[str, Any] = {}
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            stats = loaded

    if not entity_types:
        raw_types = stats.get("entity_types", [])
        if isinstance(raw_types, list):
            entity_types = [str(item) for item in raw_types if str(item).strip()]

    default_dist = stats.get("default_unknown_distribution")
    if not isinstance(default_dist, list):
        with dist_file.open("r", encoding="utf-8") as f:
            distributions = json.load(f)
        if not isinstance(distributions, dict):
            raise ValueError(f"Expected JSON object in {dist_file}")
        first_vector = next(iter(distributions.values()), None)
        if isinstance(first_vector, list) and first_vector:
            dist_dim = len(first_vector)
        elif entity_type_to_idx:
            dist_dim = max(entity_type_to_idx.values()) + 1
        else:
            raise ValueError(
                f"Could not infer distribution dimension from {dist_file}; missing metadata files."
            )
        o_index = entity_type_to_idx.get("O", len(entity_types))
        default_dist = [0.0] * dist_dim
        if 0 <= o_index < dist_dim:
            default_dist[o_index] = 1.0
    else:
        default_dist = [float(x) for x in default_dist]

    dist_dim = len(default_dist)
    o_index = entity_type_to_idx.get("O", dist_dim - 1)

    instruction = str(stats.get("prompt_instruction", "")).strip()
    if not instruction:
        instruction = build_ner_instruction(entity_types)

    return {
        "entity_types": entity_types,
        "entity_type_to_idx": entity_type_to_idx,
        "dist_dim": dist_dim,
        "o_index": o_index,
        "default_dist": default_dist,
        "instruction": instruction,
    }


def load_ner_samples(
    source: str,
    split: str | None = None,
    dataset_revision: str | None = None,
    cache_dir: str | None = None,
    instruction: str | None = None,
) -> list[dict[str, Any]]:
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Loading a dataset repo requires the `datasets` package. "
                "Install it with `uv pip install datasets` or run through `uv run --with datasets`."
            ) from exc

        dataset = load_dataset(
            source,
            split=split,
            revision=dataset_revision,
            cache_dir=cache_dir,
        )
        raw_samples = list(dataset)

    if not isinstance(raw_samples, list):
        raise ValueError(f"Expected a list of samples from {source}")

    converted: list[dict[str, Any]] = []
    for sample in raw_samples:
        if not isinstance(sample, dict):
            continue

        if "input" in sample and "output" in sample:
            output_text = sample.get("output", "")
            if isinstance(output_text, (dict, list)):
                output_text = json.dumps(output_text, ensure_ascii=False)
            else:
                output_text = str(output_text)

            input_text = str(sample.get("input", ""))
            sample_instruction = str(sample.get("instruction") or instruction or "")
            converted.append(
                {
                    **sample,
                    "instruction": sample_instruction,
                    "input": input_text,
                    "output": output_text,
                    "text": input_text,
                    "entities": parse_output_payload(output_text),
                }
            )
            continue

        if "text" in sample and ("entity" in sample or "entities" in sample):
            entities = parse_entity_records(
                sample.get("entity", sample.get("entities", []))
            )
            tokens = sample.get("tokens")
            converted_sample: dict[str, Any] = {
                "instruction": str(instruction or sample.get("instruction", "")),
                "input": str(sample.get("text", "")),
                "output": entity_records_to_output_text(entities),
                "text": str(sample.get("text", "")),
                "entities": entities,
            }
            if isinstance(tokens, list):
                converted_sample["tokens"] = [str(token) for token in tokens]
            if "filename" in sample:
                converted_sample["filename"] = str(sample.get("filename", ""))
            converted.append(converted_sample)
            continue

        raise ValueError(
            "Unsupported sample schema. Expected either Alpaca-style "
            "`instruction/input/output` rows or Hugging Face-style `text/tokens/entity` rows."
        )

    return converted
