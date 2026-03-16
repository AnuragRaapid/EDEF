from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET_NAME = "anurag-raapid/chia"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PHASE1_MODEL_PATH = str(PROJECT_ROOT / "qwen3-phase1-checkpoint")
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "chia"
DEFAULT_DIST_PATH = str(DEFAULT_ARTIFACT_DIR / "entity_distributions.json")

STRUCTURED_OUTPUT_FORMAT = "layered_inline_v1"
NER_ROOT_START = "<NER>"
NER_ROOT_END = "</NER>"
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_LAYER_TAG_RE = re.compile(
    r"(<NER>|</NER>|<S(?P<start_layer>\d+)_(?P<start_label>[^>]+)>|</S(?P<end_layer>\d+)>)"
)


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


def _normalize_model_output_text(text: str) -> str:
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = re.sub(r"^assistant\s*", "", cleaned, flags=re.IGNORECASE).strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        else:
            cleaned = "\n".join(lines[1:]).strip()

    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _iter_clean_entities(entities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entity in entities:
        text = str(entity.get("text", "")).strip()
        entity_type = str(entity.get("type", entity.get("entity_type", ""))).strip()
        if not text or not entity_type:
            continue

        normalized = {
            "text": text,
            "type": entity_type,
            "char_start": _maybe_int(entity.get("char_start")),
            "char_end": _maybe_int(entity.get("char_end")),
            "token_start": _maybe_int(entity.get("token_start")),
            "token_end": _maybe_int(entity.get("token_end")),
        }
        dedupe_key = (
            normalized["text"],
            normalized["type"],
            normalized["char_start"],
            normalized["char_end"],
            normalized["token_start"],
            normalized["token_end"],
        )
        unique[dedupe_key] = normalized
    return list(unique.values())


def _layer_entities_for_text(
    source_text: str,
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    layered: list[dict[str, Any]] = []
    sorted_entities = sorted(
        entities,
        key=lambda entity: (
            int(entity["char_start"]),
            -int(entity["char_end"]),
            str(entity["type"]).casefold(),
            str(entity["text"]).casefold(),
        ),
    )

    layer_ends: list[int] = []
    for entity in sorted_entities:
        char_start = _maybe_int(entity.get("char_start"))
        char_end = _maybe_int(entity.get("char_end"))
        if (
            char_start is None
            or char_end is None
            or char_start < 0
            or char_end <= char_start
            or char_end > len(source_text)
        ):
            continue

        selected_layer = None
        for layer, layer_end in enumerate(layer_ends):
            if char_start >= layer_end:
                selected_layer = layer
                layer_ends[layer] = char_end
                break

        if selected_layer is None:
            selected_layer = len(layer_ends)
            layer_ends.append(char_end)

        layered.append(
            {
                **entity,
                "char_start": char_start,
                "char_end": char_end,
                "layer": selected_layer,
            }
        )

    return layered


def _build_layered_inline_output(
    source_text: str,
    entities: list[dict[str, Any]],
) -> str:
    layered_entities = _layer_entities_for_text(source_text, entities)
    opens: dict[int, list[dict[str, Any]]] = defaultdict(list)
    closes: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for entity in layered_entities:
        char_start = int(entity["char_start"])
        char_end = int(entity["char_end"])
        opens[char_start].append(entity)
        closes[char_end].append(entity)

    parts: list[str] = [NER_ROOT_START]
    for char_pos in range(len(source_text)):
        if char_pos in opens:
            for entity in sorted(
                opens[char_pos],
                key=lambda item: (
                    int(item["layer"]),
                    -int(item["char_end"]),
                    str(item["type"]).casefold(),
                ),
            ):
                parts.append(f"<S{int(entity['layer'])}_{str(entity['type']).strip()}>")

        parts.append(source_text[char_pos])

        close_pos = char_pos + 1
        if close_pos in closes:
            for entity in sorted(
                closes[close_pos],
                key=lambda item: (
                    -int(item["layer"]),
                    int(item["char_start"]),
                    str(item["type"]).casefold(),
                ),
            ):
                parts.append(f"</S{int(entity['layer'])}>")

    parts.append(NER_ROOT_END)
    return "".join(parts)


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

    return _iter_clean_entities(entities)


def _parse_json_payload(output_text: str) -> list[dict[str, Any]]:
    candidates = [output_text]
    if '{"ner"' in output_text:
        start = output_text.find('{"ner"')
        end = output_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(output_text[start : end + 1])
    if "{'ner'" in output_text:
        start = output_text.find("{'ner'")
        end = output_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(output_text[start : end + 1])

    for candidate in candidates:
        try:
            payload = _parse_json_like(candidate)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

        if not isinstance(payload, dict):
            continue

        ner_items = payload.get("ner")
        if not isinstance(ner_items, list):
            continue

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

        if entities:
            return _iter_clean_entities(entities)
        return []

    return []


def _parse_layered_inline_payload(
    output_text: str,
    source_text: str | None = None,
) -> list[dict[str, Any]]:
    ner_start = output_text.find(NER_ROOT_START)
    if ner_start != -1:
        ner_end = output_text.find(NER_ROOT_END, ner_start)
        if ner_end != -1:
            output_text = output_text[ner_start : ner_end + len(NER_ROOT_END)]
        else:
            output_text = output_text[ner_start:]

    matches = list(_LAYER_TAG_RE.finditer(output_text))
    if not matches:
        return []

    plain_fragments: list[str] = []
    open_layers: dict[int, tuple[str, int]] = {}
    closed_spans: list[tuple[int, int, str]] = []
    cursor = 0
    plain_len = 0

    for match in matches:
        fragment = output_text[cursor : match.start()]
        if fragment:
            plain_fragments.append(fragment)
            plain_len += len(fragment)

        start_layer = match.group("start_layer")
        start_label = match.group("start_label")
        end_layer = match.group("end_layer")

        if start_layer is not None and start_label is not None:
            layer = int(start_layer)
            if layer not in open_layers:
                open_layers[layer] = (start_label.strip(), plain_len)
        elif end_layer is not None:
            layer = int(end_layer)
            if layer in open_layers:
                label, start = open_layers.pop(layer)
                if plain_len > start:
                    closed_spans.append((start, plain_len, label))

        cursor = match.end()

    trailing_fragment = output_text[cursor:]
    if trailing_fragment:
        plain_fragments.append(trailing_fragment)

    plain_text = "".join(plain_fragments)
    source_casefold = source_text.casefold() if source_text else None
    entities: list[dict[str, Any]] = []
    for start, end, label in sorted(
        closed_spans, key=lambda item: (item[0], item[1], item[2].casefold())
    ):
        entity_text = plain_text[start:end].strip()
        if not entity_text or not label.strip():
            continue
        if (
            source_casefold is not None
            and entity_text.casefold() not in source_casefold
        ):
            continue
        entities.append({"text": entity_text, "type": label.strip()})

    return _iter_clean_entities(entities)


def parse_output_payload(
    output_value: Any,
    source_text: str | None = None,
) -> list[dict[str, Any]]:
    if isinstance(output_value, str):
        stripped = _normalize_model_output_text(output_value)
        if not stripped:
            return []

        if "<S" in stripped or NER_ROOT_START in stripped:
            inline_entities = _parse_layered_inline_payload(
                stripped, source_text=source_text
            )
            if inline_entities:
                return inline_entities

        json_entities = _parse_json_payload(stripped)
        if json_entities:
            return json_entities

        return []

    return _parse_json_payload(json.dumps(output_value, ensure_ascii=False))


def entity_records_to_output_text(
    entities: Iterable[dict[str, Any]],
    source_text: str | None = None,
) -> str:
    clean_entities = _iter_clean_entities(entities)
    if source_text and all(
        entity.get("char_start") is not None and entity.get("char_end") is not None
        for entity in clean_entities
    ):
        return _build_layered_inline_output(source_text, clean_entities)

    payload = {
        "ner": [
            [str(entity.get("text", "")).strip(), str(entity.get("type", "")).strip()]
            for entity in clean_entities
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
    label_text = (
        ", ".join(entity_types) if entity_types else "the supported entity types"
    )
    return (
        "You are an expert biomedical Named Entity Recognition (NER) assistant. "
        "Copy the input text exactly once inside a single <NER>...</NER> block and annotate every entity mention "
        f"using exactly one of the following entity types: {label_text}. "
        "Open a span with a tag like <S0_Condition> and close the same layer with a tag like </S0>. "
        "Use the lowest available layer numbers starting from 0, and put overlapping or nested spans on different layers. "
        "Do not paraphrase, normalize, reorder, or omit any source text. "
        "If there are no entities, return the unchanged input text wrapped in <NER>...</NER>."
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

    raw_output_format = str(stats.get("output_format", "")).strip()
    output_format = raw_output_format or STRUCTURED_OUTPUT_FORMAT
    instruction = str(stats.get("prompt_instruction", "")).strip()
    if not instruction or raw_output_format != STRUCTURED_OUTPUT_FORMAT:
        instruction = build_ner_instruction(entity_types)
        output_format = STRUCTURED_OUTPUT_FORMAT

    return {
        "entity_types": entity_types,
        "entity_type_to_idx": entity_type_to_idx,
        "dist_dim": dist_dim,
        "o_index": o_index,
        "default_dist": default_dist,
        "instruction": instruction,
        "output_format": output_format,
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
                    "entities": parse_output_payload(
                        output_text, source_text=input_text
                    ),
                }
            )
            continue

        if "text" in sample and ("entity" in sample or "entities" in sample):
            entities = parse_entity_records(
                sample.get("entity", sample.get("entities", []))
            )
            source_text = str(sample.get("text", ""))
            tokens = sample.get("tokens")
            converted_sample: dict[str, Any] = {
                "instruction": str(instruction or sample.get("instruction", "")),
                "input": source_text,
                "output": entity_records_to_output_text(
                    entities, source_text=source_text
                ),
                "text": source_text,
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
