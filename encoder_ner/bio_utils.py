from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

LABEL_METADATA_FILENAME = "encoder_ner_config.json"


def build_bio_label_list(entity_types: list[str]) -> list[str]:
    ordered_types = [
        str(entity_type).strip()
        for entity_type in entity_types
        if str(entity_type).strip()
    ]
    labels = ["O"]
    for entity_type in ordered_types:
        labels.append(f"B-{entity_type}")
        labels.append(f"I-{entity_type}")
    return labels


def build_label_mappings(
    entity_types: list[str],
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    label_list = build_bio_label_list(entity_types)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for idx, label in enumerate(label_list)}
    return label_list, label_to_id, id_to_label


def save_label_metadata(
    output_dir: str | Path,
    entity_types: list[str],
    label_list: list[str],
    extra_config: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "entity_types": entity_types,
        "label_list": label_list,
    }
    if extra_config:
        payload.update(extra_config)

    output_path = Path(output_dir) / LABEL_METADATA_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_path


def load_label_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {metadata_path}")
    return payload


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _find_text_span(
    text: str,
    entity_text: str,
    occupied_spans: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if not entity_text:
        return None

    escaped = re.escape(entity_text)
    for match in re.finditer(escaped, text, flags=re.IGNORECASE):
        span = (match.start(), match.end())
        if not any(_spans_overlap(span, existing) for existing in occupied_spans):
            return span

    normalized_text = _normalize_whitespace(text).casefold()
    normalized_entity = _normalize_whitespace(entity_text).casefold()
    if not normalized_entity:
        return None
    start = normalized_text.find(normalized_entity)
    if start == -1:
        return None

    # Fall back to a raw-text search token by token so we can still derive a span
    # when whitespace in the labels is slightly inconsistent with the source text.
    entity_words = [word for word in re.split(r"\s+", entity_text.strip()) if word]
    if not entity_words:
        return None

    cursor = 0
    first_start: int | None = None
    last_end: int | None = None
    lower_text = text.casefold()
    for word in entity_words:
        word_start = lower_text.find(word.casefold(), cursor)
        if word_start == -1:
            return None
        word_end = word_start + len(word)
        if first_start is None:
            first_start = word_start
        last_end = word_end
        cursor = word_end

    if first_start is None or last_end is None:
        return None
    span = (first_start, last_end)
    if any(_spans_overlap(span, existing) for existing in occupied_spans):
        return None
    return span


def resolve_entity_spans(sample: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(sample.get("text", sample.get("input", "")))
    raw_entities = sample.get("entities", [])
    if not isinstance(raw_entities, list):
        return []

    occupied_spans: list[tuple[int, int]] = []
    resolved: list[dict[str, Any]] = []

    sortable_entities: list[dict[str, Any]] = [
        entity for entity in raw_entities if isinstance(entity, dict)
    ]
    sortable_entities.sort(
        key=lambda entity: (
            entity.get("char_start") is None,
            int(entity.get("char_start", 10**9) or 10**9),
            -len(str(entity.get("text", ""))),
        )
    )

    for entity in sortable_entities:
        entity_type = str(entity.get("type", "")).strip()
        entity_text = str(entity.get("text", "")).strip()
        if not entity_type or not entity_text:
            continue

        char_start = entity.get("char_start")
        char_end = entity.get("char_end")
        span: tuple[int, int] | None = None
        if isinstance(char_start, int) and isinstance(char_end, int):
            if 0 <= char_start < char_end <= len(text):
                span = (char_start, char_end)

        if span is None:
            span = _find_text_span(text, entity_text, occupied_spans)
        if span is None:
            continue
        if any(_spans_overlap(span, existing) for existing in occupied_spans):
            continue

        occupied_spans.append(span)
        resolved.append(
            {
                "text": text[span[0] : span[1]] or entity_text,
                "type": entity_type,
                "start": span[0],
                "end": span[1],
            }
        )

    resolved.sort(key=lambda entity: (int(entity["start"]), int(entity["end"])))
    return resolved


def labels_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    entity_spans: list[dict[str, Any]],
    label_to_id: dict[str, int],
) -> list[int]:
    del text
    o_label_id = label_to_id["O"]
    labels = [o_label_id] * len(offsets)

    for entity in entity_spans:
        entity_type = str(entity["type"])
        begin_label = label_to_id.get(f"B-{entity_type}")
        inside_label = label_to_id.get(f"I-{entity_type}")
        if begin_label is None or inside_label is None:
            continue

        token_indices = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > start and end > int(entity["start"]) and start < int(entity["end"])
        ]
        if not token_indices:
            continue
        if any(labels[idx] != o_label_id for idx in token_indices):
            continue

        labels[token_indices[0]] = begin_label
        for token_idx in token_indices[1:]:
            labels[token_idx] = inside_label

    return labels


def decode_bio_labels(
    text: str,
    offsets: list[tuple[int, int]],
    label_ids: list[int],
    id_to_label: dict[int, str],
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    active_type: str | None = None
    active_start: int | None = None
    active_end: int | None = None

    def flush() -> None:
        nonlocal active_type, active_start, active_end
        if active_type is None or active_start is None or active_end is None:
            active_type = None
            active_start = None
            active_end = None
            return
        entity_text = text[active_start:active_end].strip()
        if entity_text:
            decoded.append(
                {
                    "text": entity_text,
                    "type": active_type,
                    "start": active_start,
                    "end": active_end,
                }
            )
        active_type = None
        active_start = None
        active_end = None

    for label_id, (start, end) in zip(label_ids, offsets):
        if end <= start or label_id < 0:
            flush()
            continue

        label_name = id_to_label.get(int(label_id), "O")
        if label_name == "O":
            flush()
            continue

        prefix, _, entity_type = label_name.partition("-")
        if not entity_type:
            flush()
            continue

        if prefix == "B" or active_type != entity_type:
            flush()
            active_type = entity_type
            active_start = start
            active_end = end
            continue

        if prefix == "I":
            active_end = end
            continue

        flush()

    flush()
    return decoded


def normalize_entity_tuples(entities: list[dict[str, Any]]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for entity in entities:
        text = str(entity.get("text", "")).lower().strip()
        entity_type = str(entity.get("type", "")).lower().strip()
        if text and entity_type:
            normalized.append((text, entity_type))
    return normalized
