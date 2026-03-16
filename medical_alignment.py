from __future__ import annotations

from typing import Protocol, cast

import torch


class TokenizerLike(Protocol):
    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, object]: ...

    def prepare_for_model(
        self,
        ids: list[int],
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]: ...

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        already_has_special_tokens: bool = True,
    ) -> list[int]: ...


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


def _to_int_list(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    return [_to_int(item) for item in cast(list[object], values)]


def _to_offset_list(values: object) -> list[tuple[int, int]]:
    if not isinstance(values, list):
        return []
    offsets: list[tuple[int, int]] = []
    for item in cast(list[object], values):
        if not isinstance(item, (list, tuple)):
            continue
        pair = cast(list[object] | tuple[object, ...], item)
        if len(pair) != 2:
            continue
        offsets.append((_to_int(pair[0]), _to_int(pair[1])))
    return offsets


def locate_substring_span(text: str, substring: str) -> tuple[int, int]:
    if substring == "":
        return 0, 0
    start = text.find(substring)
    if start < 0:
        return -1, -1
    return start, start + len(substring)


def build_chunk_spans(
    num_tokens: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if num_tokens <= 0:
        return [(0, 0)]

    spans: list[tuple[int, int]] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < num_tokens:
        end = min(num_tokens, start + chunk_size)
        spans.append((start, end))
        if end >= num_tokens:
            break
        start += step
    return spans


def build_center_weights(length: int) -> torch.Tensor:
    if length <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if length == 1:
        return torch.ones(1, dtype=torch.float32)
    positions = torch.arange(length, dtype=torch.float32)
    center = (length - 1) / 2.0
    denom = max(center, 1.0)
    weights = 1.0 - torch.abs(positions - center) / denom
    return torch.clamp(weights, min=0.1)


def tokenize_with_offsets(
    tokenizer: TokenizerLike,
    text: str,
    *,
    truncation: bool,
    max_length: int | None,
) -> tuple[list[int], list[tuple[int, int]]]:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=truncation,
        max_length=max_length,
    )
    input_ids = _to_int_list(encoding.get("input_ids", []))
    offsets = _to_offset_list(encoding.get("offset_mapping", []))
    return input_ids, offsets[: len(input_ids)]


def _prepare_chunk_inputs(
    tokenizer: TokenizerLike,
    chunk_token_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    prepared = tokenizer.prepare_for_model(
        chunk_token_ids,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        truncation=False,
    )
    input_ids = _to_int_list(prepared.get("input_ids", []))
    attention_mask = _to_int_list(prepared.get("attention_mask", []))
    if not attention_mask:
        attention_mask = [1] * len(input_ids)
    try:
        special_mask = tokenizer.get_special_tokens_mask(
            input_ids,
            already_has_special_tokens=True,
        )
    except Exception:
        special_mask = [0] * len(input_ids)
    if len(special_mask) != len(input_ids):
        special_mask = [0] * len(input_ids)
    return input_ids, attention_mask, special_mask


def _find_overlapping_medical_tokens(
    medical_offsets: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> list[tuple[int, float]]:
    overlaps: list[tuple[int, float]] = []
    for idx, (tok_start, tok_end) in enumerate(medical_offsets):
        if tok_end <= span_start:
            continue
        if tok_start >= span_end:
            break
        overlap = max(0, min(span_end, tok_end) - max(span_start, tok_start))
        if overlap > 0:
            overlaps.append((idx, float(overlap)))
    return overlaps


def build_medical_alignment_features(
    *,
    prompt_text: str,
    clinical_text: str,
    prompt_tokenizer: TokenizerLike,
    medical_tokenizer: TokenizerLike,
    prompt_max_length: int,
    chunk_size: int = 448,
    chunk_overlap: int = 128,
    max_prompt_medical_tokens: int = 4,
) -> dict[str, torch.Tensor]:
    prompt_input_ids, prompt_offsets = tokenize_with_offsets(
        prompt_tokenizer,
        prompt_text,
        truncation=True,
        max_length=prompt_max_length,
    )
    clinical_start, clinical_end = locate_substring_span(prompt_text, clinical_text)

    medical_token_ids, medical_offsets = tokenize_with_offsets(
        medical_tokenizer,
        clinical_text,
        truncation=False,
        max_length=None,
    )

    chunk_spans = build_chunk_spans(
        len(medical_token_ids),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunk_spans:
        chunk_spans = [(0, 0)]

    max_chunk_seq_len = 0
    prepared_chunks: list[tuple[list[int], list[int], list[int], int, int]] = []
    for start, end in chunk_spans:
        chunk_ids = medical_token_ids[start:end]
        prepared_ids, prepared_mask, special_mask = _prepare_chunk_inputs(
            medical_tokenizer,
            chunk_ids,
        )
        max_chunk_seq_len = max(max_chunk_seq_len, len(prepared_ids))
        prepared_chunks.append((prepared_ids, prepared_mask, special_mask, start, end))

    if max_chunk_seq_len <= 0:
        max_chunk_seq_len = 1

    num_chunks = len(prepared_chunks)
    medical_chunk_input_ids = torch.zeros(
        (num_chunks, max_chunk_seq_len),
        dtype=torch.long,
    )
    medical_chunk_attention_mask = torch.zeros(
        (num_chunks, max_chunk_seq_len),
        dtype=torch.long,
    )
    medical_chunk_token_indices = torch.full(
        (num_chunks, max_chunk_seq_len),
        fill_value=-1,
        dtype=torch.long,
    )
    medical_chunk_token_weights = torch.zeros(
        (num_chunks, max_chunk_seq_len),
        dtype=torch.float32,
    )

    for chunk_idx, (prepared_ids, prepared_mask, special_mask, start, end) in enumerate(
        prepared_chunks
    ):
        seq_len = len(prepared_ids)
        medical_chunk_input_ids[chunk_idx, :seq_len] = torch.tensor(
            prepared_ids,
            dtype=torch.long,
        )
        medical_chunk_attention_mask[chunk_idx, :seq_len] = torch.tensor(
            prepared_mask,
            dtype=torch.long,
        )

        chunk_len = max(0, end - start)
        content_positions = [
            pos
            for pos, is_special in enumerate(special_mask[:seq_len])
            if int(is_special) == 0
        ]
        if chunk_len > 0 and len(content_positions) == chunk_len:
            content_weights = build_center_weights(chunk_len)
            for rel_idx, pos in enumerate(content_positions):
                medical_chunk_token_indices[chunk_idx, pos] = start + rel_idx
                medical_chunk_token_weights[chunk_idx, pos] = content_weights[rel_idx]

    max_prompt_medical_tokens = max(1, int(max_prompt_medical_tokens))
    prompt_medical_token_indices = torch.full(
        (len(prompt_input_ids), max_prompt_medical_tokens),
        fill_value=-1,
        dtype=torch.long,
    )
    prompt_medical_token_weights = torch.zeros(
        (len(prompt_input_ids), max_prompt_medical_tokens),
        dtype=torch.float32,
    )

    if clinical_start >= 0 and clinical_end >= clinical_start:
        for prompt_idx, (tok_start, tok_end) in enumerate(prompt_offsets):
            if tok_end <= clinical_start or tok_start >= clinical_end:
                continue
            rel_start = max(tok_start, clinical_start) - clinical_start
            rel_end = min(tok_end, clinical_end) - clinical_start
            overlaps = _find_overlapping_medical_tokens(
                medical_offsets,
                rel_start,
                rel_end,
            )
            if not overlaps:
                continue
            overlaps = sorted(overlaps, key=lambda item: item[1], reverse=True)[
                :max_prompt_medical_tokens
            ]
            weight_sum = sum(weight for _, weight in overlaps)
            if weight_sum <= 0:
                continue
            for slot, (token_idx, weight) in enumerate(overlaps):
                prompt_medical_token_indices[prompt_idx, slot] = token_idx
                prompt_medical_token_weights[prompt_idx, slot] = weight / weight_sum

    return {
        "medical_chunk_input_ids": medical_chunk_input_ids,
        "medical_chunk_attention_mask": medical_chunk_attention_mask,
        "medical_chunk_token_indices": medical_chunk_token_indices,
        "medical_chunk_token_weights": medical_chunk_token_weights,
        "prompt_medical_token_indices": prompt_medical_token_indices,
        "prompt_medical_token_weights": prompt_medical_token_weights,
        "medical_token_count": torch.tensor(len(medical_token_ids), dtype=torch.long),
    }


class _MockTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self._token_to_id: dict[str, int] = {}

    def _get_id(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = len(self._token_to_id) + 1000
        return self._token_to_id[token]

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, object]:
        del add_special_tokens
        tokens: list[int] = []
        offsets: list[tuple[int, int]] = []
        start = 0
        for fragment in text.split():
            idx = text.find(fragment, start)
            end = idx + len(fragment)
            tokens.append(self._get_id(fragment.lower()))
            offsets.append((idx, end))
            start = end
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]
        output: dict[str, object] = {"input_ids": tokens}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output

    def prepare_for_model(
        self,
        ids: list[int],
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]:
        del return_token_type_ids, truncation
        input_ids = list(ids)
        if add_special_tokens:
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        output: dict[str, object] = {"input_ids": input_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(input_ids)
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        already_has_special_tokens: bool = True,
    ) -> list[int]:
        del already_has_special_tokens
        return [
            1
            if token_id in {self.cls_token_id, self.sep_token_id, self.pad_token_id}
            else 0
            for token_id in token_ids_0
        ]


if __name__ == "__main__":
    prompt_tokenizer = _MockTokenizer()
    medical_tokenizer = _MockTokenizer()

    prompt = "System prompt\nPatient has severe chest pain and fever\nassistant"
    clinical = "Patient has severe chest pain and fever"

    features = build_medical_alignment_features(
        prompt_text=prompt,
        clinical_text=clinical,
        prompt_tokenizer=prompt_tokenizer,
        medical_tokenizer=medical_tokenizer,
        prompt_max_length=128,
        chunk_size=4,
        chunk_overlap=2,
        max_prompt_medical_tokens=3,
    )

    assert features["medical_chunk_input_ids"].dim() == 2
    assert (
        features["medical_chunk_attention_mask"].shape
        == features["medical_chunk_input_ids"].shape
    )
    assert (
        features["medical_chunk_token_indices"].shape
        == features["medical_chunk_input_ids"].shape
    )
    assert (
        features["medical_chunk_token_weights"].shape
        == features["medical_chunk_input_ids"].shape
    )
    assert features["prompt_medical_token_indices"].shape[0] <= 128
    assert features["prompt_medical_token_indices"].shape[1] == 3
    print("medical_alignment smoke test passed")
