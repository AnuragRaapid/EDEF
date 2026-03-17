from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if __package__:
    _bio_mod = importlib.import_module(".bio_utils", package=__package__)
else:
    _bio_mod = importlib.import_module("encoder_ner.bio_utils")

decode_bio_labels = _bio_mod.decode_bio_labels
labels_from_offsets = _bio_mod.labels_from_offsets
resolve_entity_spans = _bio_mod.resolve_entity_spans

if __package__:
    _dist_mod = importlib.import_module("distribution_alignment")
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
load_ner_samples = _dataset_utils_mod.load_ner_samples


def _to_offset_list(offset_mapping: Any) -> list[tuple[int, int]]:
    if isinstance(offset_mapping, torch.Tensor):
        return [(int(start), int(end)) for start, end in offset_mapping.tolist()]
    if isinstance(offset_mapping, list):
        return [(int(start), int(end)) for start, end in offset_mapping]
    raise TypeError(f"Unsupported offset mapping type: {type(offset_mapping)}")


class BioTokenClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        label_to_id: dict[str, int],
        max_length: int = 2048,
        word_entity_dist: dict[str, list[float]] | None = None,
        default_dist: list[float] | None = None,
        dist_dim: int | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.word_entity_dist = word_entity_dist
        self.default_dist = default_dist
        self.dist_dim = dist_dim

        self.use_edef = (
            self.word_entity_dist is not None
            and self.default_dist is not None
            and self.dist_dim is not None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = str(sample.get("text", sample.get("input", "")))
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = _to_offset_list(encoding["offset_mapping"].squeeze(0))
        entity_spans = resolve_entity_spans(sample)
        label_ids = labels_from_offsets(
            text, offset_mapping, entity_spans, self.label_to_id
        )

        labels = torch.tensor(label_ids[: len(input_ids)], dtype=torch.long)
        item: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if self.use_edef:
            assert self.word_entity_dist is not None
            assert self.default_dist is not None
            assert self.dist_dim is not None

            dist_vectors = get_token_distributions(
                text,
                self.tokenizer,
                self.word_entity_dist,
                self.default_dist,
                self.dist_dim,
            )
            if len(dist_vectors) > len(input_ids):
                dist_vectors = dist_vectors[: len(input_ids)]
            elif len(dist_vectors) < len(input_ids):
                pad_rows = len(input_ids) - len(dist_vectors)
                padding = torch.zeros(pad_rows, self.dist_dim, dtype=dist_vectors.dtype)
                dist_vectors = torch.cat([dist_vectors, padding], dim=0)
            item["entity_dist_vectors"] = dist_vectors

        return item


class BioDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.max_length = max_length

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]
        has_dist = all("entity_dist_vectors" in feature for feature in features)
        dist_vectors = (
            [feature["entity_dist_vectors"] for feature in features]
            if has_dist
            else None
        )

        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        padded_input_ids: list[torch.Tensor] = []
        padded_attention_masks: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        padded_dist_vectors: list[torch.Tensor] = []

        for item_idx, (ids, mask, item_labels) in enumerate(
            zip(input_ids, attention_masks, labels)
        ):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_input_ids.append(
                    F.pad(ids, (0, pad_len), value=self.pad_token_id)
                )
                padded_attention_masks.append(F.pad(mask, (0, pad_len), value=0))
                padded_labels.append(F.pad(item_labels, (0, pad_len), value=-100))
            elif pad_len < 0:
                padded_input_ids.append(ids[:max_len])
                padded_attention_masks.append(mask[:max_len])
                padded_labels.append(item_labels[:max_len])
            else:
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)
                padded_labels.append(item_labels)

            if dist_vectors is None:
                continue
            current_dist = dist_vectors[item_idx]
            if pad_len > 0:
                padded_dist_vectors.append(
                    F.pad(current_dist, (0, 0, 0, pad_len), value=0.0)
                )
            elif pad_len < 0:
                padded_dist_vectors.append(current_dist[:max_len])
            else:
                padded_dist_vectors.append(current_dist)

        batch: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "labels": torch.stack(padded_labels),
        }
        if dist_vectors is not None:
            batch["entity_dist_vectors"] = torch.stack(padded_dist_vectors)
        return batch


def build_bio_dataset(
    data_path: str,
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int = 2048,
    dist_path: str | None = None,
    dataset_split: str | None = None,
    dataset_revision: str | None = None,
    cache_dir: str | None = None,
    instruction: str | None = None,
) -> BioTokenClassificationDataset:
    samples = load_ner_samples(
        data_path,
        split=dataset_split,
        dataset_revision=dataset_revision,
        cache_dir=cache_dir,
        instruction=instruction,
    )

    word_entity_dist = None
    default_dist = None
    dist_dim = None
    if dist_path:
        word_entity_dist, default_dist = load_distributions(dist_path)
        dist_dim = len(default_dist)

    return BioTokenClassificationDataset(
        samples=samples,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        dist_dim=dist_dim,
    )
