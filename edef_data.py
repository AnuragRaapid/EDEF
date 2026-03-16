from __future__ import annotations

import argparse
import importlib
import json
import re
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

if __package__:
    _dist_mod = importlib.import_module(".distribution_alignment", package=__package__)
    _medical_mod = importlib.import_module(".medical_alignment", package=__package__)
    _model_mod = importlib.import_module(".edef_model", package=__package__)
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _medical_mod = importlib.import_module("medical_alignment")
    _model_mod = importlib.import_module("edef_model")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
build_medical_alignment_features = _medical_mod.build_medical_alignment_features
DEFAULT_MEDICAL_CHUNK_OVERLAP = _model_mod.DEFAULT_MEDICAL_CHUNK_OVERLAP
DEFAULT_MEDICAL_CHUNK_SIZE = _model_mod.DEFAULT_MEDICAL_CHUNK_SIZE
DEFAULT_MAX_PROMPT_MEDICAL_TOKENS = _model_mod.DEFAULT_MAX_PROMPT_MEDICAL_TOKENS
DEFAULT_SIGNAL_SOURCE = _model_mod.DEFAULT_SIGNAL_SOURCE

if __package__:
    _dataset_utils_mod = importlib.import_module(
        ".ner_dataset_utils", package=__package__
    )
else:
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

load_ner_samples = _dataset_utils_mod.load_ner_samples


class EDEFDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        *,
        signal_source: str = DEFAULT_SIGNAL_SOURCE,
        word_entity_dist: dict[str, list[float]] | None = None,
        default_dist: list[float] | None = None,
        medical_tokenizer: Any | None = None,
        max_length: int = 4096,
        dist_dim: int | None = None,
        medical_chunk_size: int = DEFAULT_MEDICAL_CHUNK_SIZE,
        medical_chunk_overlap: int = DEFAULT_MEDICAL_CHUNK_OVERLAP,
        max_prompt_medical_tokens: int = DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
        chat_template_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.signal_source = str(signal_source).strip().lower()
        self.word_entity_dist = word_entity_dist or {}
        self.default_dist = default_dist or []
        self.medical_tokenizer = medical_tokenizer
        self.max_length = max_length
        self.dist_dim = dist_dim if dist_dim is not None else len(self.default_dist)
        self.medical_chunk_size = medical_chunk_size
        self.medical_chunk_overlap = medical_chunk_overlap
        self.max_prompt_medical_tokens = max_prompt_medical_tokens
        self.chat_template_fn = chat_template_fn

        if self.signal_source == "distribution":
            if not self.default_dist:
                raise ValueError(
                    "Distribution mode requires default_dist to be provided."
                )
        elif self.signal_source == "medical_encoder":
            if self.medical_tokenizer is None:
                raise ValueError(
                    "Medical encoder mode requires medical_tokenizer to be provided."
                )
        else:
            raise ValueError(f"Unsupported signal_source={self.signal_source}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False)

        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts) + "\n"

    def _format_sample(
        self, sample: dict[str, Any], include_output: bool = True
    ) -> str:
        if self.chat_template_fn is not None:
            return self.chat_template_fn(sample)

        instruction = str(sample.get("instruction", ""))
        user_input = str(sample.get("input", ""))
        output = str(sample.get("output", "")) if include_output else ""

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output},
        ]
        return self._apply_chat_template(messages)

    def _build_labels(
        self,
        sample: dict[str, Any],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        labels = input_ids.clone()

        prompt_text = self._format_sample(sample, include_output=False)
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        if prompt_len >= len(labels):
            output_text = str(sample.get("output", ""))
            output_encoding = self.tokenizer(
                output_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            output_ids = output_encoding["input_ids"].squeeze(0)
            output_len = min(len(output_ids), len(labels))
            labels[:] = -100
            if output_len > 0:
                labels[-output_len:] = input_ids[-output_len:]
        return labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = self._format_sample(sample)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = self._build_labels(sample, input_ids)

        item: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if self.signal_source == "distribution":
            dist_vectors = get_token_distributions(
                text,
                self.tokenizer,
                self.word_entity_dist,
                self.default_dist,
                self.dist_dim,
            )
            num_tokens = len(input_ids)
            if len(dist_vectors) > num_tokens:
                dist_vectors = dist_vectors[:num_tokens]
            elif len(dist_vectors) < num_tokens:
                pad_rows = num_tokens - len(dist_vectors)
                padding = torch.zeros(pad_rows, self.dist_dim, dtype=dist_vectors.dtype)
                dist_vectors = torch.cat([dist_vectors, padding], dim=0)
            item["entity_dist_vectors"] = dist_vectors
            return item

        clinical_text = str(sample.get("input", ""))
        medical_features = build_medical_alignment_features(
            prompt_text=text,
            clinical_text=clinical_text,
            prompt_tokenizer=self.tokenizer,
            medical_tokenizer=self.medical_tokenizer,
            prompt_max_length=self.max_length,
            chunk_size=self.medical_chunk_size,
            chunk_overlap=self.medical_chunk_overlap,
            max_prompt_medical_tokens=self.max_prompt_medical_tokens,
        )
        item.update(medical_features)
        return item


class EDEFDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.max_length = max_length

    def _pad_sequence_1d(
        self,
        tensors: list[torch.Tensor],
        *,
        max_len: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            current = tensor[:max_len]
            pad_len = max_len - current.shape[0]
            if pad_len > 0:
                current = F.pad(current, (0, pad_len), value=pad_value)
            padded.append(current)
        return torch.stack(padded)

    def _pad_chunks(
        self,
        tensors: list[torch.Tensor],
        *,
        pad_value: int | float,
    ) -> torch.Tensor:
        max_chunks = max(t.shape[0] for t in tensors)
        max_chunk_len = max(t.shape[1] for t in tensors)
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            out = torch.full(
                (max_chunks, max_chunk_len),
                fill_value=pad_value,
                dtype=tensor.dtype,
            )
            out[: tensor.shape[0], : tensor.shape[1]] = tensor
            padded.append(out)
        return torch.stack(padded)

    def _pad_prompt_alignment(
        self,
        tensors: list[torch.Tensor],
        *,
        max_len: int,
        pad_value: int | float,
    ) -> torch.Tensor:
        max_overlap = max(t.shape[1] for t in tensors)
        padded: list[torch.Tensor] = []
        for tensor in tensors:
            current = tensor[:max_len]
            out = torch.full(
                (max_len, max_overlap),
                fill_value=pad_value,
                dtype=tensor.dtype,
            )
            out[: current.shape[0], : current.shape[1]] = current
            padded.append(out)
        return torch.stack(padded)

    def __call__(
        self, features: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        batch = {
            "input_ids": self._pad_sequence_1d(
                input_ids,
                max_len=max_len,
                pad_value=self.pad_token_id,
            ),
            "attention_mask": self._pad_sequence_1d(
                attention_mask,
                max_len=max_len,
                pad_value=0,
            ),
            "labels": self._pad_sequence_1d(
                labels,
                max_len=max_len,
                pad_value=-100,
            ),
        }

        if "entity_dist_vectors" in features[0]:
            dist_vectors = [f["entity_dist_vectors"] for f in features]
            padded_dists: list[torch.Tensor] = []
            for dists in dist_vectors:
                current = dists[:max_len]
                pad_len = max_len - current.shape[0]
                if pad_len > 0:
                    current = F.pad(current, (0, 0, 0, pad_len), value=0.0)
                padded_dists.append(current)
            batch["entity_dist_vectors"] = torch.stack(padded_dists)
            return batch

        batch["medical_chunk_input_ids"] = self._pad_chunks(
            [f["medical_chunk_input_ids"] for f in features],
            pad_value=self.pad_token_id,
        )
        batch["medical_chunk_attention_mask"] = self._pad_chunks(
            [f["medical_chunk_attention_mask"] for f in features],
            pad_value=0,
        )
        batch["medical_chunk_token_indices"] = self._pad_chunks(
            [f["medical_chunk_token_indices"] for f in features],
            pad_value=-1,
        )
        batch["medical_chunk_token_weights"] = self._pad_chunks(
            [f["medical_chunk_token_weights"] for f in features],
            pad_value=0.0,
        )
        batch["prompt_medical_token_indices"] = self._pad_prompt_alignment(
            [f["prompt_medical_token_indices"] for f in features],
            max_len=max_len,
            pad_value=-1,
        )
        batch["prompt_medical_token_weights"] = self._pad_prompt_alignment(
            [f["prompt_medical_token_weights"] for f in features],
            max_len=max_len,
            pad_value=0.0,
        )
        batch["medical_token_count"] = torch.stack(
            [f["medical_token_count"] for f in features]
        )
        return batch


def build_edef_dataset(
    data_path: str,
    tokenizer: Any,
    dist_path: str | None = None,
    *,
    signal_source: str = DEFAULT_SIGNAL_SOURCE,
    medical_tokenizer: Any | None = None,
    max_length: int = 4096,
    dist_dim: int | None = None,
    medical_chunk_size: int = DEFAULT_MEDICAL_CHUNK_SIZE,
    medical_chunk_overlap: int = DEFAULT_MEDICAL_CHUNK_OVERLAP,
    max_prompt_medical_tokens: int = DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
    chat_template_fn: Callable[[dict[str, Any]], str] | None = None,
    dataset_split: str | None = None,
    dataset_revision: str | None = None,
    cache_dir: str | None = None,
    instruction: str | None = None,
) -> EDEFDataset:
    samples = load_ner_samples(
        data_path,
        split=dataset_split,
        dataset_revision=dataset_revision,
        cache_dir=cache_dir,
        instruction=instruction,
    )

    signal_source = str(signal_source).strip().lower()
    word_entity_dist: dict[str, list[float]] | None = None
    default_dist: list[float] | None = None
    effective_dist_dim = dist_dim
    if signal_source == "distribution":
        if not dist_path:
            raise ValueError("Distribution mode requires dist_path.")
        word_entity_dist, default_dist = load_distributions(dist_path)
        effective_dist_dim = len(default_dist) if dist_dim is None else dist_dim
        if effective_dist_dim != len(default_dist):
            raise ValueError(
                f"dist_dim={effective_dist_dim} does not match distribution file dimension={len(default_dist)}"
            )

    return EDEFDataset(
        samples=samples,
        tokenizer=tokenizer,
        signal_source=signal_source,
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        medical_tokenizer=medical_tokenizer,
        max_length=max_length,
        dist_dim=effective_dist_dim,
        medical_chunk_size=medical_chunk_size,
        medical_chunk_overlap=medical_chunk_overlap,
        max_prompt_medical_tokens=max_prompt_medical_tokens,
        chat_template_fn=chat_template_fn,
    )


class _MockTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self._token_to_id: dict[str, int] = {"<pad>": 0}
        self._id_to_token: dict[int, str] = {0: "<pad>"}

    def _get_id(self, token: str) -> int:
        if token not in self._token_to_id:
            next_id = len(self._token_to_id)
            self._token_to_id[token] = next_id
            self._id_to_token[next_id] = token
        return self._token_to_id[token]

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        del tokenize, add_generation_prompt
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        return "\n".join(parts) + "\n"

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens
        matches = list(re.finditer(r"\S+", text))
        tokens = [m.group(0) for m in matches]
        offsets = [(m.start(), m.end()) for m in matches]

        if truncation and max_length is not None:
            tokens = tokens[:max_length]
            offsets = offsets[:max_length]

        input_ids = [self._get_id(tok) for tok in tokens]
        attention_mask = [1] * len(input_ids)
        out: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
            out["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return out

    def prepare_for_model(
        self,
        ids: list[int],
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
        truncation: bool = False,
    ) -> dict[str, Any]:
        del return_token_type_ids, truncation
        input_ids = list(ids)
        if add_special_tokens:
            input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        output: dict[str, Any] = {"input_ids": input_ids}
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
            if token_id in {self.pad_token_id, self.cls_token_id, self.sep_token_id}
            else 0
            for token_id in token_ids_0
        ]

    def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]:
        return [self._id_to_token.get(i, f"id_{i}") for i in input_ids]


def _smoke_test(
    *,
    data_path: str,
    dist_path: str | None,
    tokenizer_name: str,
    medical_tokenizer_name: str | None,
    max_length: int,
    dist_dim: int,
    signal_source: str,
) -> None:
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[:3]

    tokenizer: Any
    medical_tokenizer: Any | None = None
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded tokenizer: {tokenizer_name}")

        if signal_source == "medical_encoder":
            med_name = medical_tokenizer_name or tokenizer_name
            medical_tokenizer = AutoTokenizer.from_pretrained(
                med_name, trust_remote_code=True
            )
            print(f"Loaded medical tokenizer: {med_name}")
    except Exception as exc:
        tokenizer = _MockTokenizer()
        medical_tokenizer = _MockTokenizer()
        print(f"transformers tokenizer unavailable ({exc}); using mock tokenizers")

    kwargs: dict[str, Any] = {
        "samples": samples,
        "tokenizer": tokenizer,
        "signal_source": signal_source,
        "max_length": max_length,
    }
    if signal_source == "distribution":
        if dist_path:
            word_entity_dist, default_dist = load_distributions(dist_path)
            print(f"Loaded distributions from: {dist_path}")
        else:
            word_entity_dist = {
                "pain": [0.0] * 24 + [0.9] + [0.0] * 19 + [0.1],
                "chest": [0.0] * 18 + [0.8] + [0.0] * 25 + [0.2],
                "sodium": [0.0] * 22 + [0.85] + [0.0] * 21 + [0.15],
            }
            default_dist = [0.0] * (dist_dim - 1) + [1.0]
            print("Using mock distributions")
        kwargs["word_entity_dist"] = word_entity_dist
        kwargs["default_dist"] = default_dist
        kwargs["dist_dim"] = dist_dim
    else:
        kwargs["medical_tokenizer"] = medical_tokenizer
        kwargs["medical_chunk_size"] = 8
        kwargs["medical_chunk_overlap"] = 2
        kwargs["max_prompt_medical_tokens"] = 3

    dataset = EDEFDataset(**kwargs)
    sample0 = dataset[0]
    print("\nSingle sample checks:")
    print(f"input_ids: {tuple(sample0['input_ids'].shape)}")
    print(f"attention_mask: {tuple(sample0['attention_mask'].shape)}")
    print(f"labels: {tuple(sample0['labels'].shape)}")
    if signal_source == "distribution":
        print(f"entity_dist_vectors: {tuple(sample0['entity_dist_vectors'].shape)}")
    else:
        print(
            f"medical_chunk_input_ids: {tuple(sample0['medical_chunk_input_ids'].shape)}"
        )
        print(
            "prompt_medical_token_indices:",
            tuple(sample0["prompt_medical_token_indices"].shape),
        )
        print(f"medical_token_count: {int(sample0['medical_token_count'].item())}")

    collator = EDEFDataCollator(tokenizer=tokenizer, max_length=max_length)
    batch = collator([dataset[0], dataset[1]])
    print("\nBatch checks:")
    for key, value in batch.items():
        print(f"{key}: {tuple(value.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test for EDEFDataset and EDEFDataCollator"
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to Alpaca-format JSON data"
    )
    parser.add_argument(
        "--signal_source",
        default="distribution",
        choices=["distribution", "medical_encoder"],
    )
    parser.add_argument(
        "--dist-path",
        default=None,
        help="Path to entity distribution JSON (optional; uses mock distributions if omitted)",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="Qwen/Qwen3-4B-Instruct",
        help="Tokenizer repo id for smoke test",
    )
    parser.add_argument(
        "--medical-tokenizer-name",
        default=None,
        help="Medical tokenizer repo id for smoke test",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dist-dim", type=int, default=45)
    args = parser.parse_args()

    _smoke_test(
        data_path=args.data_path,
        dist_path=args.dist_path,
        tokenizer_name=args.tokenizer_name,
        medical_tokenizer_name=args.medical_tokenizer_name,
        max_length=args.max_length,
        dist_dim=args.dist_dim,
        signal_source=args.signal_source,
    )
