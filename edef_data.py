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
else:
    _dist_mod = importlib.import_module("distribution_alignment")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions

if __package__:
    _dataset_utils_mod = importlib.import_module(".ner_dataset_utils", package=__package__)
else:
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

load_ner_samples = _dataset_utils_mod.load_ner_samples


class EDEFDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        tokenizer: Any,
        word_entity_dist: dict[str, list[float]],
        default_dist: list[float],
        max_length: int = 4096,
        dist_dim: int | None = None,
        chat_template_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.word_entity_dist = word_entity_dist
        self.default_dist = default_dist
        self.max_length = max_length
        self.dist_dim = dist_dim if dist_dim is not None else len(default_dist)
        self.chat_template_fn = chat_template_fn

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

    def _format_sample(self, sample: dict[str, Any], include_output: bool = True) -> str:
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
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = self._build_labels(sample, input_ids)

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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "entity_dist_vectors": dist_vectors,
        }


class EDEFDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.max_length = max_length

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        dist_vectors = [f["entity_dist_vectors"] for f in features]

        max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        padded_input_ids: list[torch.Tensor] = []
        padded_attention: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        padded_dists: list[torch.Tensor] = []

        for ids, mask, labs, dists in zip(input_ids, attention_mask, labels, dist_vectors):
            pad_len = max_len - len(ids)
            if pad_len > 0:
                padded_input_ids.append(F.pad(ids, (0, pad_len), value=self.pad_token_id))
                padded_attention.append(F.pad(mask, (0, pad_len), value=0))
                padded_labels.append(F.pad(labs, (0, pad_len), value=-100))
                padded_dists.append(F.pad(dists, (0, 0, 0, pad_len), value=0.0))
            elif pad_len < 0:
                padded_input_ids.append(ids[:max_len])
                padded_attention.append(mask[:max_len])
                padded_labels.append(labs[:max_len])
                padded_dists.append(dists[:max_len])
            else:
                padded_input_ids.append(ids)
                padded_attention.append(mask)
                padded_labels.append(labs)
                padded_dists.append(dists)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention),
            "labels": torch.stack(padded_labels),
            "entity_dist_vectors": torch.stack(padded_dists),
        }


def build_edef_dataset(
    data_path: str,
    tokenizer: Any,
    dist_path: str,
    max_length: int = 4096,
    dist_dim: int | None = None,
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
    word_entity_dist, default_dist = load_distributions(dist_path)
    effective_dist_dim = len(default_dist) if dist_dim is None else dist_dim
    if effective_dist_dim != len(default_dist):
        raise ValueError(
            f"dist_dim={effective_dist_dim} does not match distribution file dimension={len(default_dist)}"
        )

    return EDEFDataset(
        samples=samples,
        tokenizer=tokenizer,
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        max_length=max_length,
        dist_dim=effective_dist_dim,
        chat_template_fn=chat_template_fn,
    )


class _MockTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
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

    def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]:
        return [self._id_to_token[i] for i in input_ids]


def _smoke_test(
    data_path: str,
    dist_path: str | None,
    tokenizer_name: str,
    max_length: int,
    dist_dim: int,
) -> None:
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[:3]

    tokenizer: Any
    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        print(f"Loaded tokenizer: {tokenizer_name}")
    except Exception as exc:
        tokenizer = _MockTokenizer()
        print(f"transformers tokenizer unavailable ({exc}); using mock tokenizer")

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

    dataset = EDEFDataset(
        samples=samples,
        tokenizer=tokenizer,
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        max_length=max_length,
        dist_dim=dist_dim,
    )

    sample0 = dataset[0]
    print("\nSingle sample checks:")
    print(f"input_ids: {tuple(sample0['input_ids'].shape)}")
    print(f"attention_mask: {tuple(sample0['attention_mask'].shape)}")
    print(f"labels: {tuple(sample0['labels'].shape)}")
    print(f"entity_dist_vectors: {tuple(sample0['entity_dist_vectors'].shape)}")
    print(
        "masked label ratio:",
        float((sample0["labels"] == -100).sum().item()) / float(sample0["labels"].numel()),
    )

    collator = EDEFDataCollator(tokenizer=tokenizer, max_length=max_length)
    batch = collator([dataset[0], dataset[1]])
    print("\nBatch checks:")
    print(f"input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"attention_mask: {tuple(batch['attention_mask'].shape)}")
    print(f"labels: {tuple(batch['labels'].shape)}")
    print(f"entity_dist_vectors: {tuple(batch['entity_dist_vectors'].shape)}")

    print("\nToken-distribution alignment preview (first 12 tokens):")
    preview_ids = sample0["input_ids"][:12].tolist()
    preview_tokens = tokenizer.convert_ids_to_tokens(preview_ids)
    preview_dists = sample0["entity_dist_vectors"][:12]
    for i, (tok, dist) in enumerate(zip(preview_tokens, preview_dists)):
        max_idx = int(dist.argmax().item())
        max_val = float(dist[max_idx].item())
        print(f"  {i:02d}: {tok:24s} -> idx={max_idx:02d} prob={max_val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for EDEFDataset and EDEFDataCollator")
    parser.add_argument("--data-path", required=True, help="Path to Alpaca-format JSON data")
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
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dist-dim", type=int, default=45)
    args = parser.parse_args()

    _smoke_test(
        data_path=args.data_path,
        dist_path=args.dist_path,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        dist_dim=args.dist_dim,
    )
