#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_scheduler

Span = Tuple[int, int, str]
ScoredSpan = Tuple[int, int, str, float]
NEGATIVE_LOGIT = -1e4


@dataclass
class SplitArtifacts:
    features: List[Dict[str, Any]]
    gold_spans_by_doc: Dict[str, Set[Span]]
    covered_gold_spans_by_doc: Dict[str, Set[Span]]
    summary: Dict[str, Any]


class NestedChunkDataset(TorchDataset):
    def __init__(self, features: Sequence[Dict[str, Any]]) -> None:
        self.features = list(features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.features[index]


class NestedSpanCollator:
    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(features)
        seq_len = len(features[0]["input_ids"])

        batch: Dict[str, Any] = {
            "input_ids": torch.tensor(
                [feature["input_ids"] for feature in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [feature["attention_mask"] for feature in features], dtype=torch.long
            ),
            "word_start_mask": torch.tensor(
                [feature["word_start_mask"] for feature in features], dtype=torch.bool
            ),
            "word_end_mask": torch.tensor(
                [feature["word_end_mask"] for feature in features], dtype=torch.bool
            ),
        }

        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor(
                [feature["token_type_ids"] for feature in features], dtype=torch.long
            )

        labels = torch.zeros(
            (batch_size, self.num_labels, seq_len, seq_len), dtype=torch.float32
        )
        metadata: List[Dict[str, Any]] = []

        for batch_index, feature in enumerate(features):
            for label_id, start_token, end_token in feature["gold_spans"]:
                labels[batch_index, label_id, start_token, end_token] = 1.0

            metadata.append(
                {
                    "doc_id": feature["doc_id"],
                    "chunk_id": feature["chunk_id"],
                    "word_ids": feature["word_ids"],
                }
            )

        batch["labels"] = labels
        batch["metadata"] = metadata
        return batch


class GlobalPointerNERModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        inner_dim: int = 64,
        dropout: float = 0.1,
        use_rope: bool = True,
        hard_negative_ratio: float = 8.0,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        if inner_dim % 2 != 0:
            raise ValueError("--inner-dim must be even when RoPE is enabled.")
        if hard_negative_ratio <= 0:
            raise ValueError("--hard-negative-ratio must be positive.")

        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.num_labels = num_labels
        self.inner_dim = inner_dim
        self.use_rope = use_rope
        self.hard_negative_ratio = hard_negative_ratio
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, num_labels * inner_dim * 2)

    @staticmethod
    def _build_valid_span_mask(
        attention_mask: torch.Tensor,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = attention_mask.size(1)
        base_mask = (
            attention_mask[:, None, :, None].bool()
            & attention_mask[:, None, None, :].bool()
            & word_start_mask[:, None, :, None].bool()
            & word_end_mask[:, None, None, :].bool()
        )
        upper_triangle = torch.triu(
            torch.ones(
                (seq_len, seq_len), device=attention_mask.device, dtype=torch.bool
            )
        )
        return base_mask & upper_triangle.unsqueeze(0).unsqueeze(0)

    def _apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, inner_dim = tensor.shape
        device = tensor.device

        position_ids = torch.arange(seq_len, dtype=tensor.dtype, device=device)
        indices = torch.arange(0, inner_dim, 2, dtype=tensor.dtype, device=device)
        inverse_frequency = torch.pow(10000.0, -indices / inner_dim)
        sinusoid = torch.einsum("n,d->nd", position_ids, inverse_frequency)
        sin = torch.sin(sinusoid)[None, :, None, :]
        cos = torch.cos(sinusoid)[None, :, None, :]

        even_tensor = tensor[..., ::2]
        odd_tensor = tensor[..., 1::2]
        rotated_even = even_tensor * cos - odd_tensor * sin
        rotated_odd = even_tensor * sin + odd_tensor * cos
        return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

    def _hard_negative_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = valid_mask.expand_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        positive_mask = (labels > 0.5) & expanded_mask
        negative_mask = (labels <= 0.5) & expanded_mask

        positive_loss = loss[positive_mask]
        negative_loss = loss[negative_mask]

        if positive_loss.numel() == 0:
            if negative_loss.numel() == 0:
                return logits.new_zeros(())
            hardest_negative_count = min(negative_loss.numel(), 256)
            return torch.topk(negative_loss, k=hardest_negative_count).values.mean()

        if negative_loss.numel():
            hardest_negative_count = max(
                1, int(math.ceil(positive_loss.numel() * self.hard_negative_ratio))
            )
            if negative_loss.numel() > hardest_negative_count:
                negative_loss = torch.topk(
                    negative_loss, k=hardest_negative_count
                ).values

        negative_term = (
            negative_loss.mean() if negative_loss.numel() else logits.new_zeros(())
        )
        return positive_loss.mean() + negative_term

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_start_mask: torch.Tensor,
        word_end_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_inputs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        sequence_output = self.encoder(**encoder_inputs).last_hidden_state
        sequence_output = self.dropout(sequence_output)

        batch_size, seq_len, _ = sequence_output.shape
        projection = self.projection(sequence_output)
        projection = projection.view(
            batch_size, seq_len, self.num_labels, 2, self.inner_dim
        )
        query = projection[..., 0, :]
        key = projection[..., 1, :]

        if self.use_rope:
            query = self._apply_rope(query)
            key = self._apply_rope(key)

        logits = torch.einsum("bmhd,bnhd->bhmn", query, key) / math.sqrt(self.inner_dim)
        valid_mask = self._build_valid_span_mask(
            attention_mask, word_start_mask, word_end_mask
        )
        masked_logits = logits.masked_fill(~valid_mask, NEGATIVE_LOGIT)

        loss = None
        if labels is not None:
            # Hard-negative mining keeps a few bad false positives from being
            # washed out by the large number of easy empty spans.
            loss = self._hard_negative_bce_loss(masked_logits, labels, valid_mask)

        return {
            "loss": loss,
            "logits": masked_logits,
            "valid_mask": valid_mask,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a chunked encoder-based nested NER model and report test precision/recall/F1."
    )

    parser.add_argument(
        "--dataset-name",
        default="anurag-raapid/chia",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="FacebookAI/roberta-large",
        help="Encoder checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/chia-nested-ner", help="Where to save outputs."
    )

    parser.add_argument("--train-split", default="train", help="Training split name.")
    parser.add_argument(
        "--validation-split", default="validation", help="Validation split name."
    )
    parser.add_argument("--test-split", default="test", help="Test split name.")

    parser.add_argument(
        "--tokens-field", default="tokens", help="Field containing tokenized words."
    )
    parser.add_argument(
        "--entities-field", default="entities", help="Field containing entity spans."
    )
    parser.add_argument(
        "--entity-label-field",
        default="type",
        help="Entity label field inside each entity.",
    )
    parser.add_argument(
        "--entity-start-field",
        default="token_start",
        help="Entity start field (inclusive).",
    )
    parser.add_argument(
        "--entity-end-field", default="token_end", help="Entity end field (exclusive)."
    )
    parser.add_argument(
        "--document-id-field",
        default="filename",
        help="Optional field used to identify documents during aggregation.",
    )

    parser.add_argument(
        "--max-length", type=int, default=512, help="Tokenizer chunk length."
    )
    parser.add_argument(
        "--stride", type=int, default=128, help="Tokenizer overflow stride."
    )
    parser.add_argument(
        "--inner-dim", type=int, default=64, help="GlobalPointer inner dimension."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applied before the span head.",
    )
    parser.add_argument(
        "--disable-rope", action="store_true", help="Disable rotary position encoding."
    )

    parser.add_argument(
        "--num-train-epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=8, help="Training batch size."
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=8, help="Evaluation batch size."
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="AdamW weight decay."
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="Linear warmup ratio."
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Gradient clipping value."
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader worker count."
    )

    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.95,
        help="Default probability threshold used during epoch-end validation.",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.80,0.90,0.95,0.97,0.99",
        help="Comma-separated thresholds to score on validation before final test evaluation.",
    )
    parser.add_argument(
        "--max-predictions-per-chunk",
        type=int,
        default=256,
        help="Candidate cap before boundary pruning; set <=0 to disable.",
    )
    parser.add_argument(
        "--max-spans-per-start",
        type=int,
        default=1,
        help="Maximum decoded spans per (label, start); set <=0 to disable.",
    )
    parser.add_argument(
        "--max-spans-per-end",
        type=int,
        default=1,
        help="Maximum decoded spans per (label, end); set <=0 to disable.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=8.0,
        help="How many hard negatives to keep per positive span during training.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional train subset size.",
    )
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=None,
        help="Optional validation subset size.",
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=None, help="Optional test subset size."
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to AutoTokenizer/AutoModel for community checkpoints that require it.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_argument: str) -> torch.device:
    if device_argument == "cpu":
        return torch.device("cpu")
    if device_argument == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_max_length(tokenizer: Any, requested_max_length: int) -> int:
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if (
        tokenizer_limit
        and tokenizer_limit < 100_000
        and requested_max_length > tokenizer_limit
    ):
        print(
            f"Requested max_length={requested_max_length}, but tokenizer supports {tokenizer_limit}. "
            f"Using {tokenizer_limit} instead."
        )
        return int(tokenizer_limit)
    return requested_max_length


def select_subset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def build_doc_id(
    example: Dict[str, Any],
    split_name: str,
    example_index: int,
    document_id_field: Optional[str],
) -> str:
    if (
        document_id_field
        and document_id_field in example
        and example[document_id_field]
    ):
        return f"{split_name}:{example[document_id_field]}"
    return f"{split_name}:example-{example_index}"


def collect_label_list(
    splits: Iterable[Dataset],
    entities_field: str,
    entity_label_field: str,
) -> List[str]:
    labels: Set[str] = set()
    for dataset in splits:
        for example in dataset:
            for entity in example[entities_field]:
                labels.add(str(entity[entity_label_field]))
    return sorted(labels)


def parse_thresholds(grid: str) -> List[float]:
    thresholds = [float(value.strip()) for value in grid.split(",") if value.strip()]
    if not thresholds:
        raise ValueError(
            "--threshold-grid must contain at least one numeric threshold."
        )
    return thresholds


def prepare_split(
    dataset: Dataset,
    split_name: str,
    tokenizer: Any,
    label_to_id: Dict[str, int],
    args: argparse.Namespace,
    max_length: int,
) -> SplitArtifacts:
    features: List[Dict[str, Any]] = []
    gold_spans_by_doc: Dict[str, Set[Span]] = {}
    covered_gold_spans_by_doc: Dict[str, Set[Span]] = {}
    label_counter: Counter[str] = Counter()
    uncovered_examples = 0
    total_chunks = 0

    for example_index, example in enumerate(
        tqdm(dataset, desc=f"Preprocessing {split_name}", leave=False)
    ):
        tokens = example[args.tokens_field]
        entities = example[args.entities_field]
        doc_id = build_doc_id(
            example, split_name, example_index, args.document_id_field
        )

        gold_spans: Set[Span] = set()
        for entity in entities:
            label = str(entity[args.entity_label_field])
            start_word = int(entity[args.entity_start_field])
            end_word = int(entity[args.entity_end_field])
            if end_word <= start_word:
                continue
            gold_spans.add((start_word, end_word, label))
            label_counter[label] += 1

        gold_spans_by_doc[doc_id] = gold_spans
        covered_gold_spans: Set[Span] = set()

        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            return_overflowing_tokens=True,
            stride=args.stride,
            max_length=max_length,
            padding="max_length",
        )

        for chunk_index in range(len(tokenized["input_ids"])):
            word_ids_optional = tokenized.word_ids(batch_index=chunk_index)
            word_ids = [
                word_id if word_id is not None else -1 for word_id in word_ids_optional
            ]

            word_start_mask = [0] * len(word_ids)
            word_end_mask = [0] * len(word_ids)
            word_to_start_token: Dict[int, int] = {}
            word_to_end_token: Dict[int, int] = {}

            for token_index, word_id in enumerate(word_ids_optional):
                if word_id is None:
                    continue
                if token_index == 0 or word_ids_optional[token_index - 1] != word_id:
                    word_start_mask[token_index] = 1
                    word_to_start_token[word_id] = token_index
                if (
                    token_index == len(word_ids_optional) - 1
                    or word_ids_optional[token_index + 1] != word_id
                ):
                    word_end_mask[token_index] = 1
                    word_to_end_token[word_id] = token_index

            chunk_gold_spans: Set[Tuple[int, int, int]] = set()
            for start_word, end_word, label in gold_spans:
                last_word = end_word - 1
                if (
                    start_word not in word_to_start_token
                    or last_word not in word_to_end_token
                ):
                    continue
                start_token = word_to_start_token[start_word]
                end_token = word_to_end_token[last_word]
                chunk_gold_spans.add((label_to_id[label], start_token, end_token))
                covered_gold_spans.add((start_word, end_word, label))

            feature: Dict[str, Any] = {
                "input_ids": tokenized["input_ids"][chunk_index],
                "attention_mask": tokenized["attention_mask"][chunk_index],
                "word_start_mask": word_start_mask,
                "word_end_mask": word_end_mask,
                "word_ids": word_ids,
                "gold_spans": sorted(chunk_gold_spans),
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}:chunk-{chunk_index}",
            }
            if "token_type_ids" in tokenized:
                feature["token_type_ids"] = tokenized["token_type_ids"][chunk_index]

            features.append(feature)
            total_chunks += 1

        covered_gold_spans_by_doc[doc_id] = covered_gold_spans
        if len(covered_gold_spans) < len(gold_spans):
            uncovered_examples += 1

    total_gold_spans = sum(len(spans) for spans in gold_spans_by_doc.values())
    total_covered_spans = sum(
        len(spans) for spans in covered_gold_spans_by_doc.values()
    )

    summary = {
        "split": split_name,
        "documents": len(dataset),
        "chunks": total_chunks,
        "total_gold_spans": total_gold_spans,
        "covered_gold_spans": total_covered_spans,
        "gold_span_coverage": round(
            (total_covered_spans / total_gold_spans) if total_gold_spans else 1.0,
            6,
        ),
        "documents_with_uncovered_spans": uncovered_examples,
        "labels": dict(sorted(label_counter.items())),
    }

    return SplitArtifacts(
        features=features,
        gold_spans_by_doc=gold_spans_by_doc,
        covered_gold_spans_by_doc=covered_gold_spans_by_doc,
        summary=summary,
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    device_batch: Dict[str, Any] = {"metadata": batch["metadata"]}
    for key, value in batch.items():
        if key == "metadata":
            continue
        if torch.is_tensor(value):
            device_batch[key] = value.to(device)
        else:
            device_batch[key] = value
    return device_batch


def span_sets_to_metrics(
    predicted_by_doc: Dict[str, Set[Span]],
    gold_by_doc: Dict[str, Set[Span]],
    label_list: Sequence[str],
) -> Dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    per_label_counts: Dict[str, Dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in label_list
    }

    for doc_id, gold_spans in gold_by_doc.items():
        predicted_spans = predicted_by_doc.get(doc_id, set())

        tp_spans = predicted_spans & gold_spans
        fp_spans = predicted_spans - gold_spans
        fn_spans = gold_spans - predicted_spans

        true_positive += len(tp_spans)
        false_positive += len(fp_spans)
        false_negative += len(fn_spans)

        for _, _, label in tp_spans:
            per_label_counts[label]["tp"] += 1
        for _, _, label in fp_spans:
            per_label_counts[label]["fp"] += 1
        for _, _, label in fn_spans:
            per_label_counts[label]["fn"] += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_label_metrics: Dict[str, Dict[str, float]] = {}
    for label, counts in per_label_counts.items():
        label_precision = (
            counts["tp"] / (counts["tp"] + counts["fp"])
            if (counts["tp"] + counts["fp"])
            else 0.0
        )
        label_recall = (
            counts["tp"] / (counts["tp"] + counts["fn"])
            if (counts["tp"] + counts["fn"])
            else 0.0
        )
        label_f1 = (
            2 * label_precision * label_recall / (label_precision + label_recall)
            if (label_precision + label_recall)
            else 0.0
        )
        per_label_metrics[label] = {
            "precision": round(label_precision, 6),
            "recall": round(label_recall, 6),
            "f1": round(label_f1, 6),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "per_label": per_label_metrics,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_predictions_jsonl(
    path: Path,
    predicted_by_doc: Dict[str, Set[Span]],
    gold_by_doc: Dict[str, Set[Span]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc_id in sorted(gold_by_doc):
            predicted = [
                {"start": start, "end": end, "label": label}
                for start, end, label in sorted(predicted_by_doc.get(doc_id, set()))
            ]
            gold = [
                {"start": start, "end": end, "label": label}
                for start, end, label in sorted(gold_by_doc[doc_id])
            ]
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "predicted_spans": predicted,
                        "gold_spans": gold,
                    }
                )
                + "\n"
            )


def decode_scored_spans(
    scored_spans: Sequence[ScoredSpan],
    threshold: float,
    max_spans_per_start: int,
    max_spans_per_end: int,
) -> Set[Span]:
    predicted_spans: Set[Span] = set()
    start_counts: Counter[Tuple[str, int]] = Counter()
    end_counts: Counter[Tuple[str, int]] = Counter()

    for start, end, label, score in sorted(
        scored_spans, key=lambda item: item[3], reverse=True
    ):
        if score < threshold:
            continue
        if (
            max_spans_per_start > 0
            and start_counts[(label, start)] >= max_spans_per_start
        ):
            continue
        if max_spans_per_end > 0 and end_counts[(label, end)] >= max_spans_per_end:
            continue

        span = (start, end, label)
        if span in predicted_spans:
            continue

        predicted_spans.add(span)
        start_counts[(label, start)] += 1
        end_counts[(label, end)] += 1

    return predicted_spans


def collect_scored_spans(
    model: GlobalPointerNERModel,
    dataloader: DataLoader,
    device: torch.device,
    label_list: Sequence[str],
    candidate_threshold: float,
    max_predictions_per_chunk: int,
) -> Tuple[Optional[float], Dict[str, List[ScoredSpan]]]:
    model.eval()
    scored_by_doc: Dict[str, List[ScoredSpan]] = defaultdict(list)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            metadata = batch["metadata"]
            device_batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
                word_start_mask=device_batch["word_start_mask"],
                word_end_mask=device_batch["word_end_mask"],
                token_type_ids=device_batch.get("token_type_ids"),
                labels=device_batch["labels"],
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                total_batches += 1

            scores = torch.sigmoid(outputs["logits"]).cpu()
            valid_mask = outputs["valid_mask"].expand_as(outputs["logits"]).cpu()

            for sample_index, sample_metadata in enumerate(metadata):
                sample_scores = scores[sample_index].masked_fill(
                    ~valid_mask[sample_index], 0.0
                )
                positive_indices = (sample_scores > candidate_threshold).nonzero(
                    as_tuple=False
                )

                if (
                    max_predictions_per_chunk > 0
                    and positive_indices.size(0) > max_predictions_per_chunk
                ):
                    positive_scores = sample_scores[
                        positive_indices[:, 0],
                        positive_indices[:, 1],
                        positive_indices[:, 2],
                    ]
                    top_indices = torch.topk(
                        positive_scores, k=max_predictions_per_chunk
                    ).indices
                    positive_indices = positive_indices[top_indices]

                word_ids = sample_metadata["word_ids"]
                doc_id = sample_metadata["doc_id"]
                for label_id, start_token, end_token in positive_indices.tolist():
                    start_word = word_ids[start_token]
                    end_word = word_ids[end_token]
                    if start_word < 0 or end_word < 0 or end_word < start_word:
                        continue

                    scored_by_doc[doc_id].append(
                        (
                            start_word,
                            end_word + 1,
                            label_list[label_id],
                            float(
                                sample_scores[label_id, start_token, end_token].item()
                            ),
                        )
                    )

    average_loss = round(total_loss / total_batches, 6) if total_batches else None
    return average_loss, dict(scored_by_doc)


def evaluate_scored_spans(
    scored_by_doc: Dict[str, List[ScoredSpan]],
    gold_by_doc: Dict[str, Set[Span]],
    covered_gold_by_doc: Dict[str, Set[Span]],
    label_list: Sequence[str],
    threshold: float,
    max_spans_per_start: int,
    max_spans_per_end: int,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    predicted_by_doc = {
        doc_id: decode_scored_spans(
            scored_by_doc.get(doc_id, []),
            threshold=threshold,
            max_spans_per_start=max_spans_per_start,
            max_spans_per_end=max_spans_per_end,
        )
        for doc_id in gold_by_doc
    }

    all_gold_metrics = span_sets_to_metrics(predicted_by_doc, gold_by_doc, label_list)
    covered_gold_metrics = span_sets_to_metrics(
        predicted_by_doc, covered_gold_by_doc, label_list
    )
    coverage = (
        sum(len(spans) for spans in covered_gold_by_doc.values())
        / sum(len(spans) for spans in gold_by_doc.values())
        if sum(len(spans) for spans in gold_by_doc.values())
        else 1.0
    )

    return {
        "loss": loss,
        "precision": all_gold_metrics["precision"],
        "recall": all_gold_metrics["recall"],
        "f1": all_gold_metrics["f1"],
        "tp": all_gold_metrics["tp"],
        "fp": all_gold_metrics["fp"],
        "fn": all_gold_metrics["fn"],
        "gold_span_coverage": round(coverage, 6),
        "covered_gold_precision": covered_gold_metrics["precision"],
        "covered_gold_recall": covered_gold_metrics["recall"],
        "covered_gold_f1": covered_gold_metrics["f1"],
        "per_label": all_gold_metrics["per_label"],
        "predicted_by_doc": predicted_by_doc,
    }


def select_best_threshold(
    scored_by_doc: Dict[str, List[ScoredSpan]],
    gold_by_doc: Dict[str, Set[Span]],
    covered_gold_by_doc: Dict[str, Set[Span]],
    label_list: Sequence[str],
    threshold_candidates: Sequence[float],
    max_spans_per_start: int,
    max_spans_per_end: int,
    loss: Optional[float] = None,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    threshold_search: List[Dict[str, Any]] = []
    selected_threshold = threshold_candidates[0]
    best_metrics: Dict[str, Any] = {}
    best_f1 = -1.0

    for threshold in threshold_candidates:
        metrics = evaluate_scored_spans(
            scored_by_doc=scored_by_doc,
            gold_by_doc=gold_by_doc,
            covered_gold_by_doc=covered_gold_by_doc,
            label_list=label_list,
            threshold=threshold,
            max_spans_per_start=max_spans_per_start,
            max_spans_per_end=max_spans_per_end,
            loss=loss,
        )
        metrics["threshold"] = threshold

        summary = dict(metrics)
        summary.pop("predicted_by_doc", None)
        threshold_search.append(summary)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            selected_threshold = threshold
            best_metrics = metrics

    return selected_threshold, threshold_search, best_metrics


def evaluate_model(
    model: GlobalPointerNERModel,
    dataloader: DataLoader,
    device: torch.device,
    label_list: Sequence[str],
    gold_by_doc: Dict[str, Set[Span]],
    covered_gold_by_doc: Dict[str, Set[Span]],
    threshold: float,
    max_predictions_per_chunk: int,
    max_spans_per_start: int,
    max_spans_per_end: int,
    candidate_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    effective_candidate_threshold = (
        threshold if candidate_threshold is None else candidate_threshold
    )
    loss, scored_by_doc = collect_scored_spans(
        model=model,
        dataloader=dataloader,
        device=device,
        label_list=label_list,
        candidate_threshold=effective_candidate_threshold,
        max_predictions_per_chunk=max_predictions_per_chunk,
    )
    return evaluate_scored_spans(
        scored_by_doc=scored_by_doc,
        gold_by_doc=gold_by_doc,
        covered_gold_by_doc=covered_gold_by_doc,
        label_list=label_list,
        threshold=threshold,
        max_spans_per_start=max_spans_per_start,
        max_spans_per_end=max_spans_per_end,
        loss=loss,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    model: GlobalPointerNERModel,
    tokenizer: Any,
    args: argparse.Namespace,
    label_list: Sequence[str],
    validation_metrics: Dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = dict(validation_metrics)
    metrics_payload.pop("predicted_by_doc", None)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_list": list(label_list),
            "model_name_or_path": args.model_name_or_path,
            "inner_dim": args.inner_dim,
            "dropout": args.dropout,
            "use_rope": not args.disable_rope,
            "hard_negative_ratio": args.hard_negative_ratio,
            "trust_remote_code": args.trust_remote_code,
        },
        checkpoint_dir / "model.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    save_json(checkpoint_dir / "validation_metrics.json", metrics_payload)


def build_dataloaders(
    train_artifacts: SplitArtifacts,
    validation_artifacts: SplitArtifacts,
    test_artifacts: SplitArtifacts,
    num_labels: int,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collator = NestedSpanCollator(num_labels=num_labels)

    train_loader = DataLoader(
        NestedChunkDataset(train_artifacts.features),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        NestedChunkDataset(validation_artifacts.features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        NestedChunkDataset(test_artifacts.features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    return train_loader, validation_loader, test_loader


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset_dict: DatasetDict = load_dataset(args.dataset_name, args.dataset_config)
    for split_name in [args.train_split, args.validation_split, args.test_split]:
        if split_name not in dataset_dict:
            raise KeyError(
                f"Dataset split '{split_name}' was not found in {args.dataset_name}."
            )

    train_dataset = select_subset(
        dataset_dict[args.train_split], args.max_train_samples
    )
    validation_dataset = select_subset(
        dataset_dict[args.validation_split], args.max_validation_samples
    )
    test_dataset = select_subset(dataset_dict[args.test_split], args.max_test_samples)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required because overflow chunking depends on word_ids()."
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token
        else:
            raise ValueError(
                "Tokenizer does not define a pad token or a usable fallback token."
            )

    max_length = resolve_max_length(tokenizer, args.max_length)
    if args.stride >= max_length:
        raise ValueError("--stride must be smaller than --max-length.")

    label_list = collect_label_list(
        [train_dataset, validation_dataset, test_dataset],
        args.entities_field,
        args.entity_label_field,
    )
    label_to_id = {label: index for index, label in enumerate(label_list)}
    print(f"Loaded {len(label_list)} labels: {', '.join(label_list)}")

    train_artifacts = prepare_split(
        train_dataset, args.train_split, tokenizer, label_to_id, args, max_length
    )
    validation_artifacts = prepare_split(
        validation_dataset,
        args.validation_split,
        tokenizer,
        label_to_id,
        args,
        max_length,
    )
    test_artifacts = prepare_split(
        test_dataset, args.test_split, tokenizer, label_to_id, args, max_length
    )

    print(json.dumps(train_artifacts.summary, indent=2))
    print(json.dumps(validation_artifacts.summary, indent=2))
    print(json.dumps(test_artifacts.summary, indent=2))

    if not train_artifacts.features:
        raise RuntimeError(
            "Training split produced zero chunks. Check your dataset fields and max_length."
        )

    train_loader, validation_loader, test_loader = build_dataloaders(
        train_artifacts,
        validation_artifacts,
        test_artifacts,
        num_labels=len(label_list),
        args=args,
    )
    threshold_candidates = parse_thresholds(args.threshold_grid)
    if args.prediction_threshold not in threshold_candidates:
        threshold_candidates.append(args.prediction_threshold)
    evaluation_candidate_threshold = min(threshold_candidates)

    model = GlobalPointerNERModel(
        model_name_or_path=args.model_name_or_path,
        num_labels=len(label_list),
        inner_dim=args.inner_dim,
        dropout=args.dropout,
        use_rope=not args.disable_rope,
        hard_negative_ratio=args.hard_negative_ratio,
        trust_remote_code=args.trust_remote_code,
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / max(args.gradient_accumulation_steps, 1)
    )
    total_train_steps = max(1, steps_per_epoch * args.num_train_epochs)
    warmup_steps = int(args.warmup_ratio * total_train_steps)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    best_validation_f1 = -1.0
    best_checkpoint_dir = output_dir / "best_checkpoint"
    training_history: List[Dict[str, Any]] = []

    for epoch_index in range(args.num_train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch_index + 1}/{args.num_train_epochs}"
        )

        for step_index, batch in enumerate(progress_bar, start=1):
            device_batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=device_batch["input_ids"],
                attention_mask=device_batch["attention_mask"],
                word_start_mask=device_batch["word_start_mask"],
                word_end_mask=device_batch["word_end_mask"],
                token_type_ids=device_batch.get("token_type_ids"),
                labels=device_batch["labels"],
            )
            if outputs["loss"] is None:
                raise RuntimeError("Training loss was not computed.")

            loss = outputs["loss"] / args.gradient_accumulation_steps
            loss.backward()
            running_loss += outputs["loss"].item()
            progress_bar.set_postfix(loss=f"{running_loss / step_index:.4f}")

            should_step = (
                step_index % args.gradient_accumulation_steps == 0
                or step_index == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_loss, validation_scored_spans = collect_scored_spans(
            model=model,
            dataloader=validation_loader,
            device=device,
            label_list=label_list,
            candidate_threshold=evaluation_candidate_threshold,
            max_predictions_per_chunk=args.max_predictions_per_chunk,
        )
        validation_metrics = evaluate_scored_spans(
            scored_by_doc=validation_scored_spans,
            gold_by_doc=validation_artifacts.gold_spans_by_doc,
            covered_gold_by_doc=validation_artifacts.covered_gold_spans_by_doc,
            label_list=label_list,
            threshold=args.prediction_threshold,
            max_spans_per_start=args.max_spans_per_start,
            max_spans_per_end=args.max_spans_per_end,
            loss=validation_loss,
        )
        validation_metrics.pop("predicted_by_doc", None)
        epoch_selected_threshold, _, epoch_best_metrics = select_best_threshold(
            scored_by_doc=validation_scored_spans,
            gold_by_doc=validation_artifacts.gold_spans_by_doc,
            covered_gold_by_doc=validation_artifacts.covered_gold_spans_by_doc,
            label_list=label_list,
            threshold_candidates=threshold_candidates,
            max_spans_per_start=args.max_spans_per_start,
            max_spans_per_end=args.max_spans_per_end,
            loss=validation_loss,
        )
        validation_metrics["epoch"] = epoch_index + 1
        validation_metrics["train_loss"] = round(
            running_loss / max(len(train_loader), 1), 6
        )
        validation_metrics["selected_threshold"] = epoch_selected_threshold
        validation_metrics["selected_threshold_f1"] = epoch_best_metrics["f1"]
        training_history.append(validation_metrics)

        print(
            f"Epoch {epoch_index + 1}: "
            f"train_loss={validation_metrics['train_loss']:.4f}, "
            f"val_precision={validation_metrics['precision']:.4f}, "
            f"val_recall={validation_metrics['recall']:.4f}, "
            f"val_f1={validation_metrics['f1']:.4f}, "
            f"best_threshold={epoch_selected_threshold:.2f}, "
            f"best_threshold_f1={epoch_best_metrics['f1']:.4f}"
        )

        if epoch_best_metrics["f1"] > best_validation_f1:
            best_validation_f1 = epoch_best_metrics["f1"]
            checkpoint_metrics = dict(epoch_best_metrics)
            checkpoint_metrics["epoch"] = epoch_index + 1
            checkpoint_metrics["train_loss"] = validation_metrics["train_loss"]
            save_checkpoint(
                best_checkpoint_dir,
                model,
                tokenizer,
                args,
                label_list,
                checkpoint_metrics,
            )

    checkpoint = torch.load(best_checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    validation_loss, validation_scored_spans = collect_scored_spans(
        model=model,
        dataloader=validation_loader,
        device=device,
        label_list=label_list,
        candidate_threshold=evaluation_candidate_threshold,
        max_predictions_per_chunk=args.max_predictions_per_chunk,
    )
    selected_threshold, threshold_search, final_validation_metrics = (
        select_best_threshold(
            scored_by_doc=validation_scored_spans,
            gold_by_doc=validation_artifacts.gold_spans_by_doc,
            covered_gold_by_doc=validation_artifacts.covered_gold_spans_by_doc,
            label_list=label_list,
            threshold_candidates=threshold_candidates,
            max_spans_per_start=args.max_spans_per_start,
            max_spans_per_end=args.max_spans_per_end,
            loss=validation_loss,
        )
    )
    print(f"Selected threshold from validation grid: {selected_threshold:.2f}")
    validation_predictions = final_validation_metrics.pop("predicted_by_doc")

    test_loss, test_scored_spans = collect_scored_spans(
        model=model,
        dataloader=test_loader,
        device=device,
        label_list=label_list,
        candidate_threshold=selected_threshold,
        max_predictions_per_chunk=args.max_predictions_per_chunk,
    )
    test_metrics = evaluate_scored_spans(
        scored_by_doc=test_scored_spans,
        gold_by_doc=test_artifacts.gold_spans_by_doc,
        covered_gold_by_doc=test_artifacts.covered_gold_spans_by_doc,
        label_list=label_list,
        threshold=selected_threshold,
        max_spans_per_start=args.max_spans_per_start,
        max_spans_per_end=args.max_spans_per_end,
        loss=test_loss,
    )
    test_predictions = test_metrics.pop("predicted_by_doc")

    save_json(output_dir / "label_list.json", {"labels": label_list})
    save_json(
        output_dir / "preprocessing_summary.json",
        {
            "train": train_artifacts.summary,
            "validation": validation_artifacts.summary,
            "test": test_artifacts.summary,
        },
    )
    save_json(
        output_dir / "run_summary.json",
        {
            "arguments": vars(args),
            "effective_max_length": max_length,
            "device": str(device),
            "threshold_search": threshold_search,
            "selected_threshold": selected_threshold,
            "training_history": training_history,
            "final_validation_metrics": final_validation_metrics,
            "test_metrics": test_metrics,
        },
    )
    save_json(output_dir / "validation_metrics.json", final_validation_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_predictions_jsonl(
        output_dir / "validation_predictions.jsonl",
        validation_predictions,
        validation_artifacts.gold_spans_by_doc,
    )
    save_predictions_jsonl(
        output_dir / "test_predictions.jsonl",
        test_predictions,
        test_artifacts.gold_spans_by_doc,
    )

    print("Validation metrics:")
    print(json.dumps(final_validation_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
