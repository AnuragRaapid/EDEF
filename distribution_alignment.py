import json
import re
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as F


class TokenizerLike(Protocol):
    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = True,
        add_special_tokens: bool = False,
    ) -> dict[str, object]: ...

    def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]: ...


class AutoTokenizerLike(Protocol):
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str, trust_remote_code: bool = True
    ) -> TokenizerLike: ...


def _ensure_dist_vector(
    dist: list[float] | None, dist_dim: int, fallback: list[float]
) -> list[float]:
    if dist is None:
        return list(fallback)
    vec = [float(x) for x in dist[:dist_dim]]
    if len(vec) < dist_dim:
        vec = vec + [0.0] * (dist_dim - len(vec))
    return vec


def _is_punctuation_only(fragment: str) -> bool:
    return fragment != "" and not any(ch.isalnum() for ch in fragment)


def _expand_to_word(text: str, start: int, end: int) -> str:
    left = start
    right = end
    while left > 0 and not text[left - 1].isspace():
        left -= 1
    while right < len(text) and not text[right].isspace():
        right += 1
    return text[left:right].strip().lower()


def _hyphen_candidates(word: str) -> list[str]:
    if "-" not in word:
        return []
    return [part for part in word.split("-") if part]


def _to_float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    items = cast(list[object], values)
    out: list[float] = []
    for x in items:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif isinstance(x, str):
            try:
                out.append(float(x))
            except ValueError:
                out.append(0.0)
        else:
            out.append(0.0)
    return out


def _to_int_list(values: object) -> list[int]:
    if not isinstance(values, list):
        return []
    items = cast(list[object], values)
    out: list[int] = []
    for x in items:
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float):
            out.append(int(x))
        elif isinstance(x, str) and x.strip().lstrip("-").isdigit():
            out.append(int(x))
    return out


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


def load_distributions(
    dist_path: str,
) -> tuple[dict[str, list[float]], list[float], dict[str, list[float]]]:
    """
    Load entity distributions, n-gram distributions, and default dist from JSON files.

    Args:
        dist_path: path to entity_distributions.json

    Returns:
        (word_entity_dist, default_dist, ngram_dist) tuple.
        ngram_dist may be empty if the file does not exist.
    """
    with open(dist_path, "r", encoding="utf-8") as f:
        raw = cast(dict[str, object], json.load(f))
    word_entity_dist: dict[str, list[float]] = {
        str(k).lower(): _to_float_list(cast(list[object], v))
        for k, v in raw.items()
        if isinstance(v, list)
    }

    dist_file = Path(dist_path).resolve()
    stats_path = dist_file.with_name("distribution_stats.json")
    index_path = dist_file.with_name("entity_type_index.json")
    ngram_path = dist_file.with_name("ngram_distributions.json")

    default_dist: list[float] | None = None
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as f:
                stats_raw = cast(dict[str, object], json.load(f))
            default_candidate = stats_raw.get("default_unknown_distribution")
            if isinstance(default_candidate, list):
                default_dist = _to_float_list(default_candidate)
        except (json.JSONDecodeError, OSError):
            default_dist = None

    vector_dim = len(next(iter(word_entity_dist.values()), []))
    if default_dist is None:
        if vector_dim == 0 and index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as f:
                    index_raw = cast(dict[str, object], json.load(f))
                numeric_indices = [
                    _to_int(value) for value in index_raw.values() if _to_int(value) >= 0
                ]
                if numeric_indices:
                    vector_dim = max(numeric_indices) + 1
            except (json.JSONDecodeError, OSError):
                vector_dim = 0

        if vector_dim == 0:
            vector_dim = 45

        default_dist = [0.0] * vector_dim
        default_dist[-1] = 1.0
    elif vector_dim and len(default_dist) != vector_dim:
        default_dist = default_dist[:vector_dim]
        if len(default_dist) < vector_dim:
            default_dist = default_dist + [0.0] * (vector_dim - len(default_dist))

    ngram_dist: dict[str, list[float]] = {}
    if ngram_path.exists():
        try:
            with ngram_path.open("r", encoding="utf-8") as f:
                ngram_raw = cast(dict[str, object], json.load(f))
            ngram_dist = {
                str(k).lower(): _to_float_list(cast(list[object], v))
                for k, v in ngram_raw.items()
                if isinstance(v, list)
            }
        except (json.JSONDecodeError, OSError):
            ngram_dist = {}

    return word_entity_dist, default_dist, ngram_dist


def _ngram_fallback(
    word: str,
    ngram_dist: dict[str, list[float]],
    dist_dim: int,
    n_range: tuple[int, int] = (3, 5),
) -> list[float] | None:
    """Aggregate n-gram distributions for an unknown word."""
    if not ngram_dist or len(word) < n_range[0]:
        return None
    accum = [0.0] * dist_dim
    count = 0
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(word) - n + 1):
            ngram = word[i : i + n]
            ng_dist = ngram_dist.get(ngram)
            if ng_dist is not None:
                for j in range(min(len(ng_dist), dist_dim)):
                    accum[j] += ng_dist[j]
                count += 1
    if count == 0:
        return None
    total = sum(accum)
    if total <= 0:
        return None
    return [v / total for v in accum]


def get_token_distributions(
    text: str,
    tokenizer: TokenizerLike,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    dist_dim: int = 45,
    ngram_dist: dict[str, list[float]] | None = None,
) -> torch.Tensor:
    """
    Given raw text, returns distribution vectors aligned to subword tokens.

    Uses a cascading lookup: word -> hyphen parts -> subword token -> n-gram fallback -> default.

    Args:
        text: raw input string
        tokenizer: Qwen3 tokenizer (HuggingFace)
        word_entity_dist: dict mapping lowercased word -> dist_dim-dim list
        default_dist: dist_dim-dim list for unknown words (global prior)
        dist_dim: dimension of distribution vector
        ngram_dist: optional dict mapping character n-grams -> dist_dim-dim list

    Returns:
        torch.Tensor of shape (num_tokens, dist_dim)
    """
    safe_default = _ensure_dist_vector(default_dist, dist_dim, [0.0] * dist_dim)
    _ngram = ngram_dist if ngram_dist is not None else {}

    if text == "":
        return torch.empty((0, dist_dim), dtype=torch.float32)

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets_any = encoding.get("offset_mapping", [])
    input_ids_any = encoding.get("input_ids", [])
    offset_items = cast(list[object], offsets_any) if isinstance(offsets_any, list) else []
    offsets: list[tuple[int, int]] = []
    for pair in offset_items:
        if not isinstance(pair, (list, tuple)):
            continue
        pair_items = cast(list[object] | tuple[object, ...], pair)
        if len(pair_items) != 2:
            continue
        offsets.append((_to_int(pair_items[0]), _to_int(pair_items[1])))
    input_ids = _to_int_list(input_ids_any)
    token_texts = tokenizer.convert_ids_to_tokens(input_ids) if input_ids else []

    aligned_vectors: list[list[float]] = []
    for idx, (start, end) in enumerate(offsets):
        if (start, end) == (0, 0):
            aligned_vectors.append(list(safe_default))
            continue

        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        span_text = text[start:end]
        token_text = token_texts[idx].lower() if idx < len(token_texts) else ""

        if any(ch.isspace() for ch in span_text):
            dist = word_entity_dist.get(token_text)
            aligned_vectors.append(_ensure_dist_vector(dist, dist_dim, safe_default))
            continue

        if _is_punctuation_only(span_text):
            aligned_vectors.append(list(safe_default))
            continue

        word = _expand_to_word(text, start, end)

        dist = word_entity_dist.get(word)
        if dist is None:
            for part in _hyphen_candidates(word):
                dist = word_entity_dist.get(part)
                if dist is not None:
                    break
        if dist is None and token_text:
            dist = word_entity_dist.get(token_text)
        if dist is None:
            dist = _ngram_fallback(word, _ngram, dist_dim)

        aligned_vectors.append(_ensure_dist_vector(dist, dist_dim, safe_default))

    if not aligned_vectors:
        return torch.empty((0, dist_dim), dtype=torch.float32)

    return torch.tensor(aligned_vectors, dtype=torch.float32)


def batch_get_distributions(
    texts: list[str],
    tokenizer: TokenizerLike,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_length: int = 4096,
    dist_dim: int = 45,
    ngram_dist: dict[str, list[float]] | None = None,
) -> torch.Tensor:
    """
    Process a batch of texts and return padded distribution tensors.

    Returns:
        torch.Tensor of shape (batch_size, max_len_in_batch, dist_dim)
        Padded positions have zero vectors.
    """
    batch_size = len(texts)
    if batch_size == 0:
        return torch.zeros((0, 0, dist_dim), dtype=torch.float32)

    per_text: list[torch.Tensor] = []
    for text in texts:
        dist = get_token_distributions(
            text=text,
            tokenizer=tokenizer,
            word_entity_dist=word_entity_dist,
            default_dist=default_dist,
            dist_dim=dist_dim,
            ngram_dist=ngram_dist,
        )
        per_text.append(dist[:max_length])

    max_len_in_batch = max((t.shape[0] for t in per_text), default=0)
    if max_len_in_batch == 0:
        return torch.zeros((batch_size, 0, dist_dim), dtype=torch.float32)

    padded: list[torch.Tensor] = []
    for t in per_text:
        pad_len = max_len_in_batch - t.shape[0]
        if pad_len > 0:
            t = F.pad(t, (0, 0, 0, pad_len), mode="constant", value=0.0)
        padded.append(t)

    return torch.stack(padded, dim=0)


if __name__ == "__main__":
    mock_dist = {
        "pain": [0.0] * 24 + [0.8] + [0.0] * 19 + [0.2],
        "chest": [0.0] * 18 + [0.7] + [0.0] * 25 + [0.3],
        "mg": [0.0] * 25 + [0.9] + [0.0] * 18 + [0.1],
    }
    default = [0.0] * 44 + [1.0]

    try:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        auto_tokenizer = cast(AutoTokenizerLike, AutoTokenizer)
        tok = auto_tokenizer.from_pretrained(
            "Qwen/Qwen3-4B-Instruct", trust_remote_code=True
        )

        test_text = "Patient has chest pain and takes 500 mg"
        dists = get_token_distributions(test_text, tok, mock_dist, default)
        print(f"Text: {test_text}")
        print(f"Num tokens: {dists.shape[0]}")
        print(f"Dist dim: {dists.shape[1]}")

        encoding = tok(test_text, return_offsets_mapping=True, add_special_tokens=False)
        ids = _to_int_list(encoding.get("input_ids", []))
        tokens = tok.convert_ids_to_tokens(ids)
        for i, (token, dist) in enumerate(zip(tokens, dists)):
            max_idx = int(dist.argmax().item())
            max_val = dist[max_idx].item()
            print(
                f"  Token {i}: {token:20s} -> max_type_idx={max_idx}, prob={max_val:.3f}"
            )
    except Exception as exc:
        class MockTokenizer:
            def __init__(self) -> None:
                self._last_tokens: list[str] = []

            def __call__(
                self,
                text: str,
                return_offsets_mapping: bool = True,
                add_special_tokens: bool = False,
            ) -> dict[str, object]:
                tokens: list[str] = []
                offsets: list[tuple[int, int]] = []
                for m in re.finditer(r"\S+", text):
                    s, e = m.span()
                    chunk = text[s:e]
                    if len(chunk) > 4:
                        mid = s + len(chunk) // 2
                        tokens.extend([chunk[: mid - s], chunk[mid - s :]])
                        offsets.extend([(s, mid), (mid, e)])
                    else:
                        tokens.append(chunk)
                        offsets.append((s, e))
                self._last_tokens = tokens
                return {"input_ids": list(range(len(tokens))), "offset_mapping": offsets}

            def convert_ids_to_tokens(self, input_ids: list[int]) -> list[str]:
                return [self._last_tokens[i] for i in input_ids]

        print(f"transformers tokenizer unavailable ({exc}); using mock tokenizer")
        tok = MockTokenizer()
        test_text = "Patient has chest pain and takes 500 mg"
        dists = get_token_distributions(test_text, tok, mock_dist, default)
        print(f"Text: {test_text}")
        print(f"Num tokens: {dists.shape[0]}")
        print(f"Dist dim: {dists.shape[1]}")
        encoding = tok(test_text, return_offsets_mapping=True, add_special_tokens=False)
        ids = _to_int_list(encoding.get("input_ids", []))
        tokens = tok.convert_ids_to_tokens(ids)
        for i, (token, dist) in enumerate(zip(tokens, dists)):
            max_idx = int(dist.argmax().item())
            max_val = dist[max_idx].item()
            print(
                f"  Token {i}: {token:20s} -> max_type_idx={max_idx}, prob={max_val:.3f}"
            )
