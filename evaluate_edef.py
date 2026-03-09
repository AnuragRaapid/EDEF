"""EDEF Model Evaluation Script.

Evaluates the EDEF-enhanced NER model on the test set and compares with baseline.
Includes ablation studies (gate=0) and per-entity-type analysis.
"""

import argparse
import ast
import importlib.util
import importlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    _dist_mod = importlib.import_module(".distribution_alignment", package=__package__)
    _edef_mod = importlib.import_module(".edef_model", package=__package__)
    _dataset_utils_mod = importlib.import_module(".ner_dataset_utils", package=__package__)
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _edef_mod = importlib.import_module("edef_model")
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
attach_edef_to_model = _edef_mod.attach_edef_to_model
load_edef_checkpoint = _edef_mod.load_edef_checkpoint
DEFAULT_DATASET_NAME = _dataset_utils_mod.DEFAULT_DATASET_NAME
DEFAULT_DIST_PATH = _dataset_utils_mod.DEFAULT_DIST_PATH
DEFAULT_PHASE1_MODEL_PATH = _dataset_utils_mod.DEFAULT_PHASE1_MODEL_PATH
load_ner_samples = _dataset_utils_mod.load_ner_samples
load_task_metadata_from_dist_path = _dataset_utils_mod.load_task_metadata_from_dist_path


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _robust_parse_ner_json(raw_text: str) -> list[tuple[str, str]]:
    """Parse NER output handling single-quotes, think tags, and truncation."""
    text = _strip_think_tags(raw_text).strip()
    if not text:
        return []

    # Strip common chat-template prefix artifacts (e.g. "assistant\n")
    text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE).strip()

    for attempt_text in [text]:
        # Try standard json.loads first (double-quoted JSON)
        try:
            data = json.loads(attempt_text)
            if "ner" in data:
                return [(str(e).lower().strip(), str(t).lower().strip()) for e, t in data["ner"]]
            return []
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Try ast.literal_eval for Python dict syntax with single quotes
        try:
            data = ast.literal_eval(attempt_text)
            if isinstance(data, dict) and "ner" in data:
                return [(str(e).lower().strip(), str(t).lower().strip()) for e, t in data["ner"]]
            return []
        except (ValueError, SyntaxError):
            pass

    # Fallback: regex-extract all complete [entity, type] pairs from truncated output
    json_match = re.search(r'\{["\']ner["\']\s*:\s*\[', text)
    if json_match:
        arr_start = json_match.end() - 1
        pairs = re.findall(
            r"""\[['"](.+?)['"],\s*['"](.+?)['"]\]""", text[arr_start:]
        )
        if pairs:
            return [(e.lower().strip(), t.lower().strip()) for e, t in pairs]

    return []


def _load_eval_functions() -> tuple[Any, Any, Any, Any, Any]:
    eval_file = Path(__file__).resolve().parents[1] / "Soft Prompt Tuning" / "evaluate_ner.py"
    spec = importlib.util.spec_from_file_location("evaluate_ner_module", str(eval_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation helpers from {eval_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.parse_ner_json,
        module.exact_match,
        module.relaxed_match,
        module.calculate_metrics,
        module.evaluate_per_type,
    )


parse_ner_json, exact_match, relaxed_match, calculate_metrics, evaluate_per_type = _load_eval_functions()


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_embed_layer(model: Any) -> torch.nn.Module:
    candidates = [
        getattr(getattr(model, "model", None), "embed_tokens", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "embed_tokens", None),
        getattr(
            getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
            "embed_tokens",
            None,
        ),
    ]
    for layer in candidates:
        if isinstance(layer, torch.nn.Module):
            return layer
    raise AttributeError("Could not locate embed_tokens layer on model")


def _resolve_edef_modules(model: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    projector_candidates = [
        getattr(model, "entity_projector", None),
        getattr(getattr(model, "base_model", None), "entity_projector", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "entity_projector", None),
    ]
    gate_candidates = [
        getattr(model, "fusion_gate", None),
        getattr(getattr(model, "base_model", None), "fusion_gate", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "fusion_gate", None),
    ]

    projector = next((m for m in projector_candidates if isinstance(m, torch.nn.Module)), None)
    gate = next((m for m in gate_candidates if isinstance(m, torch.nn.Module)), None)
    if projector is None or gate is None:
        raise AttributeError("Could not resolve entity_projector/fusion_gate on model")
    return projector, gate


def load_edef_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    task_metadata = load_task_metadata_from_dist_path(args.dist_path)
    metadata_dist_dim = int(task_metadata["dist_dim"])
    if args.dist_dim is not None and int(args.dist_dim) != metadata_dist_dim:
        raise ValueError(
            f"--dist_dim={args.dist_dim} does not match metadata dist_dim={metadata_dist_dim} in {args.dist_path}"
        )
    dist_dim = metadata_dist_dim
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.phase1_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    hidden_dim = int(getattr(model.config, "hidden_size", 2560))
    model = attach_edef_to_model(model, dist_dim=dist_dim, hidden_dim=hidden_dim)
    model = PeftModel.from_pretrained(model, args.model_path)

    edef_ckpt = os.path.join(args.model_path, "edef_checkpoint")
    if os.path.exists(edef_ckpt):
        load_edef_checkpoint(model.base_model.model, edef_ckpt)
    else:
        print(f"[WARN] EDEF checkpoint not found at: {edef_ckpt}")

    model.eval()

    tokenizer_path = args.model_path if os.path.exists(args.model_path) else args.phase1_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, task_metadata


def _build_prompt(tokenizer: Any, instruction: str, clinical_text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": clinical_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _prepare_fused_embeddings(
    model: Any,
    input_ids: torch.Tensor,
    dist_vectors: torch.Tensor,
    force_gate_zero: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embed_layer = _resolve_embed_layer(model)
    projector, gate_module = _resolve_edef_modules(model)

    inputs_embeds = embed_layer(input_ids)
    seq_len = inputs_embeds.shape[1]

    if dist_vectors.shape[1] > seq_len:
        dist_vectors = dist_vectors[:, :seq_len, :]
    elif dist_vectors.shape[1] < seq_len:
        pad = torch.zeros(
            dist_vectors.shape[0],
            seq_len - dist_vectors.shape[1],
            dist_vectors.shape[2],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        dist_vectors = torch.cat([dist_vectors.to(inputs_embeds.dtype), pad], dim=1)
    else:
        dist_vectors = dist_vectors.to(inputs_embeds.dtype)

    projected = projector(dist_vectors)
    combined = torch.cat([inputs_embeds, projected], dim=-1)
    gate_net = getattr(gate_module, "gate_net", None)
    if not isinstance(gate_net, torch.nn.Module):
        raise AttributeError("fusion_gate must expose gate_net linear layer")
    gate_vals = torch.sigmoid(gate_net(combined))
    if force_gate_zero:
        gate_vals = torch.zeros_like(gate_vals)
    fused = inputs_embeds + gate_vals * projected
    token_gate = gate_vals.mean(dim=-1)
    return fused, token_gate, dist_vectors


class _EdefEmbedHook:
    """Forward hook on embed_tokens that applies EDEF fusion during the prefill step.

    By hooking the embedding layer instead of passing inputs_embeds to generate(),
    we let generate() work with normal input_ids, avoiding output-format ambiguity.
    The hook only fires for the prefill (full-sequence) call; subsequent single-token
    autoregressive steps pass through unmodified.

    The fusion math matches training exactly (see edef_model.attach_edef_to_model):
        E = embed_tokens(input_ids)          # intercepted by this hook as `output`
        D = align(dist_vectors, seq_len)
        P = projector(D)
        F = fusion_gate(E, P)                # calls GatedFusion.forward
          = E + sigmoid(gate_net(cat[E,P])) * P
    """

    def __init__(
        self,
        projector: torch.nn.Module,
        gate_module: torch.nn.Module,
        dist_vectors: torch.Tensor,
        force_gate_zero: bool,
    ) -> None:
        self.projector = projector
        self.gate_module = gate_module
        self.dist_vectors = dist_vectors
        self.force_gate_zero = force_gate_zero
        self._prefill_done = False

    def _align_dist(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dv = self.dist_vectors
        if dv.shape[1] > seq_len:
            dv = dv[:, :seq_len, :]
        elif dv.shape[1] < seq_len:
            pad = torch.zeros(
                dv.shape[0], seq_len - dv.shape[1], dv.shape[2],
                device=device, dtype=dtype,
            )
            dv = torch.cat([dv.to(dtype), pad], dim=1)
        else:
            dv = dv.to(dtype)
        return dv

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._prefill_done:
            return output
        if output.shape[1] <= 1:
            return output
        self._prefill_done = True

        dv = self._align_dist(output.shape[1], output.device, output.dtype)

        # Match training: projector(D) then fusion_gate(E, P)
        projected = self.projector(dv)

        if self.force_gate_zero:
            # Ablation: skip fusion entirely (gate=0 → fused = E)
            self._token_gate = torch.zeros(output.shape[0], output.shape[1])
            self._aligned_dist = dv.detach().cpu()
            return output

        # Call fusion_gate.forward(E, P) — identical to training path
        fused = self.gate_module(output, projected)

        # Capture gate stats for analysis (recompute gate_vals without gradient)
        with torch.no_grad():
            combined = torch.cat([output, projected], dim=-1)
            gate_net = getattr(self.gate_module, "gate_net", None)
            if gate_net is None:
                gate_net = self.gate_module.gate_net
            gate_vals = torch.sigmoid(gate_net(combined))
            self._token_gate = gate_vals.mean(dim=-1).detach().cpu()
        self._aligned_dist = dv.detach().cpu()

        return fused


def _generate_with_mode(
    model: Any,
    tokenizer: Any,
    instruction: str,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
    force_gate_zero: bool,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    prompt = _build_prompt(tokenizer, instruction, clinical_text)
    device = _model_device(model)
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    dist_vectors = get_token_distributions(
        prompt,
        tokenizer,
        word_entity_dist,
        default_dist,
        dist_dim=len(default_dist),
    ).unsqueeze(0).to(device)

    embed_layer = _resolve_embed_layer(model)
    projector, gate_module = _resolve_edef_modules(model)
    hook = _EdefEmbedHook(projector, gate_module, dist_vectors, force_gate_zero)
    handle = embed_layer.register_forward_hook(hook)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    handle.remove()

    seq = outputs[0]
    input_len = input_ids.shape[1]
    new_tokens = seq[input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

    token_gate = getattr(hook, "_token_gate", torch.zeros(1, input_len))
    aligned_dist = getattr(hook, "_aligned_dist", dist_vectors.cpu())
    return generated.strip(), token_gate, aligned_dist


def edef_generate(
    model: Any,
    tokenizer: Any,
    instruction: str,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    return _generate_with_mode(
        model,
        tokenizer,
        instruction,
        clinical_text,
        word_entity_dist,
        default_dist,
        max_new_tokens=max_new_tokens,
        force_gate_zero=False,
    )


def ablation_generate(
    model: Any,
    tokenizer: Any,
    instruction: str,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
) -> str:
    generated, _, _ = _generate_with_mode(
        model,
        tokenizer,
        instruction,
        clinical_text,
        word_entity_dist,
        default_dist,
        max_new_tokens=max_new_tokens,
        force_gate_zero=True,
    )
    return generated


def _generate_batch_with_mode(
    model: Any,
    tokenizer: Any,
    instruction: str,
    batch_texts: list[str],
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
    force_gate_zero: bool,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
    """Batched generation with left-padding and embedding-hook EDEF fusion.

    Uses a forward hook on embed_tokens to inject EDEF fusion during the
    prefill step, then calls generate(input_ids=...) so that the output
    format is always [input_ids | generated_ids] with reliable slicing.
    """
    if not batch_texts:
        return [], [], []

    prompts = [_build_prompt(tokenizer, instruction, text) for text in batch_texts]
    device = _model_device(model)

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encodings = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    bsz, max_len = input_ids.shape
    dist_dim = len(default_dist)

    dist_vectors = torch.zeros(bsz, max_len, dist_dim, device=device, dtype=torch.float32)
    for idx, prompt in enumerate(prompts):
        dist = get_token_distributions(
            prompt, tokenizer, word_entity_dist, default_dist, dist_dim=dist_dim,
        )
        prompt_len = int(attention_mask[idx].sum().item())
        actual_len = min(dist.shape[0], prompt_len)
        start_pos = max_len - prompt_len
        dist_vectors[idx, start_pos : start_pos + actual_len, :] = dist[:actual_len]

    embed_layer = _resolve_embed_layer(model)
    projector, gate_module = _resolve_edef_modules(model)
    hook = _EdefEmbedHook(projector, gate_module, dist_vectors, force_gate_zero)
    handle = embed_layer.register_forward_hook(hook)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    handle.remove()

    # Compute gate stats from the hook's captured values
    token_gate = getattr(hook, "_token_gate", torch.zeros(bsz, max_len))
    aligned_dist = getattr(hook, "_aligned_dist", dist_vectors.cpu())

    generated_texts: list[str] = []
    per_sample_gates: list[torch.Tensor] = []
    per_sample_dists: list[torch.Tensor] = []

    for idx in range(bsz):
        seq = outputs[idx]
        new_tokens = seq[max_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated_texts.append(text)

        prompt_len = int(attention_mask[idx].sum().item())
        start = max_len - prompt_len
        if token_gate.dim() >= 2 and idx < token_gate.shape[0]:
            per_sample_gates.append(token_gate[idx : idx + 1, start:])
        else:
            per_sample_gates.append(torch.zeros(1, prompt_len))
        if aligned_dist.dim() >= 3 and idx < aligned_dist.shape[0]:
            per_sample_dists.append(aligned_dist[idx : idx + 1, start:])
        else:
            per_sample_dists.append(torch.zeros(1, prompt_len, dist_dim))

    return generated_texts, per_sample_gates, per_sample_dists


def _aggregate_metrics(
    predictions: list[list[tuple[str, str]]],
    golds: list[list[tuple[str, str]]],
    word_margin: int = 2,
) -> dict[str, Any]:
    total_exact_tp = total_exact_fp = total_exact_fn = 0
    total_relaxed_tp = total_relaxed_fp = total_relaxed_fn = 0

    for pred_list, gold_list in zip(predictions, golds):
        tp, fp, fn = exact_match(pred_list, gold_list)
        total_exact_tp += tp
        total_exact_fp += fp
        total_exact_fn += fn

        tp, fp, fn = relaxed_match(pred_list, gold_list, word_margin)
        total_relaxed_tp += tp
        total_relaxed_fp += fp
        total_relaxed_fn += fn

    exact_precision, exact_recall, exact_f1 = calculate_metrics(
        total_exact_tp,
        total_exact_fp,
        total_exact_fn,
    )
    relaxed_precision, relaxed_recall, relaxed_f1 = calculate_metrics(
        total_relaxed_tp,
        total_relaxed_fp,
        total_relaxed_fn,
    )

    return {
        "exact": {
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": exact_f1,
            "tp": total_exact_tp,
            "fp": total_exact_fp,
            "fn": total_exact_fn,
        },
        "relaxed": {
            "precision": relaxed_precision,
            "recall": relaxed_recall,
            "f1": relaxed_f1,
            "tp": total_relaxed_tp,
            "fp": total_relaxed_fp,
            "fn": total_relaxed_fn,
        },
        "per_type": evaluate_per_type(predictions, golds, word_margin),
    }


def _load_entity_type_index(dist_path: str) -> dict[int, str]:
    index_path = Path(dist_path).resolve().with_name("entity_type_index.json")
    if not index_path.exists():
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(v): str(k).lower() for k, v in raw.items()}
    except Exception:
        return {}


def _summarize_gate_stats(
    collected_gate: list[torch.Tensor],
    collected_dists: list[torch.Tensor],
    entity_type_by_idx: dict[int, str],
) -> dict[str, Any]:
    if not collected_gate:
        return {}

    gate_values = torch.cat([g.reshape(-1) for g in collected_gate], dim=0)
    dist_values = torch.cat([d.reshape(-1, d.shape[-1]) for d in collected_dists], dim=0)

    default_idx = int(dist_values.shape[-1] - 1)
    token_type_idx = torch.argmax(dist_values, dim=-1)
    is_entity = token_type_idx != default_idx

    entity_gate = gate_values[is_entity]
    non_entity_gate = gate_values[~is_entity]

    per_type: dict[str, dict[str, float | int]] = {}
    unique_idxs = torch.unique(token_type_idx[is_entity]) if is_entity.any() else torch.tensor([])
    for idx_tensor in unique_idxs:
        idx = int(idx_tensor.item())
        mask = token_type_idx == idx
        vals = gate_values[mask]
        key = entity_type_by_idx.get(idx, f"type_{idx}")
        per_type[key] = {
            "mean_gate": float(vals.mean().item()),
            "std_gate": float(vals.std(unbiased=False).item()),
            "count_tokens": int(vals.numel()),
        }

    return {
        "overall_mean_gate": float(gate_values.mean().item()),
        "entity_tokens_mean_gate": float(entity_gate.mean().item()) if entity_gate.numel() else 0.0,
        "non_entity_tokens_mean_gate": float(non_entity_gate.mean().item()) if non_entity_gate.numel() else 0.0,
        "entity_token_count": int(entity_gate.numel()),
        "non_entity_token_count": int(non_entity_gate.numel()),
        "per_entity_type_mean_gate": per_type,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, task_metadata = load_edef_model(args)
    instruction = str(task_metadata["instruction"])
    word_entity_dist, default_dist, _ngram_dist = load_distributions(args.dist_path)
    entity_type_by_idx = _load_entity_type_index(args.dist_path)
    test_samples = load_ner_samples(
        args.test_data,
        split=args.test_split,
        dataset_revision=args.dataset_revision,
        cache_dir=args.cache_dir,
        instruction=instruction,
    )
    if args.max_samples is not None:
        test_samples = test_samples[: args.max_samples]

    predictions: list[list[tuple[str, str]]] = []
    golds: list[list[tuple[str, str]]] = []
    timings: list[float] = []
    parse_errors = defaultdict(int)

    gate_subset_n = min(len(test_samples), args.gate_samples)
    collected_gate: list[torch.Tensor] = []
    collected_dists: list[torch.Tensor] = []

    num_samples = len(test_samples)
    batch_size = max(1, args.batch_size)
    num_batches = math.ceil(num_samples / batch_size)

    # Raw predictions file -- every sample's output is written here for debugging
    os.makedirs(args.output_dir, exist_ok=True)
    raw_pred_path = os.path.join(args.output_dir, "edef_raw_predictions.jsonl")
    raw_pred_file = open(raw_pred_path, "w", encoding="utf-8")

    print(f"Running EDEF evaluation on {num_samples} samples (batch_size={batch_size})")
    print(f"Raw predictions will be saved to: {raw_pred_path}")
    with tqdm(total=num_samples, desc="EDEF Eval", unit="sample", dynamic_ncols=True) as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = test_samples[start_idx:end_idx]

            batch_texts = [str(s.get("input", "")) for s in batch_samples]
            batch_gold_texts = [str(s.get("output", "")) for s in batch_samples]

            start = time.time()
            gen_texts, batch_gates, batch_dists = _generate_batch_with_mode(
                model, tokenizer, instruction, batch_texts, word_entity_dist, default_dist,
                max_new_tokens=args.max_new_tokens, force_gate_zero=False,
            )
            elapsed = time.time() - start
            per_sample_time = elapsed / len(batch_texts)

            for i, (pred_text, gold_text) in enumerate(zip(gen_texts, batch_gold_texts)):
                sample_idx = start_idx + i
                timings.append(per_sample_time)

                pred_entities = _robust_parse_ner_json(pred_text)
                gold_entities = _robust_parse_ner_json(gold_text)
                if not pred_entities and pred_text.strip():
                    parse_errors["prediction_parse_errors"] += 1
                if not gold_entities and gold_text.strip():
                    parse_errors["gold_parse_errors"] += 1

                predictions.append(pred_entities)
                golds.append(gold_entities)

                if sample_idx < gate_subset_n:
                    collected_gate.append(batch_gates[i])
                    collected_dists.append(batch_dists[i])

                # Write raw prediction for every sample
                raw_pred_file.write(json.dumps({
                    "sample_idx": sample_idx,
                    "raw_prediction": pred_text,
                    "parsed_pred_count": len(pred_entities),
                    "gold_count": len(gold_entities),
                    "gold_text": gold_text,
                }, ensure_ascii=False) + "\n")

                # Print first few samples for quick sanity check
                if sample_idx < 3:
                    print(f"\n  [Sample {sample_idx}] raw (first 200): {pred_text[:200]}")
                    print(f"  [Sample {sample_idx}] parsed preds: {len(pred_entities)}, golds: {len(gold_entities)}")

            pbar.update(len(batch_texts))

            running_tp = sum(
                len(set(p) & set(g)) for p, g in zip(predictions, golds)
            )
            pbar.set_postfix({
                "s/sample": f"{per_sample_time:.2f}",
                "batch": f"{batch_idx + 1}/{num_batches}",
                "running_tp": running_tp,
            })

    raw_pred_file.close()
    print(f"\nRaw predictions saved to: {raw_pred_path}")

    eval_metrics = _aggregate_metrics(predictions, golds, word_margin=2)
    gate_stats = _summarize_gate_stats(collected_gate, collected_dists, entity_type_by_idx)

    exact_f1_pct = eval_metrics["exact"]["f1"] * 100.0
    relaxed_f1_pct = eval_metrics["relaxed"]["f1"] * 100.0

    results: dict[str, Any] = {
        "model_path": args.model_path,
        "phase1_model": args.phase1_model,
        "test_data": args.test_data,
        "test_split": args.test_split,
        "num_samples": len(test_samples),
        "avg_inference_time": (sum(timings) / len(timings)) if timings else 0.0,
        "baseline_f1": args.baseline_f1,
        "edef_exact": eval_metrics["exact"],
        "edef_relaxed": eval_metrics["relaxed"],
        "edef_exact_f1": exact_f1_pct,
        "edef_relaxed_f1": relaxed_f1_pct,
        "per_type": eval_metrics["per_type"],
        "gate_stats": gate_stats,
        "parse_errors": dict(parse_errors),
    }
    if args.baseline_f1 is not None:
        results["improvement_vs_baseline"] = exact_f1_pct - args.baseline_f1

    if args.run_ablation:
        print("\nRunning ablation (gate forced to zero)...")
        ablation_preds: list[list[tuple[str, str]]] = []
        ablation_timings: list[float] = []
        ablation_raw_path = os.path.join(args.output_dir, "ablation_raw_predictions.jsonl")
        ablation_raw_file = open(ablation_raw_path, "w", encoding="utf-8")

        with tqdm(total=num_samples, desc="Ablation", unit="sample", dynamic_ncols=True) as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_texts = [str(s.get("input", "")) for s in test_samples[start_idx:end_idx]]

                start = time.time()
                gen_texts, _, _ = _generate_batch_with_mode(
                    model, tokenizer, instruction, batch_texts, word_entity_dist, default_dist,
                    max_new_tokens=args.max_new_tokens, force_gate_zero=True,
                )
                elapsed = time.time() - start
                per_sample_time = elapsed / len(batch_texts)

                for j, pred_text in enumerate(gen_texts):
                    ablation_timings.append(per_sample_time)
                    parsed = _robust_parse_ner_json(pred_text)
                    ablation_preds.append(parsed)
                    ablation_raw_file.write(json.dumps({
                        "sample_idx": start_idx + j,
                        "raw_prediction": pred_text,
                        "parsed_pred_count": len(parsed),
                    }, ensure_ascii=False) + "\n")

                pbar.update(len(batch_texts))
                pbar.set_postfix({"s/sample": f"{per_sample_time:.2f}"})

        ablation_raw_file.close()
        print(f"Ablation raw predictions saved to: {ablation_raw_path}")

        ablation_metrics = _aggregate_metrics(ablation_preds, golds, word_margin=2)
        results["ablation_exact"] = ablation_metrics["exact"]
        results["ablation_relaxed"] = ablation_metrics["relaxed"]
        results["ablation_exact_f1"] = ablation_metrics["exact"]["f1"] * 100.0
        results["ablation_relaxed_f1"] = ablation_metrics["relaxed"]["f1"] * 100.0
        results["ablation_avg_inference_time"] = (
            sum(ablation_timings) / len(ablation_timings) if ablation_timings else 0.0
        )
        results["edef_minus_ablation_f1"] = exact_f1_pct - results["ablation_exact_f1"]

    output_file = os.path.join(args.output_dir, "edef_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"EDEF Exact F1:   {results['edef_exact_f1']:.2f}%")
    print(f"EDEF Relaxed F1: {results['edef_relaxed_f1']:.2f}%")
    if args.run_ablation:
        print(f"Ablation Exact F1: {results['ablation_exact_f1']:.2f}%")
        print(f"EDEF gain over ablation: +{results['edef_minus_ablation_f1']:.2f}%")
    if args.baseline_f1 is not None:
        print(f"Baseline Exact F1: {args.baseline_f1:.2f}%")
        print(f"Improvement vs baseline: {results['improvement_vs_baseline']:+.2f}%")
    print(f"Results written to: {output_file}")
    print(f"Raw predictions:   {raw_pred_path}")
    print("=" * 80)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EDEF-enhanced NER model")
    parser.add_argument("--model_path", type=str, default="saves/edef-stage2")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    parser.add_argument("--phase1_model", type=str, default=DEFAULT_PHASE1_MODEL_PATH)
    parser.add_argument(
        "--test_data",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Local JSON path or Hugging Face dataset repo id for test data.",
    )
    parser.add_argument(
        "--dist_path",
        type=str,
        default=DEFAULT_DIST_PATH,
    )
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--dataset_revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference (reduce if OOM)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--baseline_f1", type=float, default=None)
    parser.add_argument("--dist_dim", type=int, default=None)
    parser.add_argument("--gate_samples", type=int, default=100)

    parser.add_argument("--run_ablation", dest="run_ablation", action="store_true")
    parser.add_argument("--no_run_ablation", dest="run_ablation", action="store_false")
    parser.set_defaults(run_ablation=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

