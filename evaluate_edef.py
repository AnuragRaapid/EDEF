"""EDEF Model Evaluation Script.

Evaluates the EDEF-enhanced NER model on the test set and compares with baseline.
Includes ablation studies (gate=0) and per-entity-type analysis.
"""

import argparse
import importlib.util
import importlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    _dist_mod = importlib.import_module(".distribution_alignment", package=__package__)
    _edef_mod = importlib.import_module(".edef_model", package=__package__)
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _edef_mod = importlib.import_module("edef_model")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
attach_edef_to_model = _edef_mod.attach_edef_to_model
load_edef_checkpoint = _edef_mod.load_edef_checkpoint


NER_INSTRUCTION = (
    "You are an expert medical Named Entity Recognition (NER) assistant. "
    "Your task is to extract and classify entities from the provided medical text. "
    "Output format should be {'ner': [['entity', 'type'], ['entity', 'type'],...]}"
)


def _load_eval_functions() -> tuple[Any, Any, Any, Any, Any]:
    eval_file = Path(__file__).resolve().parents[1] / "Multi-task Finetuning" / "evaluate_ner.py"
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


def load_edef_model(args: argparse.Namespace) -> tuple[Any, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.phase1_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    hidden_dim = int(getattr(model.config, "hidden_size", 2560))
    model = attach_edef_to_model(model, dist_dim=args.dist_dim, hidden_dim=hidden_dim)
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

    return model, tokenizer


def _build_prompt(tokenizer: Any, clinical_text: str) -> str:
    messages = [
        {"role": "system", "content": NER_INSTRUCTION},
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


def _generate_with_mode(
    model: Any,
    tokenizer: Any,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
    force_gate_zero: bool,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    prompt = _build_prompt(tokenizer, clinical_text)
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

    with torch.no_grad():
        fused, token_gate, aligned_dist = _prepare_fused_embeddings(
            model,
            input_ids=input_ids,
            dist_vectors=dist_vectors,
            force_gate_zero=force_gate_zero,
        )
        outputs = model.generate(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seq = outputs[0]
    if seq.shape[0] > input_ids.shape[1]:
        new_tokens = seq[input_ids.shape[1] :]
    else:
        new_tokens = seq
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated.strip(), token_gate.detach().cpu(), aligned_dist.detach().cpu()


def edef_generate(
    model: Any,
    tokenizer: Any,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    return _generate_with_mode(
        model,
        tokenizer,
        clinical_text,
        word_entity_dist,
        default_dist,
        max_new_tokens=max_new_tokens,
        force_gate_zero=False,
    )


def ablation_generate(
    model: Any,
    tokenizer: Any,
    clinical_text: str,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    max_new_tokens: int,
) -> str:
    generated, _, _ = _generate_with_mode(
        model,
        tokenizer,
        clinical_text,
        word_entity_dist,
        default_dist,
        max_new_tokens=max_new_tokens,
        force_gate_zero=True,
    )
    return generated


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
    model, tokenizer = load_edef_model(args)
    word_entity_dist, default_dist = load_distributions(args.dist_path)
    entity_type_by_idx = _load_entity_type_index(args.dist_path)

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    if args.max_samples is not None:
        test_samples = test_samples[: args.max_samples]

    predictions: list[list[tuple[str, str]]] = []
    golds: list[list[tuple[str, str]]] = []
    timings: list[float] = []
    parse_errors = defaultdict(int)

    gate_subset_n = min(len(test_samples), args.gate_samples)
    collected_gate: list[torch.Tensor] = []
    collected_dists: list[torch.Tensor] = []

    print(f"Running EDEF evaluation on {len(test_samples)} samples")
    for i, sample in enumerate(test_samples):
        clinical_text = str(sample.get("input", ""))
        gold_output = str(sample.get("output", ""))

        start = time.time()
        pred_output, token_gate, aligned_dist = edef_generate(
            model,
            tokenizer,
            clinical_text,
            word_entity_dist,
            default_dist,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - start
        timings.append(elapsed)

        pred_entities = parse_ner_json(pred_output)
        gold_entities = parse_ner_json(gold_output)
        if not pred_entities and pred_output.strip():
            parse_errors["prediction_parse_errors"] += 1
        if not gold_entities and gold_output.strip():
            parse_errors["gold_parse_errors"] += 1

        predictions.append(pred_entities)
        golds.append(gold_entities)

        if i < gate_subset_n:
            collected_gate.append(token_gate)
            collected_dists.append(aligned_dist)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_samples)} samples ({elapsed:.2f}s/sample)")

    eval_metrics = _aggregate_metrics(predictions, golds, word_margin=2)
    gate_stats = _summarize_gate_stats(collected_gate, collected_dists, entity_type_by_idx)

    exact_f1_pct = eval_metrics["exact"]["f1"] * 100.0
    relaxed_f1_pct = eval_metrics["relaxed"]["f1"] * 100.0

    results: dict[str, Any] = {
        "model_path": args.model_path,
        "phase1_model": args.phase1_model,
        "test_data": args.test_data,
        "num_samples": len(test_samples),
        "avg_inference_time": (sum(timings) / len(timings)) if timings else 0.0,
        "baseline_f1": args.baseline_f1,
        "edef_exact": eval_metrics["exact"],
        "edef_relaxed": eval_metrics["relaxed"],
        "edef_exact_f1": exact_f1_pct,
        "edef_relaxed_f1": relaxed_f1_pct,
        "improvement_vs_baseline": exact_f1_pct - args.baseline_f1,
        "per_type": eval_metrics["per_type"],
        "gate_stats": gate_stats,
        "parse_errors": dict(parse_errors),
    }

    if args.run_ablation:
        print("Running ablation (gate forced to zero)...")
        ablation_preds: list[list[tuple[str, str]]] = []
        ablation_timings: list[float] = []
        for i, sample in enumerate(test_samples):
            start = time.time()
            pred = ablation_generate(
                model,
                tokenizer,
                str(sample.get("input", "")),
                word_entity_dist,
                default_dist,
                max_new_tokens=args.max_new_tokens,
            )
            ablation_timings.append(time.time() - start)
            ablation_preds.append(parse_ner_json(pred))
            if (i + 1) % 100 == 0:
                print(f"  Ablation processed {i + 1}/{len(test_samples)}")

        ablation_metrics = _aggregate_metrics(ablation_preds, golds, word_margin=2)
        results["ablation_exact"] = ablation_metrics["exact"]
        results["ablation_relaxed"] = ablation_metrics["relaxed"]
        results["ablation_exact_f1"] = ablation_metrics["exact"]["f1"] * 100.0
        results["ablation_relaxed_f1"] = ablation_metrics["relaxed"]["f1"] * 100.0
        results["ablation_avg_inference_time"] = (
            sum(ablation_timings) / len(ablation_timings) if ablation_timings else 0.0
        )
        results["edef_minus_ablation_f1"] = exact_f1_pct - results["ablation_exact_f1"]

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "edef_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"EDEF Exact F1:   {results['edef_exact_f1']:.2f}%")
    print(f"EDEF Relaxed F1: {results['edef_relaxed_f1']:.2f}%")
    if args.run_ablation:
        print(f"Ablation Exact F1: {results['ablation_exact_f1']:.2f}%")
        print(f"EDEF gain over ablation: +{results['edef_minus_ablation_f1']:.2f}%")
    print(f"Baseline Exact F1: {args.baseline_f1:.2f}%")
    print(f"Improvement vs baseline: {results['improvement_vs_baseline']:+.2f}%")
    print(f"Results written to: {output_file}")
    print("=" * 80)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EDEF-enhanced NER model")
    parser.add_argument("--model_path", type=str, default="saves/edef-stage2")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct")
    parser.add_argument("--phase1_model", type=str, default="saves/phase1_merged")
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/test_ner_filtered.json",
    )
    parser.add_argument(
        "--dist_path",
        type=str,
        default="/home/anurag/NER/Soft Prompt Tuning/entity_distributions.json",
    )
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--baseline_f1", type=float, default=85.07)
    parser.add_argument("--dist_dim", type=int, default=45)
    parser.add_argument("--gate_samples", type=int, default=100)

    parser.add_argument("--run_ablation", dest="run_ablation", action="store_true")
    parser.add_argument("--no_run_ablation", dest="run_ablation", action="store_false")
    parser.set_defaults(run_ablation=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
