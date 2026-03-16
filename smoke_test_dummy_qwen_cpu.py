from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from typing import Any, Callable

import torch
from peft import LoraConfig, get_peft_model
from transformers import BertConfig, BertModel, LlamaConfig, LlamaForCausalLM

from distribution_alignment import load_distributions
from edef_data import EDEFDataCollator, EDEFDataset, _MockTokenizer
from edef_model import (
    apply_medical_encoder_lora,
    attach_edef_to_model,
    build_fused_input_embeddings,
    load_edef_checkpoint,
    save_edef_checkpoint,
)
from ner_dataset_utils import load_ner_samples, load_task_metadata_from_dist_path

DEFAULT_TRAIN_DATA = (
    "/home/anurag/NER/Multi-task Finetuning/"
    "Multitask Finetuning Phase 2 Dataset/train_ner_filtered.json"
)
DEFAULT_DIST_PATH = "/home/anurag/NER/Soft Prompt Tuning/entity_distributions.json"

warnings.filterwarnings(
    "ignore",
    message=r"Could not find a config file in .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are trying to modify a model with PEFT for a second time.*",
    category=UserWarning,
)


def report(name: str, ok: bool, detail: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    suffix = f" | {detail}" if detail else ""
    print(f"[{icon}] {name}{suffix}")
    if not ok:
        raise AssertionError(name if not detail else f"{name}: {detail}")


def _max_grad_value(
    model: torch.nn.Module, predicate: Callable[[str, torch.nn.Parameter], bool]
) -> float:
    max_value = 0.0
    for name, param in model.named_parameters():
        if not predicate(name, param):
            continue
        if param.grad is None:
            continue
        max_value = max(max_value, float(param.grad.detach().abs().max().item()))
    return max_value


def _any_grad_present(
    model: torch.nn.Module, predicate: Callable[[str, torch.nn.Parameter], bool]
) -> bool:
    for name, param in model.named_parameters():
        if predicate(name, param) and param.grad is not None:
            if bool(param.grad.detach().abs().sum().item() > 0):
                return True
    return False


def _snapshot_params(
    model: torch.nn.Module,
    predicate: Callable[[str, torch.nn.Parameter], bool],
) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if predicate(name, param):
            snapshot[name] = param.detach().clone()
    return snapshot


def _params_changed(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> bool:
    for name, param in model.named_parameters():
        before = snapshot.get(name)
        if before is None:
            continue
        if not torch.allclose(before, param.detach()):
            return True
    return False


def _make_dummy_qwen(vocab_size: int, hidden_size: int = 96) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


def _make_dummy_medical_encoder(vocab_size: int, hidden_size: int = 48) -> BertModel:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=512,
        pad_token_id=0,
    )
    return BertModel(config)


def _load_small_sample_slice(
    data_path: str, dist_path: str, num_samples: int
) -> tuple[list[dict[str, Any]], str]:
    task_metadata = load_task_metadata_from_dist_path(dist_path)
    instruction = str(task_metadata["instruction"])
    samples = load_ner_samples(data_path, instruction=instruction)
    filtered = [
        sample
        for sample in samples
        if len(str(sample.get("input", "")).strip()) >= 10
        and len(str(sample.get("output", "")).strip()) >= 10
    ]
    ranked = sorted(
        filtered,
        key=lambda sample: (
            len(str(sample.get("input", ""))) + len(str(sample.get("output", "")))
        ),
    )
    return ranked[:num_samples], instruction


def _build_distribution_batch(
    samples: list[dict[str, Any]],
    prompt_tokenizer: _MockTokenizer,
    dist_path: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    word_entity_dist, default_dist = load_distributions(dist_path)
    dataset = EDEFDataset(
        samples=samples,
        tokenizer=prompt_tokenizer,
        signal_source="distribution",
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        dist_dim=len(default_dist),
        max_length=max_length,
    )
    collator = EDEFDataCollator(tokenizer=prompt_tokenizer, max_length=max_length)
    return collator([dataset[0], dataset[1]])


def _build_medical_batch(
    samples: list[dict[str, Any]],
    prompt_tokenizer: _MockTokenizer,
    medical_tokenizer: _MockTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor]:
    dataset = EDEFDataset(
        samples=samples,
        tokenizer=prompt_tokenizer,
        signal_source="medical_encoder",
        medical_tokenizer=medical_tokenizer,
        max_length=max_length,
        medical_chunk_size=12,
        medical_chunk_overlap=4,
        max_prompt_medical_tokens=3,
    )
    collator = EDEFDataCollator(tokenizer=prompt_tokenizer, max_length=max_length)
    return collator([dataset[0], dataset[1]])


def _optimizer_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=1e-2,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Expected a scalar loss from the dummy forward pass.")
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def run_distribution_stage1_check(
    *,
    batch: dict[str, torch.Tensor],
    vocab_size: int,
) -> None:
    model = _make_dummy_qwen(vocab_size=vocab_size)
    for param in model.parameters():
        param.requires_grad = False
    model = attach_edef_to_model(
        model,
        dist_dim=int(batch["entity_dist_vectors"].shape[-1]),
        hidden_dim=int(model.config.hidden_size),
    )
    for param in model.entity_projector.parameters():
        param.requires_grad = True
    for param in model.fusion_gate.parameters():
        param.requires_grad = True

    projector_snapshot = _snapshot_params(
        model,
        lambda name, param: "entity_projector" in name,
    )
    loss = _optimizer_step(model, batch)
    report(
        "Distribution Stage 1 loss is finite",
        torch.isfinite(torch.tensor(loss)).item(),
        f"loss={loss:.4f}",
    )
    report(
        "Distribution projector gradients flow",
        _max_grad_value(model, lambda name, param: "entity_projector" in name) > 0,
    )
    report(
        "Distribution gate gradients flow",
        _max_grad_value(model, lambda name, param: "fusion_gate" in name) > 0,
    )
    report(
        "Frozen Qwen base stays frozen",
        not _any_grad_present(
            model,
            lambda name, param: (
                "entity_projector" not in name
                and "fusion_gate" not in name
                and param.requires_grad
            ),
        ),
    )
    report(
        "Distribution optimizer updates projector",
        _params_changed(model, projector_snapshot),
    )


def run_medical_stage1_check(
    *,
    batch: dict[str, torch.Tensor],
    vocab_size: int,
) -> None:
    model = _make_dummy_qwen(vocab_size=vocab_size)
    model = attach_edef_to_model(
        model,
        hidden_dim=int(model.config.hidden_size),
        signal_source="medical_encoder",
        medical_encoder_model_name="dummy-clinicalbert",
        medical_encoder_model=_make_dummy_medical_encoder(vocab_size=vocab_size),
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.entity_projector.parameters():
        param.requires_grad = True
    for param in model.fusion_gate.parameters():
        param.requires_grad = True
    model = apply_medical_encoder_lora(
        model,
        r=4,
        alpha=8,
        target_modules=["query", "value"],
        top_layers=2,
        use_dora=False,
    )

    projector_snapshot = _snapshot_params(
        model,
        lambda name, param: (
            "entity_projector" in name
            or ("medical_encoder" in name and "lora_" in name)
        ),
    )
    with torch.no_grad():
        fused_embeds, projected = build_fused_input_embeddings(
            model,
            input_ids=batch["input_ids"],
            medical_chunk_input_ids=batch["medical_chunk_input_ids"],
            medical_chunk_attention_mask=batch["medical_chunk_attention_mask"],
            medical_chunk_token_indices=batch["medical_chunk_token_indices"],
            medical_chunk_token_weights=batch["medical_chunk_token_weights"],
            prompt_medical_token_indices=batch["prompt_medical_token_indices"],
            prompt_medical_token_weights=batch["prompt_medical_token_weights"],
            medical_token_count=batch["medical_token_count"],
        )
    report(
        "Medical fused embeddings shape matches inputs",
        fused_embeds.shape[:2] == batch["input_ids"].shape and projected is not None,
        f"fused={tuple(fused_embeds.shape)} projected={None if projected is None else tuple(projected.shape)}",
    )

    loss = _optimizer_step(model, batch)
    report(
        "Medical Stage 1 loss is finite",
        torch.isfinite(torch.tensor(loss)).item(),
        f"loss={loss:.4f}",
    )
    report(
        "Medical projector gradients flow",
        _max_grad_value(model, lambda name, param: "entity_projector" in name) > 0,
    )
    report(
        "Medical gate gradients flow",
        _max_grad_value(model, lambda name, param: "fusion_gate" in name) > 0,
    )
    report(
        "Medical encoder LoRA gradients flow",
        _max_grad_value(
            model, lambda name, param: "medical_encoder" in name and "lora_" in name
        )
        > 0,
    )
    report(
        "Frozen Qwen decoder stays frozen in Stage 1",
        not _any_grad_present(
            model,
            lambda name, param: (
                "entity_projector" not in name
                and "fusion_gate" not in name
                and "medical_encoder" not in name
                and param.requires_grad
            ),
        ),
    )
    report(
        "Medical Stage 1 optimizer updates trainable params",
        _params_changed(model, projector_snapshot),
    )

    with tempfile.TemporaryDirectory(prefix="dummy-medical-stage1-") as tmpdir:
        save_edef_checkpoint(model, tmpdir)
        reloaded = _make_dummy_qwen(vocab_size=vocab_size)
        reloaded = attach_edef_to_model(
            reloaded,
            hidden_dim=int(reloaded.config.hidden_size),
            signal_source="medical_encoder",
            medical_encoder_model_name="dummy-clinicalbert",
            medical_encoder_model=_make_dummy_medical_encoder(vocab_size=vocab_size),
        )
        load_edef_checkpoint(reloaded, tmpdir, medical_encoder_trainable=True)
        with torch.no_grad():
            reloaded_out = reloaded(**batch)
        report(
            "Medical checkpoint reload forward works",
            hasattr(reloaded_out, "logits"),
            f"logits={tuple(reloaded_out.logits.shape)}",
        )


def run_medical_stage2_check(
    *,
    batch: dict[str, torch.Tensor],
    vocab_size: int,
) -> None:
    model = _make_dummy_qwen(vocab_size=vocab_size)
    model = attach_edef_to_model(
        model,
        hidden_dim=int(model.config.hidden_size),
        signal_source="medical_encoder",
        medical_encoder_model_name="dummy-clinicalbert",
        medical_encoder_model=_make_dummy_medical_encoder(vocab_size=vocab_size),
    )
    model = apply_medical_encoder_lora(
        model,
        r=4,
        alpha=8,
        target_modules=["query", "value"],
        top_layers=2,
        use_dora=False,
    )
    qwen_lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["entity_projector", "fusion_gate"],
        lora_dropout=0.0,
        bias="none",
        use_dora=False,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, qwen_lora_config)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    stage2_snapshot = _snapshot_params(
        model,
        lambda name, param: (
            "entity_projector" in name or "fusion_gate" in name or "lora_" in name
        ),
    )
    loss = _optimizer_step(model, batch)
    report(
        "Medical Stage 2 loss is finite",
        torch.isfinite(torch.tensor(loss)).item(),
        f"loss={loss:.4f}",
    )
    report(
        "Stage 2 Qwen LoRA gradients flow",
        _max_grad_value(
            model, lambda name, param: "lora_" in name and "medical_encoder" not in name
        )
        > 0,
    )
    report(
        "Stage 2 medical LoRA gradients flow",
        _max_grad_value(
            model, lambda name, param: "medical_encoder" in name and "lora_" in name
        )
        > 0,
    )
    report(
        "Stage 2 projector gradients flow",
        _max_grad_value(model, lambda name, param: "entity_projector" in name) > 0,
    )
    report(
        "Stage 2 gate gradients flow",
        _max_grad_value(model, lambda name, param: "fusion_gate" in name) > 0,
    )
    report(
        "Stage 2 optimizer updates trainable params",
        _params_changed(model, stage2_snapshot),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU smoke test for dummy Qwen-style medical fusion."
    )
    parser.add_argument("--data-path", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--dist-path", default=DEFAULT_DIST_PATH)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    if not os.path.isfile(args.dist_path):
        raise FileNotFoundError(f"Distribution file not found: {args.dist_path}")

    samples, instruction = _load_small_sample_slice(
        args.data_path,
        args.dist_path,
        args.num_samples,
    )
    report("Loaded small dataset slice", len(samples) >= 2, f"samples={len(samples)}")
    report("Loaded instruction", bool(instruction), instruction[:80])

    prompt_tokenizer = _MockTokenizer()
    medical_tokenizer = _MockTokenizer()
    vocab_size = 65536

    distribution_batch = _build_distribution_batch(
        samples=samples,
        prompt_tokenizer=prompt_tokenizer,
        dist_path=args.dist_path,
        max_length=args.max_length,
    )
    medical_batch = _build_medical_batch(
        samples=samples,
        prompt_tokenizer=prompt_tokenizer,
        medical_tokenizer=medical_tokenizer,
        max_length=args.max_length,
    )
    report(
        "Distribution batch ready",
        "entity_dist_vectors" in distribution_batch,
        f"shape={tuple(distribution_batch['entity_dist_vectors'].shape)}",
    )
    report(
        "Medical batch ready",
        "medical_chunk_input_ids" in medical_batch,
        f"shape={tuple(medical_batch['medical_chunk_input_ids'].shape)}",
    )

    run_distribution_stage1_check(batch=distribution_batch, vocab_size=vocab_size)
    run_medical_stage1_check(batch=medical_batch, vocab_size=vocab_size)
    run_medical_stage2_check(batch=medical_batch, vocab_size=vocab_size)
    print("\nAll dummy CPU smoke tests passed.")


if __name__ == "__main__":
    main()
