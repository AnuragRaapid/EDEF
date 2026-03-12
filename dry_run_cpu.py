"""CPU smoke test for the late-correction EDEF redesign."""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Callable

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from edef_data import EDEFDataCollator, EDEFDataset, _MockTokenizer
from edef_model import (
    attach_edef_to_model,
    edef_runtime_context,
    get_edef_host,
    get_last_edef_stats,
    load_edef_checkpoint,
    save_edef_checkpoint,
)
from train_stage1 import EDEFTrainer as Stage1Trainer
from train_stage1 import GateLoggingCallback as Stage1GateCallback
from train_stage2 import EDEFTrainer as Stage2Trainer
from train_stage2 import GateLoggingCallback as Stage2GateCallback
from train_stage2 import (
    _set_corrector_warmup_trainable_params as set_corrector_warmup_trainable_params,
)
from train_stage2 import _set_stage2_trainable_params as set_stage2_trainable_params

DIST_DIM = 45
SAMPLE_TEXT = "Patient presents with chest pain and shortness of breath. Prescribed aspirin 500 mg daily."
NER_INSTRUCTION = (
    "You are an expert medical Named Entity Recognition (NER) assistant. "
    "Your task is to extract and classify entities from the provided medical text. "
    "Output format should be {'ner': [['entity', 'type'], ['entity', 'type'],...]}"
)

PASSED = 0
FAILED = 0
ERRORS: list[tuple[str, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    icon = "PASS" if ok else "FAIL"
    if ok:
        PASSED += 1
    else:
        FAILED += 1
        ERRORS.append((name, detail))
    suffix = f" - {detail}" if detail else ""
    print(f"  [{icon}] {name}{suffix}")


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def flush_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_tiny_qwen3(vocab_size: int = 512) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    model = Qwen3ForCausalLM(config)
    model.config.pad_token_id = 0
    return model


def build_mock_dataset() -> tuple[EDEFDataset, EDEFDataCollator, _MockTokenizer]:
    tokenizer = _MockTokenizer()
    samples = [
        {
            "instruction": NER_INSTRUCTION,
            "input": "Patient has chest pain and takes aspirin 500 mg daily.",
            "output": '{"ner": [["chest pain", "Sign_Symptom"], ["aspirin", "Drug"], ["500 mg", "Dose_Med"]]}',
        },
        {
            "instruction": NER_INSTRUCTION,
            "input": "Shortness of breath improved after albuterol treatment.",
            "output": '{"ner": [["shortness of breath", "Sign_Symptom"], ["albuterol", "Drug"]]}',
        },
        {
            "instruction": NER_INSTRUCTION,
            "input": "Blood pressure is stable and no fever is reported.",
            "output": '{"ner": [["blood pressure", "Lab_Test"], ["fever", "Sign_Symptom"]]}',
        },
        {
            "instruction": NER_INSTRUCTION,
            "input": "Patient reports sodium 140 and started metformin.",
            "output": '{"ner": [["sodium", "Lab_Test"], ["140", "Lab_Value"], ["metformin", "Drug"]]}',
        },
    ]
    default_dist = [0.0] * (DIST_DIM - 1) + [1.0]
    word_entity_dist = {
        "chest": [0.0] * 18 + [0.8] + [0.0] * 25 + [0.2],
        "pain": [0.0] * 24 + [0.9] + [0.0] * 19 + [0.1],
        "aspirin": [0.0] * 9 + [0.9] + [0.0] * 34 + [0.1],
        "500": [0.0] * 7 + [0.9] + [0.0] * 36 + [0.1],
        "mg": [0.0] * 7 + [0.8] + [0.0] * 36 + [0.2],
        "shortness": [0.0] * 24 + [0.8] + [0.0] * 19 + [0.2],
        "breath": [0.0] * 24 + [0.8] + [0.0] * 19 + [0.2],
        "albuterol": [0.0] * 9 + [0.9] + [0.0] * 34 + [0.1],
        "blood": [0.0] * 30 + [0.8] + [0.0] * 13 + [0.2],
        "pressure": [0.0] * 30 + [0.8] + [0.0] * 13 + [0.2],
        "fever": [0.0] * 24 + [0.9] + [0.0] * 19 + [0.1],
        "sodium": [0.0] * 30 + [0.85] + [0.0] * 13 + [0.15],
        "140": [0.0] * 29 + [0.9] + [0.0] * 14 + [0.1],
        "metformin": [0.0] * 9 + [0.9] + [0.0] * 34 + [0.1],
    }
    dataset = EDEFDataset(
        samples=samples,
        tokenizer=tokenizer,
        word_entity_dist=word_entity_dist,
        default_dist=default_dist,
        max_length=128,
        dist_dim=DIST_DIM,
    )
    collator = EDEFDataCollator(tokenizer=tokenizer, max_length=128)
    return dataset, collator, tokenizer


def build_batch(
    dataset: EDEFDataset, collator: EDEFDataCollator
) -> dict[str, torch.Tensor]:
    return collator([dataset[0], dataset[1]])


def move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def snapshot_named_parameters(
    model: Any,
    name_filter: Callable[[str, torch.nn.Parameter], bool],
) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name_filter(name, param)
    }


def compute_deltas(model: Any, snapshot: dict[str, torch.Tensor]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    named_params = dict(model.named_parameters())
    for name, before in snapshot.items():
        after = named_params[name].detach()
        deltas[name] = float((after - before).abs().max().item())
    return deltas


def load_last_jsonl_record(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [line.strip() for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL records found in {path}")
    return json.loads(rows[-1])


def run_manual_step(
    model: Any,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> tuple[float, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    grad_norms = {
        name: float(param.grad.norm().item())
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item()), grad_norms


def build_lora_config(insertion_layer: int) -> LoraConfig:
    return LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        layers_to_transform=list(range(insertion_layer + 1, 4)),
        layers_pattern="layers",
        modules_to_save=["entity_projector", "fusion_gate", "late_corrector"],
        lora_dropout=0.0,
        bias="none",
        use_dora=False,
        task_type="CAUSAL_LM",
    )


def test_modules_and_dataset() -> None:
    section("Phase 1: Dataset And Late Modules")
    dataset, collator, _ = build_mock_dataset()
    sample = dataset[0]
    batch = build_batch(dataset, collator)

    report("Sample includes prompt mask", "entity_prompt_mask" in sample)
    report(
        "Prompt mask has active tokens",
        bool(sample["entity_prompt_mask"].any().item()),
        f"prompt_tokens={int(sample['entity_prompt_mask'].sum().item())}",
    )
    report(
        "Assistant-side distributions are zeroed",
        bool(
            (sample["entity_dist_vectors"][~sample["entity_prompt_mask"]])
            .abs()
            .sum()
            .item()
            == 0.0
        ),
    )
    report(
        "Batch prompt mask collates correctly",
        batch["entity_prompt_mask"].dtype == torch.bool
        and batch["entity_prompt_mask"].shape[:2] == batch["input_ids"].shape,
        f"shape={tuple(batch['entity_prompt_mask'].shape)}",
    )


def test_stage1_manual_and_trainer(export_dir: str) -> str:
    section("Phase 2: Stage 1 Gradient Flow")
    dataset, collator, _ = build_mock_dataset()
    batch = build_batch(dataset, collator)

    model = build_tiny_qwen3()
    for param in model.parameters():
        param.requires_grad = False
    model = attach_edef_to_model(
        model,
        dist_dim=DIST_DIM,
        hidden_dim=model.config.hidden_size,
        insertion_layer=1,
        projector_bottleneck_dim=16,
        projector_use_temperature=True,
        fusion_projected_norm=True,
        corrector_layers=0,
        corrector_dim=32,
        corrector_heads=4,
        edef_dtype=torch.float32,
    )
    host = get_edef_host(model)
    for param in host.entity_projector.parameters():
        param.requires_grad = True
    for param in host.fusion_gate.parameters():
        param.requires_grad = True
    report(
        "Stage 1 projector temperature enabled",
        host.entity_projector.log_temperature is not None,
    )
    report(
        "Stage 1 fusion normalization enabled",
        not isinstance(host.fusion_gate.projected_norm, torch.nn.Identity),
    )

    device = next(model.parameters()).device
    batch = move_batch_to_device(batch, device)
    before = snapshot_named_parameters(model, lambda _name, param: param.requires_grad)
    frozen_before = next(model.model.layers[0].parameters()).detach().clone()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(host.entity_projector.parameters()), "lr": 1e-3},
            {"params": list(host.fusion_gate.parameters()), "lr": 3e-3},
        ]
    )
    loss, grad_norms = run_manual_step(model, batch, optimizer)
    deltas = compute_deltas(model, before)

    projector_grad = max(v for k, v in grad_norms.items() if "entity_projector" in k)
    gate_grad = max(v for k, v in grad_norms.items() if "fusion_gate" in k)
    report(
        "Stage 1 loss is finite",
        bool(torch.isfinite(torch.tensor(loss)).item()),
        f"loss={loss:.4f}",
    )
    report(
        "Stage 1 projector grad norm > 0", projector_grad > 0, f"{projector_grad:.6f}"
    )
    report("Stage 1 gate grad norm > 0", gate_grad > 0, f"{gate_grad:.6f}")
    report(
        "Stage 1 projector params changed",
        max(v for k, v in deltas.items() if "entity_projector" in k) > 0,
    )
    report(
        "Stage 1 gate params changed",
        max(v for k, v in deltas.items() if "fusion_gate" in k) > 0,
    )
    frozen_after = next(model.model.layers[0].parameters()).detach()
    report(
        "Frozen backbone stayed unchanged",
        bool(torch.allclose(frozen_before, frozen_after)),
    )

    stage1_ckpt = os.path.join(export_dir, "stage1_ckpt")
    save_edef_checkpoint(model, stage1_ckpt)
    saved_state = {
        key: value.detach().clone()
        for key, value in host.entity_projector.state_dict().items()
    }
    with torch.no_grad():
        next(host.entity_projector.parameters()).add_(1.0)
    load_edef_checkpoint(model, stage1_ckpt)
    restored = all(
        torch.allclose(saved_state[key], value)
        for key, value in host.entity_projector.state_dict().items()
    )
    report("Stage 1 save/load round-trip", restored)

    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=1e-3,
            bf16=False,
            fp16=False,
            no_cuda=True,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )
        trainer = Stage1Trainer(
            model=model,
            args=training_args,
            train_dataset=torch.utils.data.Subset(dataset, range(2)),
            data_collator=collator,
            callbacks=[Stage1GateCallback(model=model, log_every_steps=1)],
            projector_learning_rate=1e-3,
            gate_learning_rate=3e-3,
        )
        try:
            trainer.train()
            report("Stage 1 trainer one-step run", True)
            diag_path = os.path.join(tmpdir, "stage1_training_diagnostics.jsonl")
            diag_payload = load_last_jsonl_record(diag_path)
            report("Stage 1 diagnostics file written", os.path.isfile(diag_path))
            report(
                "Stage 1 diagnostics include grad ratio",
                "projector_to_gate_grad_rms_ratio" in diag_payload.get("ratios", {}),
            )
            report(
                "Stage 1 diagnostics include gate update stats",
                "update_rms" in diag_payload.get("groups", {}).get("gate", {}),
            )
        except Exception as exc:
            report("Stage 1 trainer one-step run", False, str(exc))

    flush_memory()
    return stage1_ckpt


def test_stage2_manual_trainer_and_prefill(stage1_ckpt: str) -> None:
    section("Phase 3: Stage 2 Joint Update")
    dataset, collator, tokenizer = build_mock_dataset()
    batch = build_batch(dataset, collator)

    model = build_tiny_qwen3()
    model = attach_edef_to_model(
        model,
        dist_dim=DIST_DIM,
        hidden_dim=model.config.hidden_size,
        insertion_layer=1,
        projector_bottleneck_dim=16,
        projector_use_temperature=True,
        fusion_projected_norm=True,
        corrector_layers=2,
        corrector_dim=32,
        corrector_heads=4,
        edef_dtype=torch.float32,
    )
    load_edef_checkpoint(model, stage1_ckpt)
    peft_model = get_peft_model(model, build_lora_config(insertion_layer=1))
    warmup_ready = set_corrector_warmup_trainable_params(peft_model)
    warmup_trainable = [
        name
        for name, param in peft_model.named_parameters()
        if param.requires_grad
        and any(
            token in name
            for token in ("late_corrector", "fusion_gate", "entity_projector", "lora_")
        )
    ]
    report(
        "Corrector warmup isolates corrector params",
        warmup_ready
        and warmup_trainable
        and all("late_corrector" in name for name in warmup_trainable),
        f"trainable={len(warmup_trainable)}",
    )
    set_stage2_trainable_params(peft_model)

    device = next(peft_model.parameters()).device
    batch = move_batch_to_device(batch, device)
    before = snapshot_named_parameters(
        peft_model, lambda _name, param: param.requires_grad
    )

    edef_params = [
        param
        for name, param in peft_model.named_parameters()
        if param.requires_grad
        and any(
            token in name
            for token in ("entity_projector", "fusion_gate", "late_corrector")
        )
    ]
    lora_params = [
        param
        for name, param in peft_model.named_parameters()
        if param.requires_grad and "lora_" in name
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": edef_params, "lr": 1e-3},
            {"params": lora_params, "lr": 2e-4},
        ]
    )
    loss, grad_norms = run_manual_step(peft_model, batch, optimizer)
    deltas = compute_deltas(peft_model, before)

    projector_grad = max(v for k, v in grad_norms.items() if "entity_projector" in k)
    gate_grad = max(v for k, v in grad_norms.items() if "fusion_gate" in k)
    corrector_grad = max(v for k, v in grad_norms.items() if "late_corrector" in k)
    lora_grad = max(v for k, v in grad_norms.items() if "lora_" in k)
    report(
        "Stage 2 loss is finite",
        bool(torch.isfinite(torch.tensor(loss)).item()),
        f"loss={loss:.4f}",
    )
    report(
        "Stage 2 projector grad norm > 0", projector_grad > 0, f"{projector_grad:.6f}"
    )
    report("Stage 2 gate grad norm > 0", gate_grad > 0, f"{gate_grad:.6f}")
    report(
        "Stage 2 corrector grad norm > 0", corrector_grad > 0, f"{corrector_grad:.6f}"
    )
    report("Stage 2 LoRA grad norm > 0", lora_grad > 0, f"{lora_grad:.6f}")
    report(
        "Stage 2 projector params changed",
        max(v for k, v in deltas.items() if "entity_projector" in k) > 0,
    )
    report(
        "Stage 2 gate params changed",
        max(v for k, v in deltas.items() if "fusion_gate" in k) > 0,
    )
    report(
        "Stage 2 corrector params changed",
        max(v for k, v in deltas.items() if "late_corrector" in k) > 0,
    )
    report(
        "Stage 2 LoRA params changed",
        max(v for k, v in deltas.items() if "lora_" in k) > 0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        stage2_ckpt = os.path.join(tmpdir, "stage2_ckpt")
        save_edef_checkpoint(peft_model, stage2_ckpt)

        fresh_model = build_tiny_qwen3()
        fresh_model = attach_edef_to_model(
            fresh_model,
            dist_dim=DIST_DIM,
            hidden_dim=fresh_model.config.hidden_size,
            insertion_layer=1,
            projector_bottleneck_dim=16,
            projector_use_temperature=True,
            fusion_projected_norm=True,
            corrector_layers=2,
            corrector_dim=32,
            corrector_heads=4,
            edef_dtype=torch.float32,
        )
        fresh_peft = get_peft_model(fresh_model, build_lora_config(insertion_layer=1))
        load_edef_checkpoint(fresh_peft, stage2_ckpt)

        saved_host = get_edef_host(peft_model)
        fresh_host = get_edef_host(fresh_peft)
        parity = all(
            torch.allclose(value, fresh_host.late_corrector.state_dict()[key])
            for key, value in saved_host.late_corrector.state_dict().items()
        )
        report("Stage 2 corrector save/load parity", parity)

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": NER_INSTRUCTION},
                {"role": "user", "content": SAMPLE_TEXT},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_encoding = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_encoding["input_ids"].to(device)
        prompt_attention = prompt_encoding["attention_mask"].to(device)
        prompt_dists = torch.zeros(
            (1, prompt_ids.shape[1], DIST_DIM), device=device, dtype=torch.float32
        )
        prompt_dists[:, : min(6, prompt_ids.shape[1]), -1] = 1.0
        prompt_mask = torch.zeros(
            (1, prompt_ids.shape[1]), dtype=torch.bool, device=device
        )
        prompt_mask[:, : min(6, prompt_ids.shape[1])] = True
        with torch.inference_mode():
            with edef_runtime_context(
                peft_model,
                entity_dist_vectors=prompt_dists,
                entity_prompt_mask=prompt_mask,
                prefill_only=True,
            ):
                generated = peft_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attention,
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=0,
                )
        stats = get_last_edef_stats(peft_model)
        report("Prefill-only late hook executed", stats["token_gate"] is not None)
        report(
            "Generation completed with late hook",
            generated.shape[1] > prompt_ids.shape[1],
            f"generated_tokens={generated.shape[1] - prompt_ids.shape[1]}",
        )

        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=2e-4,
            bf16=False,
            fp16=False,
            no_cuda=True,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )
        trainer = Stage2Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=torch.utils.data.Subset(dataset, range(2)),
            data_collator=collator,
            callbacks=[Stage2GateCallback(model=peft_model, log_every_steps=1)],
            lora_learning_rate=2e-4,
            edef_learning_rate=1e-3,
        )
        try:
            trainer.train()
            report("Stage 2 trainer one-step run", True)
            diag_path = os.path.join(tmpdir, "stage2_training_diagnostics.jsonl")
            diag_payload = load_last_jsonl_record(diag_path)
            report("Stage 2 diagnostics file written", os.path.isfile(diag_path))
            report(
                "Stage 2 diagnostics include grad ratio",
                "projector_to_gate_grad_rms_ratio" in diag_payload.get("ratios", {}),
            )
            report(
                "Stage 2 diagnostics include LoRA stats",
                "lora" in diag_payload.get("groups", {}),
            )
        except Exception as exc:
            report("Stage 2 trainer one-step run", False, str(exc))

    flush_memory()


def main() -> None:
    print("\n" + "=" * 72)
    print("  Late Correction EDEF CPU Smoke Test")
    print("=" * 72)
    start = time.time()

    try:
        test_modules_and_dataset()
    except Exception:
        report("Phase 1 crashed", False, traceback.format_exc())

    try:
        with tempfile.TemporaryDirectory() as export_dir:
            stage1_ckpt = test_stage1_manual_and_trainer(export_dir)
            test_stage2_manual_trainer_and_prefill(stage1_ckpt)
    except Exception:
        report("Training/inference smoke", False, traceback.format_exc())

    total_time = time.time() - start
    section("Summary")
    print(f"  Passed: {PASSED}")
    print(f"  Failed: {FAILED}")
    print(f"  Time:   {total_time:.1f}s")
    if ERRORS:
        print("\n  Failures:")
        for name, detail in ERRORS:
            print(f"  - {name}: {detail[:200]}")

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
