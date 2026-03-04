# AGENTS.md — EDEF Project Continuation Guide

## Project Identity

**Name**: EDEF (Entity Distribution Embedding Fusion) for Clinical NER
**Location**: `/home/anurag/NER/Soft Prompt Tuning/`
**Goal**: Boost clinical NER F1 from 85.07% to 86–88% by fusing per-token entity type distribution vectors into Qwen3-4B's embedding space.
**Status**: All 10 implementation files complete and CPU-validated. Awaiting GPU training.

---

## Current State (as of March 2026)

### What's Done

| # | File | Lines | Status | Description |
|---|------|-------|--------|-------------|
| 1 | `build_entity_distributions.py` | 310 | ✅ Ran + verified | Builds word→P(entity_type\|word) lookup from training data |
| 2 | `distribution_alignment.py` | 260 | ✅ Smoke tested | Maps word-level distributions to subword tokens |
| 3 | `edef_modules.py` | ~100 | ✅ Smoke tested | EntityDistProjector (45→2560) + GatedFusion (gate init=-2.0) |
| 4 | `edef_model.py` | 240 | ✅ Smoke tested | `attach_edef_to_model()`, `save/load_edef_checkpoint()`, `EDEFWrapper` |
| 5 | `edef_data.py` | 330 | ✅ Smoke tested | `EDEFDataset`, `EDEFDataCollator`, `build_edef_dataset()` |
| 6 | `train_stage1.py` | 245 | ✅ Verified | Stage 1: freeze LLM, train projector+gate only |
| 7 | `train_stage2.py` | 311 | ✅ Verified | Stage 2: joint LoRA + EDEF training |
| 8 | `evaluate_edef.py` | 524 | ✅ Verified | EDEF eval + ablation (gate=0) + gate statistics |
| 9 | `edef_inference.py` | 323 | ✅ Bug fixed + verified | `EDEFInferencePipeline` class + CLI |
| 10 | `evaluate_baseline.py` | 404 | ✅ Verified | Baseline re-evaluation for fair F1 comparison |
| — | `dry_run_cpu.py` | 786 | ✅ All passing | Comprehensive CPU validation test suite |

### Generated Data Files

| File | Size | Contents |
|------|------|----------|
| `entity_distributions.json` | 67MB | 287,948 word → 45-dim probability distributions |
| `entity_type_index.json` | 943B | 45 entity type name → index mapping (0-44, "O"=44) |
| `distribution_stats.json` | 34KB | Per-type top words and statistics |

### What's NOT Done

1. **GPU Training** — No GPU available on current machine. Training requires:
   - Stage 1: ~20GB VRAM, ~1 hour
   - Stage 2: ~35GB VRAM, ~2 hours
2. **Evaluation** — Depends on trained model
3. **Hyperparameter tuning** — May need iteration after first results

---

## Architecture Overview

```
Input Text → Tokenizer → embed_tokens(input_ids) = h
                       → get_token_distributions() = d (45-dim per token)
                       → EntityDistProjector(d) = proj(d) (2560-dim)
                       → GatedFusion: h' = h + σ(W·[h; proj(d)]) · proj(d)
                       → Transformer layers → LM Head → NER output
```

**Key numbers**:
- Model: Qwen3-4B (4.02B params, hidden_dim=2560, 36 layers)
- EDEF: 19.78M params (6.67M projector + 13.11M gate)
- Distribution dim: 45 (44 entity types + "O")
- Gate init: bias=-2.0 → sigmoid≈0.12 (starts near-identity)

---

## How to Continue This Project

### Scenario 1: Run Training on GPU Machine

```bash
# Navigate to project
cd /home/anurag/NER/Soft\ Prompt\ Tuning

# Stage 1: Projector alignment (3 epochs, ~1hr)
python train_stage1.py \
    --phase1_model qwen3-phase1-checkpoint \
    --output_dir saves/edef-stage1

# Stage 2: Joint LoRA + EDEF (2 epochs, ~2hrs)
python train_stage2.py \
    --phase1_model qwen3-phase1-checkpoint \
    --stage1_checkpoint saves/edef-stage1/edef_checkpoint \
    --output_dir saves/edef-stage2

# Evaluate EDEF
python evaluate_edef.py \
    --phase1_model qwen3-phase1-checkpoint \
    --model_path saves/edef-stage2

# Evaluate baseline (for comparison)
python evaluate_baseline.py \
    --model_path qwen3-phase1-checkpoint
```

### Scenario 2: Results Are Below Target (F1 < 86%)

Ordered fallback strategies:

1. **Increase Stage 2 epochs** to 3–4 (first thing to try)
2. **Deeper projector**: Change `edef_modules.py` to 3-layer MLP (45→1280→2560→2560)
3. **Different injection layer**: Instead of after embed_tokens, try after layer 4 or 8
4. **Multi-layer injection**: Inject at layers 0, 12, 24 simultaneously
5. **Learnable per-type embeddings**: Replace distribution vectors with learned type embeddings
6. **Higher LoRA rank**: Try r=64, alpha=128 in Stage 2

### Scenario 3: Training Crashes (OOM)

- Reduce `--batch_size` from 4 to 2
- Reduce `--max_length` from 4096 to 2048
- Enable gradient checkpointing (already enabled in scripts)
- Use `--grad_accum 16` to compensate for smaller batch

### Scenario 4: Modify EDEF Architecture

All EDEF logic is in 3 files:
- **`edef_modules.py`** — Change projector architecture or gate mechanism here
- **`edef_model.py`** — Change injection point (which layer) or forward logic
- **`edef_data.py`** — Change how distributions are prepared for training

---

## Critical Implementation Details

### Model Loading Order (DO NOT CHANGE)

```python
# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(phase1_model, ...)

# 2. Attach EDEF (creates projector + gate, matches model dtype)
model = attach_edef_to_model(model, dist_dim=45, hidden_dim=2560)

# 3. Load Stage 1 checkpoint (if Stage 2)
load_edef_checkpoint(model, stage1_checkpoint_path)

# 4. Apply LoRA (if Stage 2) — EDEF modules go in modules_to_save
lora_config = LoraConfig(..., modules_to_save=["entity_projector", "fusion_gate"])
model = get_peft_model(model, lora_config)
```

### Dtype Matching

`attach_edef_to_model()` automatically detects the model's dtype (bf16/fp16/fp32) and creates EDEF modules in the same dtype. This was a bug that was fixed — do not create EDEF modules separately without matching dtype.

### Why No Packing

EDEF requires per-token distribution vectors aligned with input tokens. Packing (concatenating multiple samples) would break this alignment. All training scripts have packing disabled.

### Why Custom Trainer

HuggingFace `Trainer` strips unknown keys from inputs by default. `EDEFTrainer` overrides `compute_loss` to ensure `entity_dist_vectors` passes through to the model's forward method. The `remove_unused_columns=False` flag in TrainingArguments is also critical.

### Inference with EDEF

`model.generate()` doesn't support `entity_dist_vectors`. The workaround:
1. Compute embeddings manually via `embed_tokens(input_ids)`
2. Project and fuse distribution vectors
3. Pass `inputs_embeds=fused` to `model.generate()` (NOT `input_ids`)

This is implemented in `edef_inference.py` and `evaluate_edef.py`.

---

## File Dependency Map

```
build_entity_distributions.py  (standalone — run first)
    ├── entity_distributions.json
    ├── entity_type_index.json
    └── distribution_stats.json

edef_modules.py  (standalone — no imports)
    ├── EntityDistProjector
    └── GatedFusion

distribution_alignment.py  (uses entity_distributions.json)
    ├── load_distributions()
    └── get_token_distributions()

edef_model.py  (imports edef_modules)
    ├── attach_edef_to_model()
    ├── save_edef_checkpoint()
    ├── load_edef_checkpoint()
    └── EDEFWrapper

edef_data.py  (imports distribution_alignment)
    ├── EDEFDataset
    ├── EDEFDataCollator
    └── build_edef_dataset()

train_stage1.py  (imports edef_model, edef_data)
train_stage2.py  (imports edef_model, edef_data, peft)
evaluate_edef.py  (imports edef_model, distribution_alignment, evaluate_ner)
evaluate_baseline.py  (standalone — copies eval functions)
edef_inference.py  (imports edef_model, distribution_alignment, peft)
```

---

## Data Paths

| Data | Path | Samples |
|------|------|---------|
| Phase 1 model | `./qwen3-phase1-checkpoint/` | Qwen3-4B merged weights |
| Phase 1 train | `/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase1 Dataset/train.json` | 165K (multi-task) |
| Phase 2 train (NER) | `/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/train_ner_filtered.json` | 43,970 |
| Phase 2 val (NER) | `/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/val_ner_filtered.json` | 7,319 |
| Phase 2 test (NER) | `/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/test_ner_filtered.json` | 2,272 |
| Entity types | `/home/anurag/NER/Multi-task Finetuning/Labels_NER.txt` | 44 types |
| Existing eval code | `/home/anurag/NER/Multi-task Finetuning/evaluate_ner.py` | Reference |
| Existing inference | `/home/anurag/NER/Multi-task Finetuning/vllm_inference.py` | Reference |

---

## NER Instruction (EXACT — do not modify)

```
You are an expert medical Named Entity Recognition (NER) assistant. Your task is to extract and classify entities from the provided medical text. Output format should be {'ner': [['entity', 'type'], ['entity', 'type'],...]}
```

## NER Output Format

```json
{"ner": [["chest pain", "Sign_Symptom"], ["aspirin", "Drug"], ["500 mg", "Dose_Med"]]}
```

---

## Training Configuration Reference

### Stage 1 (Projector Alignment)

| Parameter | Value |
|-----------|-------|
| Trainable | EntityDistProjector + GatedFusion only (19.78M) |
| LLM | Frozen |
| LR | 1e-3 |
| Epochs | 3 |
| Optimizer | AdamW |
| Batch | 4 × 8 grad_accum = 32 effective |
| Scheduler | Cosine with 10% warmup |
| Max length | 4096 |
| Save steps | 500 |

### Stage 2 (Joint LoRA + EDEF)

| Parameter | Value |
|-----------|-------|
| Trainable | LoRA (r=32, alpha=64, DoRA) + EDEF (modules_to_save) |
| LoRA targets | q/k/v/o/gate/up/down_proj |
| LR | 2e-4 |
| Epochs | 2 |
| Optimizer | AdamW 8-bit |
| Batch | 4 × 8 grad_accum = 32 effective |
| Scheduler | Cosine with 10% warmup |
| Max length | 4096 |
| Save steps | 300 |

---

## Evaluation Methodology

- **Exact match F1**: Entity text + type must match exactly (after lowering + stripping)
- **Relaxed match F1**: Allows partial overlap (word_margin=2)
- **Baseline F1**: 85.07% (exact match)
- **Ablation**: Same model, same prompt, but gate forced to zero → isolates EDEF contribution
- **Per-entity-type**: Breakdown by all 44 types

---

## Known Issues & Notes

1. **CPU training is too slow**: 4B model forward+backward on CPU takes >5 minutes per step. Use GPU.
2. **Tokenizer vocab_size discrepancy**: `tokenizer.vocab_size` returns 151643 (base vocab) while `config.json` says 151936 (padded). Not a functional issue — padding is handled correctly.
3. **Chat template format**: Phase 1 used user+assistant (no system role). EDEF uses system+user+assistant. This is intentional since we're retraining.
4. **No vLLM support**: EDEF requires custom embedding fusion that vLLM doesn't support. Use standard HuggingFace generate.

---

## Research Papers Referenced

| Paper | Year | Contribution to EDEF |
|-------|------|---------------------|
| LLaVA | NeurIPS 2023 | MLP projector design for cross-modal alignment |
| GEMNET | NAACL 2021 | Gated fusion mechanism for entity knowledge |
| SoftLexicon | ACL 2020 | Soft gazetteer features for NER |
| DEER | EMNLP 2025 | Distribution-based entity embeddings for retrieval |
| Llama SLayer | 2024 | Layer-specific injection strategies |

---

## Quick Validation

Run the CPU dry-run test to verify everything works:

```bash
cd /home/anurag/NER/Soft\ Prompt\ Tuning
python dry_run_cpu.py
```

Expected: 48+ tests pass, covering file verification, module tests, distribution alignment, dataset building, model loading, EDEF attachment, PEFT compatibility, and checkpoint save/load.
