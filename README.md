# EDEF: Entity Distribution Embedding Fusion for Clinical NER

Boost clinical Named Entity Recognition F1 beyond **85.07%** by fusing per-token entity type distribution vectors directly into Qwen3-4B's embedding space — without adding any information to the prompt.

## Architecture

```
                          ┌──────────────────────┐
   Clinical Text ────────►│  Qwen3 Tokenizer     │
                          └──────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐          ┌────────▼────────┐
              │ embed_     │          │ Distribution    │
              │ tokens()   │          │ Alignment       │
              │            │          │ (word→subword)  │
              └─────┬──────┘          └────────┬────────┘
                    │ h (2560)                 │ d (45)
                    │                   ┌──────▼──────┐
                    │                   │ MLP Projector│
                    │                   │ 45→2560→2560 │
                    │                   └──────┬──────┘
                    │                          │ proj(d) (2560)
                    │                   ┌──────▼──────┐
                    ├──────────────────►│ Gated Fusion │
                    │                   │ σ(W·[h;p])·p │
                    │                   └──────┬──────┘
                    │                          │
                    └──────► h' = h + gate·p ◄─┘
                                 │
                          ┌──────▼──────┐
                          │ Transformer │
                          │ Layers 0-35 │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   LM Head   │
                          └─────────────┘
```

**Key idea**: For each word in the training data, we precompute P(entity_type | word) — a 45-dimensional probability distribution (44 entity types + "O"). During training and inference, this distribution is projected into the model's embedding space via a learnable MLP and fused with the token embeddings through a learned gate. The model learns to leverage this signal without any prompt overhead.

## Project Structure

```
Soft Prompt Tuning/
│
├── qwen3-phase1-checkpoint/       # Phase 1 fine-tuned Qwen3-4B (from S3)
│
├── entity_distributions.json      # 287K word→45-dim distribution lookup (67MB)
├── entity_type_index.json         # 45 entity type → index mapping
├── distribution_stats.json        # Distribution statistics
│
├── build_entity_distributions.py  # Step 1: Build distributions from training data
├── distribution_alignment.py      # Step 2: Map word distributions → subword tokens
├── edef_modules.py                # Step 3: EntityDistProjector + GatedFusion nn.Modules
├── edef_model.py                  # Step 4: attach_edef_to_model(), save/load checkpoints
├── edef_data.py                   # Step 5: EDEFDataset + EDEFDataCollator
├── train_stage1.py                # Step 6: Stage 1 training (projector alignment)
├── train_stage2.py                # Step 7: Stage 2 training (joint LoRA + EDEF)
├── evaluate_edef.py               # Step 8: EDEF evaluation + ablation
├── edef_inference.py              # Step 9: Inference pipeline (EDEFInferencePipeline)
├── evaluate_baseline.py           # Step 10: Baseline re-evaluation for fair comparison
│
├── dry_run_cpu.py                 # CPU validation test suite
├── README.md                      # This file
└── AGENTS.md                      # Agent continuation guide
```

## Requirements

```
torch>=2.0
transformers>=4.45
peft>=0.10
accelerate
bitsandbytes
```

Tested with: Python 3.13, torch 2.9.0, transformers 4.51.0, peft 0.18.1

## Quick Start

### 1. Build Entity Distributions (already done)

Distributions are precomputed from the Phase 1 training data. To rebuild:

```bash
python build_entity_distributions.py \
    --train_path "/path/to/train.json" \
    --output_dir "."
```

**Output**: `entity_distributions.json` (287,948 words), `entity_type_index.json` (45 types)

### 2. Stage 1 Training — Projector Alignment

Freezes Qwen3-4B and trains only the EDEF modules (~19.78M params) to learn useful projections.

```bash
python train_stage1.py \
    --phase1_model qwen3-phase1-checkpoint \
    --train_data "/path/to/train_ner_filtered.json" \
    --val_data "/path/to/val_ner_filtered.json" \
    --output_dir saves/edef-stage1 \
    --epochs 3 \
    --lr 1e-3
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Trainable params | 19.78M | Projector (6.67M) + Gate (13.11M) |
| LR | 1e-3 | Higher LR since only EDEF trains |
| Epochs | 3 | Projector alignment |
| Optimizer | AdamW | Standard |
| Effective batch | 32 | batch=4 × grad_accum=8 |

### 3. Stage 2 Training — Joint LoRA + EDEF

Loads the Stage 1 checkpoint and jointly trains LoRA adapters + EDEF modules.

```bash
python train_stage2.py \
    --phase1_model qwen3-phase1-checkpoint \
    --stage1_checkpoint saves/edef-stage1/edef_checkpoint \
    --train_data "/path/to/train_ner_filtered.json" \
    --val_data "/path/to/val_ner_filtered.json" \
    --output_dir saves/edef-stage2 \
    --epochs 2 \
    --lr 2e-4
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA | r=32, alpha=64, DoRA | Matches Phase 1 config |
| LoRA targets | q/k/v/o/gate/up/down_proj | All attention + FFN |
| EDEF modules | modules_to_save | Full params, not LoRA'd |
| Optimizer | AdamW 8-bit | Saves VRAM |
| Epochs | 2 | Joint fine-tuning |

### 4. Evaluation

```bash
# EDEF evaluation (with ablation)
python evaluate_edef.py \
    --phase1_model qwen3-phase1-checkpoint \
    --model_path saves/edef-stage2 \
    --test_data "/path/to/test_ner_filtered.json" \
    --output_dir evaluation_results

# Baseline re-evaluation (fair comparison)
python evaluate_baseline.py \
    --model_path qwen3-phase1-checkpoint \
    --test_data "/path/to/test_ner_filtered.json" \
    --output_dir evaluation_results
```

The EDEF evaluation script automatically:
- Runs inference with EDEF fusion
- Runs ablation (gate forced to zero) for comparison
- Reports exact + relaxed F1, per-entity-type breakdown, gate statistics
- Saves results to `evaluation_results/edef_results.json`

### 5. Inference

**CLI**:
```bash
python edef_inference.py \
    --model_path saves/edef-stage2 \
    --phase1_model qwen3-phase1-checkpoint \
    --dist_path entity_distributions.json \
    --input "Patient presents with chest pain and shortness of breath."
```

**Python API**:
```python
from edef_inference import EDEFInferencePipeline

pipeline = EDEFInferencePipeline.from_pretrained(
    model_path="saves/edef-stage2",
    phase1_model="qwen3-phase1-checkpoint",
    dist_path="entity_distributions.json",
)

result = pipeline.predict("Patient has chest pain and takes aspirin 500 mg daily.")
# {"ner": [["chest pain", "Sign_Symptom"], ["aspirin", "Drug"], ["500 mg", "Dose_Med"]]}
```

## Entity Types (44 + O)

The system recognizes 44 clinical entity types including:

| Category | Types |
|----------|-------|
| **Medications** | Drug, Dose_Med, Route_Med, Frequency_Med, Duration_Med, Unit_Med |
| **Conditions** | Sign_Symptom, Disease_Disorder |
| **Anatomy** | Anatomical_Structure, Body_Location |
| **Procedures** | Diagnostic_Procedure, Therapeutic_Procedure |
| **Lab** | Lab_Value, Lab_Test |
| **Demographics** | Age, Gender, Race_Ethnicity |
| **Temporal** | Date, Time, Duration |

Full list in `entity_type_index.json`.

## Technical Details

### EDEF Module Specs

| Component | Params | Architecture |
|-----------|--------|-------------|
| EntityDistProjector | 6.67M | Linear(45→2560) → GELU → Linear(2560→2560) |
| GatedFusion | 13.11M | σ(Linear(5120→2560)) × proj(d), bias init=-2.0 |
| **Total EDEF** | **19.78M** | 0.49% of Qwen3-4B's 4.02B params |

### Gate Initialization

The gate bias is initialized to -2.0, giving sigmoid ≈ 0.12. This means EDEF starts near-identity (minimal perturbation to embeddings) and learns to increase the gate as needed during training.

### Distribution Alignment

Word-level distributions are mapped to subword tokens:
- Each subword of a word gets the **same** distribution vector as the full word
- Special tokens / unknown words get the default distribution (heavily weighted toward "O")
- Distributions always sum to 1.0

### Training Data

| Split | Samples | Source |
|-------|---------|--------|
| Train | 43,970 | NER-filtered Phase 2 |
| Val | 7,319 | NER-filtered Phase 2 |
| Test | 2,272 | NER-filtered Phase 2 |

### Hardware Requirements

| Stage | VRAM | Time (est.) |
|-------|------|-------------|
| Stage 1 | ~20GB | ~1 hour (A40/A100) |
| Stage 2 | ~35GB | ~2 hours (A40/A100) |
| Inference | ~10GB | ~1s/sample |

### Model Loading Order (Critical)

```
1. Load Phase 1 merged model
2. attach_edef_to_model()          ← creates projector + gate
3. Load Stage 1 checkpoint         ← pre-trained projector weights
4. get_peft_model() with LoRA      ← wraps model, modules_to_save keeps EDEF
5. Train / Evaluate / Infer
```

## Research Foundation

This approach draws from:

- **LLaVA** (NeurIPS 2023) — MLP projector for cross-modal alignment
- **GEMNET** (NAACL 2021) — Gated fusion for entity knowledge injection
- **SoftLexicon** (ACL 2020) — Soft gazetteer features for NER
- **DEER** (EMNLP 2025) — Distribution-based entity embeddings

## CPU Dry-Run

To validate the pipeline without GPU:

```bash
python dry_run_cpu.py
```

Tests all modules, data pipeline, tokenizer integration, model loading, PEFT compatibility, and checkpoint save/load.

## File Dependency Graph

```
build_entity_distributions.py
    └── entity_distributions.json, entity_type_index.json

edef_modules.py (standalone)
    ├── EntityDistProjector
    └── GatedFusion

distribution_alignment.py
    └── uses entity_distributions.json

edef_model.py
    └── imports edef_modules.py

edef_data.py
    └── imports distribution_alignment.py

train_stage1.py
    └── imports edef_model.py, edef_data.py

train_stage2.py
    └── imports edef_model.py, edef_data.py, peft

evaluate_edef.py
    └── imports edef_model.py, distribution_alignment.py, evaluate_ner.py

evaluate_baseline.py (standalone, copies eval functions)

edef_inference.py
    └── imports edef_model.py, distribution_alignment.py, peft
```
