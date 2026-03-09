"""CPU Dry-Run: End-to-end EDEF pipeline validation.

Tests every component of the EDEF pipeline on CPU without GPU.
Uses the real Phase 1 checkpoint, real tokenizer, and real data.
Designed for memory-constrained environments (~15GB RAM).
"""

import gc
import json
import os
import sys
import time
import traceback

# ── Paths ────────────────────────────────────────────────────────────────────
PHASE1_MODEL = "/home/anurag/NER/Soft Prompt Tuning/qwen3-phase1-checkpoint"
DIST_PATH = "/home/anurag/NER/Soft Prompt Tuning/entity_distributions.json"
TYPE_INDEX_PATH = "/home/anurag/NER/Soft Prompt Tuning/entity_type_index.json"
TRAIN_DATA = "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/train_ner_filtered.json"
VAL_DATA = "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/val_ner_filtered.json"
TEST_DATA = "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset/test_ner_filtered.json"

SAMPLE_TEXT = "Patient presents with chest pain and shortness of breath. Prescribed aspirin 500 mg daily."

PASSED = 0
FAILED = 0
ERRORS = []


def report(name: str, ok: bool, detail: str = ""):
    global PASSED, FAILED
    icon = "✅" if ok else "❌"
    if ok:
        PASSED += 1
    else:
        FAILED += 1
        ERRORS.append((name, detail))
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{suffix}")


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def flush_memory():
    gc.collect()
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: File & Data Verification (no model loading)
# ═══════════════════════════════════════════════════════════════════════════

def test_phase1_files():
    section("Phase 1: File & Data Verification")

    # Check checkpoint exists
    report("Phase 1 checkpoint dir exists",
           os.path.isdir(PHASE1_MODEL),
           PHASE1_MODEL)

    # Check model files
    safetensor_files = [f for f in os.listdir(PHASE1_MODEL) if f.endswith(".safetensors")]
    report("Safetensor model files present",
           len(safetensor_files) >= 2,
           f"Found {len(safetensor_files)} safetensor files")

    # Check tokenizer files
    tok_config = os.path.join(PHASE1_MODEL, "tokenizer_config.json")
    report("Tokenizer config present", os.path.isfile(tok_config))

    # Check config.json
    config_path = os.path.join(PHASE1_MODEL, "config.json")
    report("Model config.json present", os.path.isfile(config_path))
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
        report("Hidden size = 2560", config.get("hidden_size") == 2560, f"got {config.get('hidden_size')}")
        report("Model type = qwen3", config.get("model_type") == "qwen3")
        report("Vocab size = 151936", config.get("vocab_size") == 151936)

    # Check entity distributions
    report("entity_distributions.json exists",
           os.path.isfile(DIST_PATH),
           f"{os.path.getsize(DIST_PATH) / 1e6:.1f} MB" if os.path.isfile(DIST_PATH) else "MISSING")

    # Check entity_type_index.json
    report("entity_type_index.json exists", os.path.isfile(TYPE_INDEX_PATH))
    if os.path.isfile(TYPE_INDEX_PATH):
        with open(TYPE_INDEX_PATH) as f:
            type_index = json.load(f)
        report("45 entity types (44 + O)",
               len(type_index) == 45,
               f"got {len(type_index)}")

    # Check training data
    for label, path in [("Train data", TRAIN_DATA), ("Val data", VAL_DATA), ("Test data", TEST_DATA)]:
        exists = os.path.isfile(path)
        if exists:
            with open(path) as f:
                data = json.load(f)
            report(f"{label} exists", True, f"{len(data)} samples")
        else:
            report(f"{label} exists", False, "MISSING")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2: Module-Level Tests (lightweight, no model loading)
# ═══════════════════════════════════════════════════════════════════════════

def test_phase2_modules():
    section("Phase 2: EDEF Module Tests (no model loading)")

    import torch
    from edef_modules import EntityDistProjector, GatedFusion

    # Test EntityDistProjector
    proj = EntityDistProjector(dist_dim=45, hidden_dim=2560)
    param_count = sum(p.numel() for p in proj.parameters())
    report("EntityDistProjector created",
           param_count > 0,
           f"{param_count:,} params ({param_count/1e6:.2f}M)")

    x = torch.randn(2, 10, 45)  # batch=2, seq=10, dist_dim=45
    out = proj(x)
    report("Projector forward OK",
           out.shape == (2, 10, 2560),
           f"in={x.shape} → out={out.shape}")

    # Test GatedFusion
    gate = GatedFusion(hidden_dim=2560)
    gate_params = sum(p.numel() for p in gate.parameters())
    report("GatedFusion created",
           gate_params > 0,
           f"{gate_params:,} params ({gate_params/1e6:.2f}M)")

    h = torch.randn(2, 10, 2560)
    p = torch.randn(2, 10, 2560)
    fused = gate(h, p)
    report("GatedFusion forward OK",
           fused.shape == h.shape,
           f"output shape={fused.shape}")

    # Check gate init (bias = -2.0 → sigmoid ≈ 0.12)
    gate_bias = gate.gate_net.bias.data
    sigmoid_mean = torch.sigmoid(gate_bias).mean().item()
    report("Gate init near-identity",
           sigmoid_mean < 0.15,
           f"sigmoid(bias) mean={sigmoid_mean:.4f}, expected ~0.12")

    # Gradient flow
    h.requires_grad_(True)
    fused = gate(h, p)
    loss = fused.sum()
    loss.backward()
    report("Gradient flows through gate",
           h.grad is not None and h.grad.abs().sum() > 0)

    del proj, gate, x, out, h, p, fused
    flush_memory()


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3: Distribution Alignment Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_phase3_distribution_alignment():
    section("Phase 3: Distribution Alignment")

    from distribution_alignment import load_distributions, get_token_distributions
    from transformers import AutoTokenizer

    # Load distributions
    t0 = time.time()
    word_entity_dist, default_dist, ngram_dist = load_distributions(DIST_PATH)
    elapsed = time.time() - t0
    report("load_distributions() OK",
           len(word_entity_dist) > 0,
           f"{len(word_entity_dist):,} words loaded in {elapsed:.1f}s")
    report("ngram_dist loaded",
           isinstance(ngram_dist, dict),
           f"{len(ngram_dist):,} n-grams")

    report("default_dist is 45-dim",
           len(default_dist) == 45,
           f"dim={len(default_dist)}")

    # Spot-check known words
    if "pain" in word_entity_dist:
        pain_dist = word_entity_dist["pain"]
        max_idx = pain_dist.index(max(pain_dist))
        report("'pain' distribution loaded",
               max(pain_dist) > 0.5,
               f"max prob={max(pain_dist):.3f} at idx {max_idx}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    report("Tokenizer loaded from checkpoint",
           tokenizer.vocab_size == 151936,
           f"vocab={tokenizer.vocab_size}")

    # Test get_token_distributions
    import torch
    dist_vectors = get_token_distributions(
        SAMPLE_TEXT, tokenizer, word_entity_dist, default_dist, dist_dim=45
    )
    report("get_token_distributions() OK",
           isinstance(dist_vectors, torch.Tensor),
           f"shape={dist_vectors.shape}, dtype={dist_vectors.dtype}")

    tokens = tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)
    report("Token count matches dist vector length",
           dist_vectors.shape[0] == len(tokens),
           f"tokens={len(tokens)}, vectors={dist_vectors.shape[0]}")

    report("Distributions sum to ~1.0",
           (dist_vectors.sum(dim=-1) - 1.0).abs().max().item() < 0.01,
           f"max deviation from 1.0: {(dist_vectors.sum(dim=-1) - 1.0).abs().max().item():.6f}")

    del word_entity_dist, default_dist, dist_vectors, tokenizer
    flush_memory()


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4: Dataset & Collator Tests (uses tokenizer + data, no model)
# ═══════════════════════════════════════════════════════════════════════════

def test_phase4_dataset():
    section("Phase 4: Dataset & Collator")

    from transformers import AutoTokenizer
    from edef_data import build_edef_dataset, EDEFDataCollator

    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset (limit to 5 samples for speed)
    t0 = time.time()
    dataset = build_edef_dataset(
        data_path=TRAIN_DATA,
        tokenizer=tokenizer,
        dist_path=DIST_PATH,
        max_length=512,  # Short for CPU test
        dist_dim=45,
    )
    elapsed = time.time() - t0

    # Check dataset size (should load all but we test a subset)
    full_size = len(dataset)
    report("build_edef_dataset() OK",
           full_size > 0,
           f"{full_size} samples built in {elapsed:.1f}s")

    # Check a sample
    sample = dataset[0]
    required_keys = {"input_ids", "attention_mask", "labels", "entity_dist_vectors"}
    has_keys = required_keys.issubset(set(sample.keys()))
    report("Sample has required keys",
           has_keys,
           f"keys={list(sample.keys())}")

    import torch
    if has_keys:
        report("input_ids is tensor",
               isinstance(sample["input_ids"], torch.Tensor),
               f"shape={sample['input_ids'].shape}")
        report("entity_dist_vectors shape",
               sample["entity_dist_vectors"].shape[-1] == 45,
               f"shape={sample['entity_dist_vectors'].shape}")
        report("Labels have masking (-100)",
               (sample["labels"] == -100).any().item(),
               f"masked={int((sample['labels'] == -100).sum())}/{len(sample['labels'])}")

    # Test collator
    collator = EDEFDataCollator(tokenizer=tokenizer, max_length=512)
    batch = collator([dataset[0], dataset[1]])
    report("EDEFDataCollator produces batch",
           "input_ids" in batch and "entity_dist_vectors" in batch,
           f"batch keys={list(batch.keys())}")

    if "input_ids" in batch:
        report("Batch shapes consistent",
               batch["input_ids"].shape[0] == 2,
               f"batch_size={batch['input_ids'].shape[0]}, seq={batch['input_ids'].shape[1]}")

    if "entity_dist_vectors" in batch:
        report("Dist vectors in batch",
               batch["entity_dist_vectors"].shape[0] == 2 and batch["entity_dist_vectors"].shape[-1] == 45,
               f"shape={batch['entity_dist_vectors'].shape}")

    del dataset, collator, batch, tokenizer
    flush_memory()


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 5: Full Model Loading + EDEF Attach (heavy — needs ~8GB)
# ═══════════════════════════════════════════════════════════════════════════

def test_phase5_model_loading():
    section("Phase 5: Model Loading + EDEF Attach")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from edef_model import attach_edef_to_model, save_edef_checkpoint, load_edef_checkpoint

    # Load model in bf16 to save memory
    print("  ⏳ Loading Qwen3-4B on CPU (bf16, ~8GB)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        PHASE1_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    elapsed = time.time() - t0
    report("Model loaded on CPU",
           model is not None,
           f"in {elapsed:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    report("Model param count",
           total_params > 3e9,
           f"{total_params:,} ({total_params/1e9:.2f}B)")

    # Check embedding layer
    embed = model.model.embed_tokens
    report("embed_tokens accessible",
           embed is not None,
           f"shape={embed.weight.shape}")

    # Attach EDEF
    model = attach_edef_to_model(model, dist_dim=45, hidden_dim=2560, dist_dropout=0.2, use_refiner=True)
    report("attach_edef_to_model() OK",
           hasattr(model, "entity_projector") and hasattr(model, "fusion_gate"))

    edef_params = (
        sum(p.numel() for p in model.entity_projector.parameters()) +
        sum(p.numel() for p in model.fusion_gate.parameters())
    )
    report("EDEF params count",
           edef_params > 19e6,
           f"{edef_params:,} ({edef_params/1e6:.2f}M)")

    # Test forward with entity_dist_vectors
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("Hello world", return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    dist_v = torch.zeros(1, seq_len, 45, dtype=torch.bfloat16)

    print("  ⏳ Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            entity_dist_vectors=dist_v,
        )
    report("Forward with entity_dist_vectors OK",
           hasattr(outputs, "logits"),
           f"logits shape={outputs.logits.shape}")

    # Test forward WITHOUT entity_dist_vectors (should still work)
    with torch.no_grad():
        outputs_no_edef = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    report("Forward WITHOUT entity_dist_vectors OK",
           hasattr(outputs_no_edef, "logits"))

    # Test save/load checkpoint
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "edef_ckpt")
        save_edef_checkpoint(model, ckpt_path)
        saved_files = os.listdir(ckpt_path)
        report("save_edef_checkpoint() OK",
               len(saved_files) >= 2,
               f"files={saved_files}")

        # Load back
        load_edef_checkpoint(model, ckpt_path)
        report("load_edef_checkpoint() OK", True)

    # Test PEFT compatibility
    print("  ⏳ Testing PEFT (LoRA) compatibility...")
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,  # Small r for testing
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["entity_projector", "fusion_gate"],
        lora_dropout=0,
        bias="none",
        use_dora=False,  # Faster for CPU test
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    report("PEFT get_peft_model() with EDEF OK",
           peft_model is not None)

    # Test PEFT forward
    with torch.no_grad():
        peft_out = peft_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            entity_dist_vectors=dist_v,
        )
    report("PEFT forward with entity_dist_vectors OK",
           hasattr(peft_out, "logits"),
           f"logits shape={peft_out.logits.shape}")

    # Cleanup to free memory
    del peft_model, model, tokenizer, outputs, outputs_no_edef
    flush_memory()
    print("  🧹 Memory cleaned after Phase 5")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 6: Training Dry-Run (1 step each)
# ═══════════════════════════════════════════════════════════════════════════

def test_phase6_training():
    section("Phase 6: Training Script Dry-Runs")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from edef_model import attach_edef_to_model, save_edef_checkpoint
    from edef_data import build_edef_dataset, EDEFDataCollator

    # Import the trainers
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_stage1 import EDEFTrainer as S1Trainer, GateLoggingCallback as S1GateCallback
    from train_stage2 import EDEFTrainer as S2Trainer, GateLoggingCallback as S2GateCallback

    report("train_stage1.py imports OK", True)
    report("train_stage2.py imports OK", True)

    # Load model
    print("  ⏳ Loading model for training dry-run...")
    model = AutoModelForCausalLM.from_pretrained(
        PHASE1_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Stage 1 dry-run ──
    print("  ⏳ Stage 1 dry-run: freeze model + train projector...")

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    # Attach EDEF
    model = attach_edef_to_model(model, dist_dim=45, hidden_dim=2560, dist_dropout=0.2, use_refiner=True)
    for param in model.entity_projector.parameters():
        param.requires_grad = True
    for param in model.fusion_gate.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report("Stage 1: Only EDEF trainable",
           trainable < 25e6,
           f"{trainable:,} trainable params")

    # Build tiny dataset
    dataset = build_edef_dataset(
        data_path=TRAIN_DATA,
        tokenizer=tokenizer,
        dist_path=DIST_PATH,
        max_length=128,
        dist_dim=45,
    )
    # Use just 4 samples
    tiny_dataset = torch.utils.data.Subset(dataset, range(min(4, len(dataset))))
    collator = EDEFDataCollator(tokenizer=tokenizer, max_length=128)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=1e-3,
            bf16=False,  # CPU doesn't support bf16 training
            fp16=False,
            no_cuda=True,
            remove_unused_columns=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
        )

        trainer = S1Trainer(
            model=model,
            args=training_args,
            train_dataset=tiny_dataset,
            data_collator=collator,
            callbacks=[S1GateCallback(model=model, log_every_steps=1)],
        )

        t0 = time.time()
        try:
            trainer.train()
            elapsed = time.time() - t0
            report("Stage 1 training step OK", True, f"1 step in {elapsed:.1f}s")
        except Exception as e:
            report("Stage 1 training step OK", False, str(e))

        # Save EDEF checkpoint
        ckpt_path = os.path.join(tmpdir, "edef_ckpt")
        save_edef_checkpoint(model, ckpt_path)
        report("Stage 1 EDEF checkpoint saved",
               os.path.isdir(ckpt_path) and len(os.listdir(ckpt_path)) >= 2)

    # ── Stage 2 dry-run ──
    print("  ⏳ Stage 2 dry-run: LoRA + EDEF joint training...")

    # Unfreeze model for LoRA
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        modules_to_save=["entity_projector", "fusion_gate"],
        lora_dropout=0,
        bias="none",
        use_dora=False,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    peft_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    report("Stage 2: LoRA + EDEF trainable",
           peft_trainable > trainable,
           f"{peft_trainable:,} trainable params")

    with tempfile.TemporaryDirectory() as tmpdir:
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

        trainer = S2Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tiny_dataset,
            data_collator=collator,
            callbacks=[S2GateCallback(model=peft_model, log_every_steps=1)],
        )

        t0 = time.time()
        try:
            trainer.train()
            elapsed = time.time() - t0
            report("Stage 2 training step OK", True, f"1 step in {elapsed:.1f}s")
        except Exception as e:
            report("Stage 2 training step OK", False, str(e))

    del peft_model, model, tokenizer, dataset, tiny_dataset, collator, trainer
    flush_memory()
    print("  🧹 Memory cleaned after Phase 6")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 7: Inference Dry-Run
# ═══════════════════════════════════════════════════════════════════════════

def test_phase7_inference():
    section("Phase 7: Inference & Generation Dry-Run")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from edef_model import attach_edef_to_model
    from distribution_alignment import load_distributions, get_token_distributions

    # Load model
    print("  ⏳ Loading model for inference dry-run...")
    model = AutoModelForCausalLM.from_pretrained(
        PHASE1_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(PHASE1_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = attach_edef_to_model(model, dist_dim=45, hidden_dim=2560, dist_dropout=0.2, use_refiner=True)
    model.eval()

    # Load distributions
    word_entity_dist, default_dist, _ngram_dist = load_distributions(DIST_PATH)
    report("Distributions loaded for inference", len(word_entity_dist) > 0)

    # Build prompt
    NER_INSTRUCTION = (
        "You are an expert medical Named Entity Recognition (NER) assistant. "
        "Your task is to extract and classify entities from the provided medical text. "
        "Output format should be {'ner': [['entity', 'type'], ['entity', 'type'],...]}"
    )
    messages = [
        {"role": "system", "content": NER_INSTRUCTION},
        {"role": "user", "content": SAMPLE_TEXT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    report("Chat template applied", len(prompt) > 0, f"prompt length={len(prompt)} chars")

    # Tokenize
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    report("Tokenized prompt", True, f"tokens={input_ids.shape[1]}")

    # Build distribution vectors
    dist_vectors = get_token_distributions(prompt, tokenizer, word_entity_dist, default_dist, dist_dim=45)
    dist_vectors = dist_vectors.unsqueeze(0).to(dtype=torch.bfloat16)
    report("Distribution vectors built", dist_vectors.shape[1] == input_ids.shape[1],
           f"shape={dist_vectors.shape}")

    # Test fused embeddings approach (same as inference/eval scripts)
    with torch.no_grad():
        # Get embeddings
        inputs_embeds = model.model.embed_tokens(input_ids)
        report("embed_tokens forward OK", inputs_embeds.shape[-1] == 2560)

        # Project + fuse
        projected = model.entity_projector(dist_vectors)
        fused = model.fusion_gate(inputs_embeds, projected)
        report("Fused embeddings computed",
               fused.shape == inputs_embeds.shape,
               f"shape={fused.shape}")

        # Generate with fused embeddings (just 5 tokens for speed)
        print("  ⏳ Generating (5 tokens, CPU — slow is normal)...")
        t0 = time.time()
        output_ids = model.generate(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.time() - t0
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        report("model.generate() with fused embeddings OK",
               len(output_ids[0]) > input_ids.shape[1],
               f"generated '{generated[:50]}...' in {elapsed:.1f}s")

    # Test edef_inference.py module import
    from edef_inference import parse_ner_output
    result = parse_ner_output('{"ner": [["chest pain", "Sign_Symptom"]]}')
    report("parse_ner_output() OK",
           result.get("ner") == [["chest pain", "Sign_Symptom"]])

    # Test evaluate_baseline.py module import
    from evaluate_baseline import parse_ner_json, exact_match, relaxed_match, calculate_metrics
    entities = parse_ner_json('{"ner": [["chest pain", "Sign_Symptom"], ["aspirin", "Drug"]]}')
    report("evaluate_baseline.parse_ner_json() OK",
           len(entities) == 2,
           f"parsed {len(entities)} entities")

    tp, fp, fn = exact_match(entities, entities)
    report("exact_match() OK", tp == 2 and fp == 0 and fn == 0)

    p, r, f1 = calculate_metrics(tp, fp, fn)
    report("calculate_metrics() OK", f1 == 1.0, f"F1={f1}")

    # Test evaluate_edef.py module import
    from evaluate_edef import _aggregate_metrics
    metrics = _aggregate_metrics([entities], [entities])
    report("evaluate_edef._aggregate_metrics() OK",
           metrics["exact"]["f1"] == 1.0)

    del model, tokenizer, word_entity_dist
    flush_memory()
    print("  🧹 Memory cleaned after Phase 7")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "🔬" * 35)
    print("  EDEF Pipeline CPU Dry-Run")
    print("🔬" * 35)

    start = time.time()

    # Phase 1: Files
    try:
        test_phase1_files()
    except Exception as e:
        report("Phase 1 CRASHED", False, traceback.format_exc())

    # Phase 2: Modules
    try:
        test_phase2_modules()
    except Exception as e:
        report("Phase 2 CRASHED", False, traceback.format_exc())

    # Phase 3: Distribution alignment
    try:
        test_phase3_distribution_alignment()
    except Exception as e:
        report("Phase 3 CRASHED", False, traceback.format_exc())

    # Phase 4: Dataset + collator
    try:
        test_phase4_dataset()
    except Exception as e:
        report("Phase 4 CRASHED", False, traceback.format_exc())

    # Phase 5: Model + EDEF + PEFT
    try:
        test_phase5_model_loading()
    except Exception as e:
        report("Phase 5 CRASHED", False, traceback.format_exc())

    # Phase 6: Training
    try:
        test_phase6_training()
    except Exception as e:
        report("Phase 6 CRASHED", False, traceback.format_exc())

    # Phase 7: Inference
    try:
        test_phase7_inference()
    except Exception as e:
        report("Phase 7 CRASHED", False, traceback.format_exc())

    total_time = time.time() - start

    # Summary
    section("SUMMARY")
    print(f"  ✅ Passed: {PASSED}")
    print(f"  ❌ Failed: {FAILED}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")

    if ERRORS:
        print(f"\n  Failures:")
        for name, detail in ERRORS:
            print(f"    ❌ {name}: {detail[:200]}")

    print()
    if FAILED == 0:
        print("  🎉 ALL TESTS PASSED — Pipeline is ready for GPU training!")
    else:
        print(f"  ⚠️  {FAILED} test(s) failed — see above for details")

    return FAILED == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
