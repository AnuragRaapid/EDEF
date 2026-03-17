# Encoder-Style BIO NER

This package adds a token-classification path on top of the trained EDEF backbone.

Why this exists:
- The generative JSON setup is good at entity typing, but it can miss exact boundaries.
- A BIO tagger optimizes token boundaries directly.
- The Qwen backbone is still decoder-only, so this implementation adds a bidirectional tagging head on top of the final hidden states to recover right-context for tagging.
- The default head is now `BiLSTM + CRF`, which is typically stronger than plain token-wise softmax for BIO boundary consistency.

## Train

```bash
uv run python3 train_encoder_ner.py \
  --phase1_model qwen3-phase1-checkpoint \
  --stage2_adapter /path/to/your/stage2-output \
  --stage2_edef_checkpoint /path/to/your/stage2-output/edef_checkpoint \
  --head_type bilstm_crf \
  --train_data anurag-raapid/lct-corpus \
  --val_data anurag-raapid/lct-corpus \
  --train_split train \
  --val_split validation \
  --dist_path artifacts/lct-corpus/entity_distributions.json \
  --output_dir saves/encoder-ner
```

## Evaluate

```bash
uv run python3 evaluate_encoder_ner.py \
  --model_path saves/encoder-ner \
  --phase1_model qwen3-phase1-checkpoint \
  --stage2_adapter /path/to/your/stage2-output \
  --stage2_edef_checkpoint /path/to/your/stage2-output/edef_checkpoint \
  --test_data anurag-raapid/lct-corpus \
  --test_split test \
  --dist_path artifacts/lct-corpus/entity_distributions.json
```

## Notes

- `--head_type bilstm_crf` is the default and is recommended for boundary repair.
- The CRF uses BIO transition constraints, so it discourages invalid sequences such as starting with `I-*`.
- If you want a pure text tagger without EDEF fusion, add `--disable_edef`.
- The saved adapter depends on the same Phase 1 + Stage 2 backbone recreation path during evaluation/inference.
