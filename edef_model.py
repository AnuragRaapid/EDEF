# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import torch
import torch.nn as nn

from edef_modules import EntityDistProjector, GatedFusion

DEFAULT_SIGNAL_SOURCE = "distribution"
DEFAULT_MEDICAL_ENCODER_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_MEDICAL_CHUNK_SIZE = 448
DEFAULT_MEDICAL_CHUNK_OVERLAP = 128
DEFAULT_MAX_PROMPT_MEDICAL_TOKENS = 4
DEFAULT_MEDICAL_LORA_TARGET_MODULES = ("query", "value")
DEFAULT_MEDICAL_LORA_TOP_LAYERS = 4


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_embed_tokens_module(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "model", None)
    embed_tokens = getattr(backbone, "embed_tokens", None)
    if not isinstance(embed_tokens, nn.Module):
        raise AttributeError(
            "Model must expose embedding layer at model.model.embed_tokens"
        )
    return embed_tokens


def get_edef_host(model: nn.Module | Any) -> nn.Module | None:
    if hasattr(model, "entity_projector") and hasattr(model, "fusion_gate"):
        return model
    for attr in ("base_model", "model"):
        child = getattr(model, attr, None)
        if isinstance(child, nn.Module):
            found = get_edef_host(child)
            if found is not None:
                return found
    return None


def _get_edef_modules(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    projector = getattr(model, "entity_projector", None)
    fusion_gate = getattr(model, "fusion_gate", None)
    if not isinstance(projector, nn.Module) or not isinstance(fusion_gate, nn.Module):
        raise AttributeError("Model must have entity_projector and fusion_gate modules")
    return projector, fusion_gate


def _get_medical_encoder_module(model: nn.Module) -> nn.Module:
    medical_encoder = getattr(model, "medical_encoder", None)
    if not isinstance(medical_encoder, nn.Module):
        raise AttributeError(
            "Model must have a medical_encoder module for semantic fusion"
        )
    return medical_encoder


def _projector_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    projector, _ = _get_edef_modules(model)
    param = next(projector.parameters())
    return param.device, param.dtype


def get_edef_config(model: nn.Module | Any) -> dict[str, Any]:
    host = get_edef_host(model)
    if host is None:
        return {}
    raw = getattr(host, "_edef_config", None)
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def load_edef_config(checkpoint_path: str) -> dict[str, Any]:
    config_path = os.path.join(checkpoint_path, "edef_config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _align_signal_to_seq(
    signal: torch.Tensor,
    seq_len: int,
    batch_size: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
    signal_name: str,
) -> torch.Tensor:
    if signal.dim() != 3:
        raise ValueError(f"{signal_name} must have shape (batch, seq_len, signal_dim)")
    if signal.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: {signal_name} batch={signal.shape[0]} vs embeddings batch={batch_size}"
        )

    aligned = signal.to(device=target_device, dtype=target_dtype)
    if aligned.shape[1] > seq_len:
        return aligned[:, :seq_len, :]
    if aligned.shape[1] < seq_len:
        padding = torch.zeros(
            aligned.shape[0],
            seq_len - aligned.shape[1],
            aligned.shape[2],
            device=target_device,
            dtype=target_dtype,
        )
        return torch.cat([aligned, padding], dim=1)
    return aligned


def _infer_medical_signal_dim(medical_encoder: nn.Module) -> int:
    candidates: list[Any] = [medical_encoder]
    base_model = getattr(medical_encoder, "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None:
            candidates.append(inner_model)
    if hasattr(medical_encoder, "get_base_model"):
        try:
            candidates.append(medical_encoder.get_base_model())
        except Exception:
            pass

    for candidate in candidates:
        config = getattr(candidate, "config", None)
        for attr in ("hidden_size", "dim", "d_model"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    raise AttributeError("Could not infer medical encoder hidden size from config")


def _load_medical_encoder(
    model_name: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> nn.Module:
    from transformers import AutoModel

    medical_encoder = AutoModel.from_pretrained(
        model_name,
        torch_dtype=target_dtype,
        trust_remote_code=True,
    )
    medical_encoder.to(device=target_device, dtype=target_dtype)
    return medical_encoder


def apply_medical_encoder_lora(
    model: nn.Module | Any,
    *,
    r: int = 8,
    alpha: int = 16,
    target_modules: list[str] | tuple[str, ...] | None = None,
    top_layers: int | None = DEFAULT_MEDICAL_LORA_TOP_LAYERS,
    dropout: float = 0.0,
    use_dora: bool = False,
) -> nn.Module | Any:
    host = get_edef_host(model)
    if host is None:
        raise ValueError("Could not locate attached EDEF modules on model.")

    config = get_edef_config(host)
    if config.get("signal_source") != "medical_encoder":
        return model

    medical_encoder = _get_medical_encoder_module(host)
    if hasattr(medical_encoder, "peft_config"):
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    resolved_targets = list(target_modules or DEFAULT_MEDICAL_LORA_TARGET_MODULES)
    layers_to_transform: list[int] | None = None
    num_layers = getattr(
        getattr(medical_encoder, "config", None), "num_hidden_layers", None
    )
    if isinstance(num_layers, int) and num_layers > 0 and top_layers is not None:
        if 0 < int(top_layers) < num_layers:
            start = max(0, num_layers - int(top_layers))
            layers_to_transform = list(range(start, num_layers))

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=resolved_targets,
        lora_dropout=dropout,
        bias="none",
        use_dora=use_dora,
        task_type=TaskType.FEATURE_EXTRACTION,
        layers_to_transform=layers_to_transform,
    )
    host.medical_encoder = get_peft_model(medical_encoder, lora_config)
    host._edef_config = {
        **config,
        "medical_lora_r": int(r),
        "medical_lora_alpha": int(alpha),
        "medical_lora_target_modules": resolved_targets,
        "medical_lora_top_layers": None if top_layers is None else int(top_layers),
        "medical_lora_dropout": float(dropout),
        "medical_lora_use_dora": bool(use_dora),
    }
    return model


def _build_aligned_medical_features(
    model: nn.Module,
    *,
    prompt_medical_token_indices: torch.Tensor,
    prompt_medical_token_weights: torch.Tensor,
    medical_chunk_input_ids: torch.Tensor,
    medical_chunk_attention_mask: torch.Tensor,
    medical_chunk_token_indices: torch.Tensor,
    medical_chunk_token_weights: torch.Tensor,
    medical_token_count: torch.Tensor | None,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    if prompt_medical_token_indices.dim() != 3:
        raise ValueError(
            "prompt_medical_token_indices must have shape (batch, seq_len, max_overlap)"
        )
    if prompt_medical_token_weights.shape != prompt_medical_token_indices.shape:
        raise ValueError(
            "prompt_medical_token_weights shape must match prompt_medical_token_indices"
        )
    if medical_chunk_input_ids.dim() != 3:
        raise ValueError(
            "medical_chunk_input_ids must have shape (batch, num_chunks, chunk_seq)"
        )
    if medical_chunk_attention_mask.shape != medical_chunk_input_ids.shape:
        raise ValueError(
            "medical_chunk_attention_mask shape must match medical_chunk_input_ids"
        )
    if medical_chunk_token_indices.shape != medical_chunk_input_ids.shape:
        raise ValueError(
            "medical_chunk_token_indices shape must match medical_chunk_input_ids"
        )
    if medical_chunk_token_weights.shape != medical_chunk_input_ids.shape:
        raise ValueError(
            "medical_chunk_token_weights shape must match medical_chunk_input_ids"
        )

    batch_size, num_chunks, chunk_seq_len = medical_chunk_input_ids.shape
    seq_len = prompt_medical_token_indices.shape[1]

    medical_encoder = _get_medical_encoder_module(model)
    encoder_device = _get_model_device(medical_encoder)
    flat_input_ids = medical_chunk_input_ids.reshape(-1, chunk_seq_len).to(
        encoder_device
    )
    flat_attention_mask = medical_chunk_attention_mask.reshape(-1, chunk_seq_len).to(
        encoder_device
    )
    flat_token_indices = medical_chunk_token_indices.reshape(-1, chunk_seq_len)
    flat_token_weights = medical_chunk_token_weights.reshape(-1, chunk_seq_len)

    valid_mask = flat_attention_mask.sum(dim=-1) > 0
    hidden_size = _infer_medical_signal_dim(medical_encoder)
    if not bool(valid_mask.any()):
        return torch.zeros(
            batch_size,
            seq_len,
            hidden_size,
            device=target_device,
            dtype=target_dtype,
        )

    flat_batch_indices = (
        torch.arange(batch_size, device=encoder_device)
        .unsqueeze(1)
        .expand(batch_size, num_chunks)
        .reshape(-1)
    )
    valid_input_ids = flat_input_ids[valid_mask]
    valid_attention_mask = flat_attention_mask[valid_mask]
    valid_batch_indices = flat_batch_indices[valid_mask]
    valid_token_indices = flat_token_indices[valid_mask]
    valid_token_weights = flat_token_weights[valid_mask]

    outputs = medical_encoder(
        input_ids=valid_input_ids,
        attention_mask=valid_attention_mask,
        return_dict=True,
    )
    valid_hidden = outputs.last_hidden_state
    hidden_size = int(valid_hidden.shape[-1])

    max_med_tokens = 0
    if medical_token_count is not None and medical_token_count.numel() > 0:
        max_med_tokens = max(max_med_tokens, int(medical_token_count.max().item()))
    if bool((valid_token_indices >= 0).any()):
        max_med_tokens = max(
            max_med_tokens,
            int(valid_token_indices[valid_token_indices >= 0].max().item()) + 1,
        )
    if bool((prompt_medical_token_indices >= 0).any()):
        max_med_tokens = max(
            max_med_tokens,
            int(
                prompt_medical_token_indices[prompt_medical_token_indices >= 0]
                .max()
                .item()
            )
            + 1,
        )

    if max_med_tokens <= 0:
        return torch.zeros(
            batch_size,
            seq_len,
            hidden_size,
            device=target_device,
            dtype=target_dtype,
        )

    merged_sum = torch.zeros(
        batch_size,
        max_med_tokens,
        hidden_size,
        device=encoder_device,
        dtype=valid_hidden.dtype,
    )
    merged_weight = torch.zeros(
        batch_size,
        max_med_tokens,
        1,
        device=encoder_device,
        dtype=valid_hidden.dtype,
    )

    for row_idx in range(valid_hidden.shape[0]):
        batch_idx = int(valid_batch_indices[row_idx].item())
        token_idx_row = valid_token_indices[row_idx].to(device=encoder_device)
        token_weight_row = valid_token_weights[row_idx].to(
            device=encoder_device,
            dtype=valid_hidden.dtype,
        )
        keep_mask = token_idx_row >= 0
        if not bool(keep_mask.any()):
            continue
        token_idx = token_idx_row[keep_mask].long()
        weights = token_weight_row[keep_mask].unsqueeze(-1)
        hidden_states = valid_hidden[row_idx, keep_mask, :]
        merged_sum[batch_idx].index_add_(0, token_idx, hidden_states * weights)
        merged_weight[batch_idx].index_add_(0, token_idx, weights)

    merged_features = merged_sum / merged_weight.clamp_min(1e-6)
    prompt_indices = prompt_medical_token_indices.to(device=encoder_device)
    prompt_weights = prompt_medical_token_weights.to(
        device=encoder_device,
        dtype=merged_features.dtype,
    )
    safe_indices = prompt_indices.clamp_min(0)
    batch_lookup = torch.arange(batch_size, device=encoder_device)[:, None, None]
    gathered = merged_features[batch_lookup, safe_indices]
    valid_prompt_mask = (prompt_indices >= 0).unsqueeze(-1)
    weighted = gathered * prompt_weights.unsqueeze(-1) * valid_prompt_mask
    denom = (prompt_weights.unsqueeze(-1) * valid_prompt_mask).sum(dim=2)
    aligned = weighted.sum(dim=2)
    aligned = torch.where(
        denom > 0,
        aligned / denom.clamp_min(1e-6),
        torch.zeros_like(aligned),
    )
    return aligned.to(device=target_device, dtype=target_dtype)


def prepare_projected_features(
    model: nn.Module | Any,
    *,
    input_ids: torch.Tensor,
    entity_dist_vectors: torch.Tensor | None = None,
    medical_chunk_input_ids: torch.Tensor | None = None,
    medical_chunk_attention_mask: torch.Tensor | None = None,
    medical_chunk_token_indices: torch.Tensor | None = None,
    medical_chunk_token_weights: torch.Tensor | None = None,
    prompt_medical_token_indices: torch.Tensor | None = None,
    prompt_medical_token_weights: torch.Tensor | None = None,
    medical_token_count: torch.Tensor | None = None,
) -> torch.Tensor | None:
    host = get_edef_host(model)
    if host is None:
        raise ValueError("Could not locate attached EDEF modules on model.")
    projector, _ = _get_edef_modules(host)
    target_device, target_dtype = _projector_device_dtype(host)
    config = get_edef_config(host)
    signal_source = str(config.get("signal_source", DEFAULT_SIGNAL_SOURCE))
    batch_size = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])

    if signal_source == "distribution":
        if entity_dist_vectors is None:
            return None
        aligned_signal = _align_signal_to_seq(
            entity_dist_vectors,
            seq_len=seq_len,
            batch_size=batch_size,
            target_device=target_device,
            target_dtype=target_dtype,
            signal_name="entity_dist_vectors",
        )
        return projector(aligned_signal)

    if signal_source == "medical_encoder":
        required_medical = [
            medical_chunk_input_ids,
            medical_chunk_attention_mask,
            medical_chunk_token_indices,
            medical_chunk_token_weights,
            prompt_medical_token_indices,
            prompt_medical_token_weights,
        ]
        if any(item is None for item in required_medical):
            return None
        aligned_signal = _build_aligned_medical_features(
            host,
            prompt_medical_token_indices=prompt_medical_token_indices,
            prompt_medical_token_weights=prompt_medical_token_weights,
            medical_chunk_input_ids=medical_chunk_input_ids,
            medical_chunk_attention_mask=medical_chunk_attention_mask,
            medical_chunk_token_indices=medical_chunk_token_indices,
            medical_chunk_token_weights=medical_chunk_token_weights,
            medical_token_count=medical_token_count,
            target_device=target_device,
            target_dtype=target_dtype,
        )
        return projector(aligned_signal)

    raise ValueError(f"Unsupported signal_source={signal_source}")


def build_fused_input_embeddings(
    model: nn.Module | Any,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
    entity_dist_vectors: torch.Tensor | None = None,
    medical_chunk_input_ids: torch.Tensor | None = None,
    medical_chunk_attention_mask: torch.Tensor | None = None,
    medical_chunk_token_indices: torch.Tensor | None = None,
    medical_chunk_token_weights: torch.Tensor | None = None,
    prompt_medical_token_indices: torch.Tensor | None = None,
    prompt_medical_token_weights: torch.Tensor | None = None,
    medical_token_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    host = get_edef_host(model)
    if host is None:
        raise ValueError("Could not locate attached EDEF modules on model.")

    token_embed_layer = _get_embed_tokens_module(host)
    token_embeddings = (
        inputs_embeds if inputs_embeds is not None else token_embed_layer(input_ids)
    )
    projected = prepare_projected_features(
        host,
        input_ids=input_ids,
        entity_dist_vectors=entity_dist_vectors,
        medical_chunk_input_ids=medical_chunk_input_ids,
        medical_chunk_attention_mask=medical_chunk_attention_mask,
        medical_chunk_token_indices=medical_chunk_token_indices,
        medical_chunk_token_weights=medical_chunk_token_weights,
        prompt_medical_token_indices=prompt_medical_token_indices,
        prompt_medical_token_weights=prompt_medical_token_weights,
        medical_token_count=medical_token_count,
    )
    if projected is None:
        return token_embeddings, None

    _, fusion_gate = _get_edef_modules(host)
    projected = _align_signal_to_seq(
        projected,
        seq_len=token_embeddings.shape[1],
        batch_size=token_embeddings.shape[0],
        target_device=token_embeddings.device,
        target_dtype=token_embeddings.dtype,
        signal_name="projected_signal",
    )
    fused = fusion_gate(token_embeddings, projected)
    return fused, projected


def attach_edef_to_model(
    model: nn.Module,
    dist_dim: int = 45,
    hidden_dim: int = 2560,
    *,
    signal_source: str = DEFAULT_SIGNAL_SOURCE,
    medical_encoder_model_name: str = DEFAULT_MEDICAL_ENCODER_MODEL,
    medical_chunk_size: int = DEFAULT_MEDICAL_CHUNK_SIZE,
    medical_chunk_overlap: int = DEFAULT_MEDICAL_CHUNK_OVERLAP,
    max_prompt_medical_tokens: int = DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
    medical_encoder_model: nn.Module | None = None,
) -> nn.Module:
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    signal_source = str(signal_source).strip().lower()
    if signal_source not in {"distribution", "medical_encoder"}:
        raise ValueError(f"Unsupported signal_source={signal_source}")

    signal_dim = int(dist_dim)
    edef_config: dict[str, Any] = {
        "signal_source": signal_source,
        "hidden_dim": int(hidden_dim),
        "dist_dim": int(dist_dim),
    }

    if signal_source == "medical_encoder":
        if medical_encoder_model is None:
            medical_encoder_model = _load_medical_encoder(
                medical_encoder_model_name,
                target_device=model_device,
                target_dtype=model_dtype,
            )
        else:
            medical_encoder_model = medical_encoder_model.to(
                device=model_device,
                dtype=model_dtype,
            )
        signal_dim = _infer_medical_signal_dim(medical_encoder_model)
        model.medical_encoder = medical_encoder_model
        edef_config.update(
            {
                "medical_encoder_model_name": medical_encoder_model_name,
                "medical_signal_dim": int(signal_dim),
                "medical_chunk_size": int(medical_chunk_size),
                "medical_chunk_overlap": int(medical_chunk_overlap),
                "max_prompt_medical_tokens": int(max_prompt_medical_tokens),
            }
        )

    model.entity_projector = EntityDistProjector(signal_dim, hidden_dim).to(
        dtype=model_dtype,
        device=model_device,
    )
    model.fusion_gate = GatedFusion(hidden_dim).to(
        dtype=model_dtype, device=model_device
    )
    model._edef_config = edef_config

    if not hasattr(model, "_edef_original_forward"):
        model._edef_original_forward = model.forward
    original_forward = model._edef_original_forward

    def edef_forward(
        input_ids: torch.Tensor | None = None,
        entity_dist_vectors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        medical_chunk_input_ids: torch.Tensor | None = None,
        medical_chunk_attention_mask: torch.Tensor | None = None,
        medical_chunk_token_indices: torch.Tensor | None = None,
        medical_chunk_token_weights: torch.Tensor | None = None,
        prompt_medical_token_indices: torch.Tensor | None = None,
        prompt_medical_token_weights: torch.Tensor | None = None,
        medical_token_count: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if input_ids is None and inputs_embeds is None:
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )

        if input_ids is None:
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )

        fused_embeds, projected = build_fused_input_embeddings(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            entity_dist_vectors=entity_dist_vectors,
            medical_chunk_input_ids=medical_chunk_input_ids,
            medical_chunk_attention_mask=medical_chunk_attention_mask,
            medical_chunk_token_indices=medical_chunk_token_indices,
            medical_chunk_token_weights=medical_chunk_token_weights,
            prompt_medical_token_indices=prompt_medical_token_indices,
            prompt_medical_token_weights=prompt_medical_token_weights,
            medical_token_count=medical_token_count,
        )
        if projected is None:
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )
        return original_forward(inputs_embeds=fused_embeds, **kwargs)

    model.forward = edef_forward
    return model


class EDEFWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dist_dim: int = 45,
        hidden_dim: int = 2560,
        *,
        signal_source: str = DEFAULT_SIGNAL_SOURCE,
    ) -> None:
        super().__init__()
        self.model = attach_edef_to_model(
            model,
            dist_dim=dist_dim,
            hidden_dim=hidden_dim,
            signal_source=signal_source,
        )
        self.entity_projector = self.model.entity_projector
        self.fusion_gate = self.model.fusion_gate

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = list(self.entity_projector.parameters()) + list(
            self.fusion_gate.parameters()
        )
        medical_encoder = getattr(self.model, "medical_encoder", None)
        if hasattr(medical_encoder, "parameters"):
            params.extend(
                param for param in medical_encoder.parameters() if param.requires_grad
            )
        return params

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)


def load_edef_checkpoint(
    model: nn.Module | Any,
    checkpoint_path: str,
    *,
    medical_encoder_trainable: bool = False,
) -> nn.Module | Any:
    host = get_edef_host(model)
    if host is None:
        raise ValueError("Could not locate attached EDEF modules on model.")

    projector_path = os.path.join(checkpoint_path, "entity_projector.pt")
    fusion_path = os.path.join(checkpoint_path, "fusion_gate.pt")
    projector_module, fusion_module = _get_edef_modules(host)

    if os.path.exists(projector_path):
        projector_state = torch.load(projector_path, map_location="cpu")
        projector_module.load_state_dict(projector_state)
    if os.path.exists(fusion_path):
        fusion_state = torch.load(fusion_path, map_location="cpu")
        fusion_module.load_state_dict(fusion_state)

    config = get_edef_config(host)
    if str(config.get("signal_source", DEFAULT_SIGNAL_SOURCE)) == "medical_encoder":
        adapter_dir = os.path.join(checkpoint_path, "medical_encoder_adapter")
        if os.path.isdir(adapter_dir):
            from peft import PeftModel

            medical_encoder = _get_medical_encoder_module(host)
            base_encoder = (
                medical_encoder.get_base_model()
                if hasattr(medical_encoder, "get_base_model")
                else medical_encoder
            )
            host.medical_encoder = PeftModel.from_pretrained(
                base_encoder,
                adapter_dir,
                is_trainable=medical_encoder_trainable,
            )

    device = _get_model_device(host)
    projector_module.to(device)
    fusion_module.to(device)
    medical_encoder = getattr(host, "medical_encoder", None)
    if isinstance(medical_encoder, nn.Module):
        medical_encoder.to(device)
    return model


def save_edef_checkpoint(model: nn.Module | Any, save_path: str) -> None:
    host = get_edef_host(model)
    if host is None:
        raise ValueError("Could not locate attached EDEF modules on model.")

    projector_module, fusion_module = _get_edef_modules(host)
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        projector_module.state_dict(), os.path.join(save_path, "entity_projector.pt")
    )
    torch.save(fusion_module.state_dict(), os.path.join(save_path, "fusion_gate.pt"))

    config_path = os.path.join(save_path, "edef_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(get_edef_config(host), f, indent=2)

    if (
        str(get_edef_config(host).get("signal_source", DEFAULT_SIGNAL_SOURCE))
        == "medical_encoder"
    ):
        medical_encoder = getattr(host, "medical_encoder", None)
        if hasattr(medical_encoder, "peft_config") and hasattr(
            medical_encoder, "save_pretrained"
        ):
            adapter_dir = os.path.join(save_path, "medical_encoder_adapter")
            medical_encoder.save_pretrained(adapter_dir)


if __name__ == "__main__":

    class MockBackbone(nn.Module):
        def __init__(self, vocab_size: int = 1000, hidden_dim: int = 2560) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

    class MockCausalLM(nn.Module):
        def __init__(self, vocab_size: int = 1000, hidden_dim: int = 2560) -> None:
            super().__init__()
            self.model = MockBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            del kwargs
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided."
                    )
                inputs_embeds = self.model.embed_tokens(input_ids)
            logits = self.lm_head(inputs_embeds)
            return {"logits": logits}

    class MockMedicalEncoder(nn.Module):
        def __init__(self, vocab_size: int = 2048, hidden_dim: int = 64) -> None:
            super().__init__()
            self.config = type(
                "MockConfig", (), {"hidden_size": hidden_dim, "num_hidden_layers": 6}
            )()
            self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        def forward(
            self, input_ids=None, attention_mask=None, return_dict=True, **kwargs
        ):
            del attention_mask, kwargs
            hidden = self.embeddings(input_ids)
            if return_dict:
                return type("MockOutput", (), {"last_hidden_state": hidden})()
            return (hidden,)

    torch.manual_seed(7)
    batch_size = 2
    seq_len = 16
    dist_dim = 45
    hidden_dim = 2560
    vocab_size = 1000

    # Distribution path smoke test
    dist_model = MockCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    attach_edef_to_model(dist_model, dist_dim=dist_dim, hidden_dim=hidden_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    entity_dist_vectors = torch.randn(batch_size, seq_len - 3, dist_dim)
    outputs = dist_model(input_ids=input_ids, entity_dist_vectors=entity_dist_vectors)
    assert outputs["logits"].shape == (batch_size, seq_len, vocab_size)

    # Medical path smoke test
    med_model = MockCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    med_encoder = MockMedicalEncoder(hidden_dim=64)
    attach_edef_to_model(
        med_model,
        hidden_dim=hidden_dim,
        signal_source="medical_encoder",
        medical_encoder_model_name="mock-medical",
        medical_encoder_model=med_encoder,
    )
    medical_chunk_input_ids = torch.tensor(
        [
            [[101, 11, 12, 13, 102], [101, 13, 14, 15, 102]],
            [[101, 21, 22, 0, 102], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    medical_chunk_attention_mask = torch.tensor(
        [
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
            [[1, 1, 1, 0, 1], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.long,
    )
    medical_chunk_token_indices = torch.tensor(
        [
            [[-1, 0, 1, 2, -1], [-1, 2, 3, 4, -1]],
            [[-1, 0, 1, -1, -1], [-1, -1, -1, -1, -1]],
        ],
        dtype=torch.long,
    )
    medical_chunk_token_weights = torch.tensor(
        [
            [[0.0, 1.0, 1.0, 0.5, 0.0], [0.0, 0.5, 1.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    prompt_medical_token_indices = torch.tensor(
        [
            [
                [-1, -1],
                [0, 1],
                [2, -1],
                [3, 4],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
            [
                [-1, -1],
                [0, 1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
            ],
        ],
        dtype=torch.long,
    )
    prompt_medical_token_weights = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.6, 0.4],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.7, 0.3],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    medical_token_count = torch.tensor([5, 2], dtype=torch.long)
    med_outputs = med_model(
        input_ids=input_ids,
        medical_chunk_input_ids=medical_chunk_input_ids,
        medical_chunk_attention_mask=medical_chunk_attention_mask,
        medical_chunk_token_indices=medical_chunk_token_indices,
        medical_chunk_token_weights=medical_chunk_token_weights,
        prompt_medical_token_indices=prompt_medical_token_indices,
        prompt_medical_token_weights=prompt_medical_token_weights,
        medical_token_count=medical_token_count,
    )
    assert med_outputs["logits"].shape == (batch_size, seq_len, vocab_size)

    with tempfile.TemporaryDirectory(prefix="edef_ckpt_") as ckpt_dir:
        save_edef_checkpoint(dist_model, ckpt_dir)
        projector_before_state = {
            key: value.detach().clone()
            for key, value in dist_model.entity_projector.state_dict().items()
        }
        with torch.no_grad():
            first_param = next(dist_model.entity_projector.parameters())
            first_param.add_(1.0)
        load_edef_checkpoint(dist_model, ckpt_dir)
        for key, value in dist_model.entity_projector.state_dict().items():
            assert torch.allclose(projector_before_state[key], value)

    print(
        "Smoke test passed: distribution path, medical path, and checkpoint save/load all succeeded."
    )
