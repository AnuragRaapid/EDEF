# pyright: reportMissingImports=false

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn

from edef_modules import EntityDistProjector, GatedFusion, LateCorrectionTransformer


def _search_module_with_layers(module: nn.Module, depth: int = 4) -> nn.Module | None:
    if hasattr(module, "layers") and isinstance(
        getattr(module, "layers"), nn.ModuleList
    ):
        return module
    if depth <= 0:
        return None
    for child in module.children():
        found = _search_module_with_layers(child, depth=depth - 1)
        if found is not None:
            return found
    return None


def _resolve_backbone(model: nn.Module) -> nn.Module:
    backbone = _search_module_with_layers(model)
    if backbone is None:
        raise AttributeError(
            "Could not locate decoder backbone with a .layers ModuleList"
        )
    return backbone


def _resolve_decoder_layers(model: nn.Module) -> nn.ModuleList:
    backbone = _resolve_backbone(model)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise AttributeError(
            "Model backbone must expose decoder layers as nn.ModuleList"
        )
    return layers


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _align_dist_to_seq(
    dist: torch.Tensor,
    seq_len: int,
    batch_size: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    if dist.dim() != 3:
        raise ValueError(
            "entity_dist_vectors must have shape (batch, seq_len, dist_dim)"
        )
    if dist.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: dist batch={dist.shape[0]} vs hidden batch={batch_size}"
        )

    aligned = dist.to(device=target_device, dtype=target_dtype)
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


def _align_mask_to_seq(
    prompt_mask: torch.Tensor | None,
    seq_len: int,
    batch_size: int,
    target_device: torch.device,
) -> torch.Tensor | None:
    if prompt_mask is None:
        return None
    if prompt_mask.dim() != 2:
        raise ValueError("entity_prompt_mask must have shape (batch, seq_len)")
    if prompt_mask.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: prompt mask batch={prompt_mask.shape[0]} vs hidden batch={batch_size}"
        )

    aligned = prompt_mask.to(device=target_device, dtype=torch.bool)
    if aligned.shape[1] > seq_len:
        return aligned[:, :seq_len]
    if aligned.shape[1] < seq_len:
        padding = torch.zeros(
            aligned.shape[0],
            seq_len - aligned.shape[1],
            device=target_device,
            dtype=torch.bool,
        )
        return torch.cat([aligned, padding], dim=1)
    return aligned


def _extract_hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("Decoder layer output does not expose a tensor hidden state")
    return hidden


def _replace_hidden_in_layer_output(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def _module_dtype_name(module: nn.Module) -> str:
    try:
        return str(next(module.parameters()).dtype).replace("torch.", "")
    except StopIteration:
        return "float32"


def _resolve_edef_dtype(
    model: nn.Module, requested_dtype: torch.dtype | None
) -> torch.dtype:
    if requested_dtype is not None:
        return requested_dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _get_edef_modules(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    projector = getattr(model, "entity_projector", None)
    fusion_gate = getattr(model, "fusion_gate", None)
    if not isinstance(projector, nn.Module) or not isinstance(fusion_gate, nn.Module):
        raise AttributeError(
            "Model must expose entity_projector and fusion_gate modules"
        )
    return projector, fusion_gate


def _get_late_corrector(model: nn.Module) -> nn.Module | None:
    corrector = getattr(model, "late_corrector", None)
    return corrector if isinstance(corrector, nn.Module) else None


def _infer_edef_config_from_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    projector_path = os.path.join(checkpoint_path, "entity_projector.pt")
    if os.path.isfile(projector_path):
        try:
            projector_state = torch.load(projector_path, map_location="cpu")
        except Exception:
            projector_state = None
        if isinstance(projector_state, dict):
            if (
                "projector.4.weight" in projector_state
                and "projector.0.weight" in projector_state
            ):
                inferred["projector_bottleneck_dim"] = int(
                    projector_state["projector.0.weight"].shape[0]
                )
            else:
                inferred["projector_bottleneck_dim"] = None
            inferred["projector_use_temperature"] = "log_temperature" in projector_state
            first_tensor = next(
                (
                    value
                    for value in projector_state.values()
                    if isinstance(value, torch.Tensor)
                ),
                None,
            )
            if isinstance(first_tensor, torch.Tensor):
                inferred["edef_dtype"] = str(first_tensor.dtype).replace("torch.", "")

    fusion_path = os.path.join(checkpoint_path, "fusion_gate.pt")
    if os.path.isfile(fusion_path):
        try:
            fusion_state = torch.load(fusion_path, map_location="cpu")
        except Exception:
            fusion_state = None
        if isinstance(fusion_state, dict):
            inferred["fusion_projected_norm"] = any(
                key.startswith("projected_norm.") for key in fusion_state
            )

    return inferred


def _load_module_state(
    module: nn.Module, state_dict: dict[str, Any], module_name: str
) -> None:
    current_state = module.state_dict()
    compatible_state: dict[str, Any] = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if (
            current_value is None
            or not isinstance(value, torch.Tensor)
            or current_value.shape != value.shape
        ):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    incompatible = module.load_state_dict(compatible_state, strict=False)
    if skipped_keys or incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[WARN] Loaded {module_name} with compatibility fallback: "
            f"missing={len(incompatible.missing_keys)}, "
            f"skipped={len(skipped_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )


def get_edef_host(model: nn.Module) -> nn.Module:
    visited: set[int] = set()

    def _search(module: nn.Module) -> nn.Module | None:
        module_id = id(module)
        if module_id in visited:
            return None
        visited.add(module_id)

        if hasattr(module, "entity_projector") and hasattr(module, "fusion_gate"):
            return module

        for child_name in ("base_model", "model"):
            child = getattr(module, child_name, None)
            if isinstance(child, nn.Module):
                found = _search(child)
                if found is not None:
                    return found

        for child in module.children():
            found = _search(child)
            if found is not None:
                return found
        return None

    host = _search(model)
    if host is None:
        raise AttributeError("Could not locate the model carrying EDEF modules")
    return host


def get_decoder_layer_count(model: nn.Module) -> int:
    host = (
        get_edef_host(model)
        if hasattr(model, "entity_projector") or hasattr(model, "base_model")
        else model
    )
    return len(_resolve_decoder_layers(host))


def load_edef_config(checkpoint_path: str) -> dict[str, Any]:
    config_path = os.path.join(checkpoint_path, "edef_config.json")
    inferred = _infer_edef_config_from_checkpoint(checkpoint_path)
    if not os.path.isfile(config_path):
        return inferred
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {**inferred, **payload}


class _LateEDEFRuntime:
    def __init__(self, host_model: nn.Module) -> None:
        self.host_model = host_model
        self.entity_dist_vectors: torch.Tensor | None = None
        self.entity_prompt_mask: torch.Tensor | None = None
        self.ablate = False
        self.prefill_only = False
        self.prefill_done = False
        self.last_token_gate: torch.Tensor | None = None
        self.last_aligned_dist: torch.Tensor | None = None
        self.last_prompt_mask: torch.Tensor | None = None

    def set_context(
        self,
        entity_dist_vectors: torch.Tensor,
        entity_prompt_mask: torch.Tensor | None = None,
        *,
        ablate: bool = False,
        prefill_only: bool = False,
    ) -> None:
        self.entity_dist_vectors = entity_dist_vectors
        self.entity_prompt_mask = entity_prompt_mask
        self.ablate = ablate
        self.prefill_only = prefill_only
        self.prefill_done = False
        self.last_token_gate = None
        self.last_aligned_dist = None
        self.last_prompt_mask = None

    def clear_context(self) -> None:
        self.entity_dist_vectors = None
        self.entity_prompt_mask = None
        self.ablate = False
        self.prefill_only = False
        self.prefill_done = False

    def _derive_prompt_mask(self, aligned_dist: torch.Tensor) -> torch.Tensor:
        return aligned_dist.abs().sum(dim=-1) > 0

    def _should_apply(self, hidden_states: torch.Tensor) -> bool:
        if self.entity_dist_vectors is None:
            return False
        if not self.prefill_only:
            return True
        if self.prefill_done:
            return False
        return hidden_states.shape[1] > 1

    def _apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.entity_dist_vectors is None:
            return hidden_states

        original_dtype = hidden_states.dtype
        device = hidden_states.device
        batch_size, seq_len = hidden_states.shape[:2]
        projector, fusion_gate = _get_edef_modules(self.host_model)
        compute_dtype = _get_module_dtype(projector)
        aligned_dist = _align_dist_to_seq(
            self.entity_dist_vectors,
            seq_len=seq_len,
            batch_size=batch_size,
            target_device=device,
            target_dtype=compute_dtype,
        )
        prompt_mask = _align_mask_to_seq(
            self.entity_prompt_mask,
            seq_len=seq_len,
            batch_size=batch_size,
            target_device=device,
        )
        if prompt_mask is None:
            prompt_mask = self._derive_prompt_mask(aligned_dist)

        autocast_enabled = (
            device.type == "cuda" and compute_dtype in {torch.float16, torch.bfloat16}
        ) or (device.type == "cpu" and compute_dtype == torch.bfloat16)
        autocast_context = (
            torch.amp.autocast(device_type=device.type, dtype=compute_dtype)
            if autocast_enabled
            else nullcontext()
        )

        with autocast_context:
            hidden_input = hidden_states.to(compute_dtype)
            projected = projector(aligned_dist)

            if self.ablate:
                corrected = hidden_input
                token_gate = hidden_input.new_zeros(hidden_input.shape[:2])
            else:
                fused, gate_values = fusion_gate(
                    hidden_input,
                    projected,
                    prompt_mask=prompt_mask,
                    return_gate=True,
                )
                corrected = fused
                corrector = _get_late_corrector(self.host_model)
                if corrector is not None:
                    corrected = corrector(corrected, prompt_mask=prompt_mask)
                token_gate = gate_values.mean(dim=-1)

        self.last_token_gate = token_gate.detach().cpu()
        self.last_aligned_dist = aligned_dist.detach().cpu()
        self.last_prompt_mask = prompt_mask.detach().cpu()
        return corrected.to(original_dtype)

    def hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        hidden_states = _extract_hidden_from_layer_output(output)
        if not self._should_apply(hidden_states):
            return output
        corrected = self._apply(hidden_states)
        if self.prefill_only:
            self.prefill_done = True
        return _replace_hidden_in_layer_output(output, corrected)


@contextmanager
def edef_runtime_context(
    model: nn.Module,
    entity_dist_vectors: torch.Tensor,
    entity_prompt_mask: torch.Tensor | None = None,
    *,
    ablate: bool = False,
    prefill_only: bool = False,
):
    host = get_edef_host(model)
    runtime = getattr(host, "_edef_runtime", None)
    if runtime is None:
        raise AttributeError("EDEF runtime is not attached to the model")
    runtime.set_context(
        entity_dist_vectors,
        entity_prompt_mask=entity_prompt_mask,
        ablate=ablate,
        prefill_only=prefill_only,
    )
    try:
        yield host
    finally:
        runtime.clear_context()


def get_last_edef_stats(model: nn.Module) -> dict[str, torch.Tensor | None]:
    host = get_edef_host(model)
    runtime = getattr(host, "_edef_runtime", None)
    if runtime is None:
        return {"token_gate": None, "aligned_dist": None, "prompt_mask": None}
    return {
        "token_gate": runtime.last_token_gate,
        "aligned_dist": runtime.last_aligned_dist,
        "prompt_mask": runtime.last_prompt_mask,
    }


def attach_edef_to_model(
    model: nn.Module,
    dist_dim: int = 45,
    hidden_dim: int | None = None,
    insertion_layer: int | None = None,
    projector_bottleneck_dim: int | None = None,
    projector_use_temperature: bool = False,
    fusion_projected_norm: bool = False,
    corrector_layers: int = 0,
    corrector_dim: int = 512,
    corrector_heads: int = 8,
    edef_dtype: torch.dtype | None = None,
) -> nn.Module:
    if hidden_dim is None:
        hidden_dim = int(getattr(getattr(model, "config", None), "hidden_size", 2560))
    if projector_bottleneck_dim is not None and projector_bottleneck_dim <= 0:
        projector_bottleneck_dim = None

    if (
        hasattr(model, "_edef_hook_handle")
        and getattr(model, "_edef_hook_handle") is not None
    ):
        model._edef_hook_handle.remove()
        model._edef_hook_handle = None

    layers = _resolve_decoder_layers(model)
    num_layers = len(layers)
    if insertion_layer is None:
        insertion_layer = max(0, num_layers - 8)
    if insertion_layer < 0 or insertion_layer >= num_layers:
        raise ValueError(
            f"insertion_layer must be in [0, {num_layers - 1}], got {insertion_layer}"
        )

    insertion_device = _get_module_device(layers[insertion_layer])
    resolved_edef_dtype = _resolve_edef_dtype(model, edef_dtype)
    model.entity_projector = EntityDistProjector(
        dist_dim=dist_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=projector_bottleneck_dim,
        use_temperature_scaling=projector_use_temperature,
    ).to(
        device=insertion_device,
        dtype=resolved_edef_dtype,
    )
    model.fusion_gate = GatedFusion(
        hidden_dim=hidden_dim,
        normalize_projected=fusion_projected_norm,
    ).to(
        device=insertion_device,
        dtype=resolved_edef_dtype,
    )
    if corrector_layers > 0:
        model.late_corrector = LateCorrectionTransformer(
            hidden_dim=hidden_dim,
            bottleneck_dim=corrector_dim,
            num_layers=corrector_layers,
            num_heads=corrector_heads,
        ).to(device=insertion_device, dtype=resolved_edef_dtype)
    elif hasattr(model, "late_corrector"):
        delattr(model, "late_corrector")

    model.edef_dist_dim = int(dist_dim)
    model.edef_hidden_dim = int(hidden_dim)
    model.edef_insertion_layer = int(insertion_layer)
    model.edef_projector_bottleneck_dim = (
        int(projector_bottleneck_dim) if projector_bottleneck_dim is not None else None
    )
    model.edef_projector_use_temperature = bool(projector_use_temperature)
    model.edef_fusion_projected_norm = bool(fusion_projected_norm)
    model.edef_corrector_layers = int(corrector_layers)
    model.edef_corrector_dim = int(corrector_dim)
    model.edef_corrector_heads = int(corrector_heads)
    model.edef_dtype = str(resolved_edef_dtype).replace("torch.", "")

    runtime = _LateEDEFRuntime(model)
    handle = layers[insertion_layer].register_forward_hook(runtime.hook)
    model._edef_runtime = runtime
    model._edef_hook_handle = handle

    if not hasattr(model, "_edef_original_forward"):
        model._edef_original_forward = model.forward
    original_forward = model._edef_original_forward

    def edef_forward(
        input_ids: torch.Tensor | None = None,
        entity_dist_vectors: torch.Tensor | None = None,
        entity_prompt_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        late_edef_ablate: bool = False,
        **kwargs: Any,
    ) -> Any:
        if entity_dist_vectors is None:
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )
        with edef_runtime_context(
            model,
            entity_dist_vectors=entity_dist_vectors,
            entity_prompt_mask=entity_prompt_mask,
            ablate=late_edef_ablate,
            prefill_only=False,
        ):
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )

    model.forward = edef_forward
    return model


def load_edef_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    host = get_edef_host(model)
    projector_module, fusion_module = _get_edef_modules(host)
    corrector_module = _get_late_corrector(host)

    projector_state = torch.load(
        os.path.join(checkpoint_path, "entity_projector.pt"),
        map_location="cpu",
    )
    fusion_state = torch.load(
        os.path.join(checkpoint_path, "fusion_gate.pt"),
        map_location="cpu",
    )
    _load_module_state(projector_module, projector_state, "entity_projector")
    _load_module_state(fusion_module, fusion_state, "fusion_gate")

    corrector_path = os.path.join(checkpoint_path, "late_corrector.pt")
    if corrector_module is not None and os.path.isfile(corrector_path):
        corrector_state = torch.load(corrector_path, map_location="cpu")
        _load_module_state(corrector_module, corrector_state, "late_corrector")

    target_device = _get_module_device(
        _resolve_decoder_layers(host)[host.edef_insertion_layer]
    )
    projector_module.to(device=target_device)
    fusion_module.to(device=target_device)
    if corrector_module is not None:
        corrector_module.to(device=target_device)
    return model


def save_edef_checkpoint(model: nn.Module, save_path: str) -> None:
    host = get_edef_host(model)
    projector_module, fusion_module = _get_edef_modules(host)
    corrector_module = _get_late_corrector(host)

    os.makedirs(save_path, exist_ok=True)
    torch.save(
        projector_module.state_dict(), os.path.join(save_path, "entity_projector.pt")
    )
    torch.save(fusion_module.state_dict(), os.path.join(save_path, "fusion_gate.pt"))
    if corrector_module is not None:
        torch.save(
            corrector_module.state_dict(), os.path.join(save_path, "late_corrector.pt")
        )

    payload = {
        "dist_dim": int(getattr(host, "edef_dist_dim", 45)),
        "hidden_dim": int(getattr(host, "edef_hidden_dim", 2560)),
        "insertion_layer": int(getattr(host, "edef_insertion_layer", 0)),
        "projector_bottleneck_dim": getattr(
            host, "edef_projector_bottleneck_dim", None
        ),
        "projector_use_temperature": bool(
            getattr(host, "edef_projector_use_temperature", False)
        ),
        "fusion_projected_norm": bool(
            getattr(host, "edef_fusion_projected_norm", False)
        ),
        "corrector_layers": int(getattr(host, "edef_corrector_layers", 0)),
        "corrector_dim": int(getattr(host, "edef_corrector_dim", 512)),
        "corrector_heads": int(getattr(host, "edef_corrector_heads", 8)),
        "edef_dtype": str(
            getattr(host, "edef_dtype", _module_dtype_name(projector_module))
        ),
    }
    with open(os.path.join(save_path, "edef_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = Qwen3ForCausalLM(config)
    model = attach_edef_to_model(
        model,
        dist_dim=45,
        hidden_dim=64,
        insertion_layer=1,
        corrector_layers=2,
        corrector_dim=32,
        corrector_heads=4,
    )

    input_ids = torch.randint(0, 128, (2, 10), dtype=torch.long)
    entity_dist_vectors = torch.randn(2, 7, 45)
    entity_prompt_mask = torch.zeros(2, 10, dtype=torch.bool)
    entity_prompt_mask[:, :7] = True

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        entity_dist_vectors=entity_dist_vectors,
        entity_prompt_mask=entity_prompt_mask,
        labels=input_ids,
    )
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (2, 10, 128)

    stats = get_last_edef_stats(model)
    assert stats["token_gate"] is not None
    assert stats["token_gate"].shape == (2, 10)

    with tempfile.TemporaryDirectory(prefix="late_edef_ckpt_") as ckpt_dir:
        save_edef_checkpoint(model, ckpt_dir)
        projector_before = {
            key: value.detach().clone()
            for key, value in model.entity_projector.state_dict().items()
        }
        with torch.no_grad():
            first_param = next(model.entity_projector.parameters())
            first_param.add_(1.0)
        load_edef_checkpoint(model, ckpt_dir)
        for key, value in model.entity_projector.state_dict().items():
            assert torch.allclose(projector_before[key], value)

    print(
        "Late EDEF smoke test passed: forward path, stats capture, and checkpoint round-trip."
    )
