# pyright: reportMissingImports=false

import os
import tempfile
from typing import Any

import torch
import torch.nn as nn

from edef_modules import EntityDistProjector, GatedFusion


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


def _get_edef_modules(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    projector = getattr(model, "entity_projector", None)
    fusion_gate = getattr(model, "fusion_gate", None)
    if not isinstance(projector, nn.Module) or not isinstance(fusion_gate, nn.Module):
        raise AttributeError("Model must have entity_projector and fusion_gate modules")
    return projector, fusion_gate


def resolve_trainable_module(module: nn.Module) -> nn.Module:
    modules_to_save = getattr(module, "modules_to_save", None)
    if isinstance(modules_to_save, nn.ModuleDict) and len(modules_to_save) > 0:
        active_adapter = getattr(module, "active_adapter", None)
        if isinstance(active_adapter, str) and active_adapter in modules_to_save:
            return modules_to_save[active_adapter]
        if isinstance(active_adapter, (list, tuple)):
            for adapter_name in active_adapter:
                if isinstance(adapter_name, str) and adapter_name in modules_to_save:
                    return modules_to_save[adapter_name]
        return next(iter(modules_to_save.values()))
    return module


def _normalize_saved_module_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    module_prefixes: list[str] = []
    for key in state_dict:
        if key.startswith("modules_to_save."):
            parts = key.split(".", 2)
            if len(parts) == 3:
                module_prefixes.append(f"modules_to_save.{parts[1]}.")

    if module_prefixes:
        preferred_prefix = (
            "modules_to_save.default."
            if "modules_to_save.default." in module_prefixes
            else module_prefixes[0]
        )
        normalized = {
            key[len(preferred_prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(preferred_prefix)
        }
        if normalized:
            return normalized

    if any(key.startswith("original_module.") for key in state_dict):
        normalized = {
            key[len("original_module.") :]: value
            for key, value in state_dict.items()
            if key.startswith("original_module.")
        }
        if normalized:
            return normalized

    return state_dict


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
            f"Batch size mismatch: dist batch={dist.shape[0]} vs embeddings batch={batch_size}"
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


def attach_edef_to_model(
    model,
    dist_dim=45,
    hidden_dim=2560,
    trainable_dtype: torch.dtype | None = None,
):
    from edef_modules import EntityDistProjector, GatedFusion

    # Detect model dtype/device to match EDEF modules
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    edef_dtype = model_dtype if trainable_dtype is None else trainable_dtype

    model.entity_projector = EntityDistProjector(dist_dim, hidden_dim).to(
        dtype=edef_dtype, device=model_device
    )
    model.fusion_gate = GatedFusion(hidden_dim).to(
        dtype=edef_dtype, device=model_device
    )

    original_forward = model.forward

    def edef_forward(
        input_ids=None, entity_dist_vectors=None, inputs_embeds=None, **kwargs
    ):
        if entity_dist_vectors is None or input_ids is None:
            return original_forward(
                input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs
            )

        token_embed_layer = _get_embed_tokens_module(model)
        inputs_embeds_local = token_embed_layer(input_ids)

        seq_len = inputs_embeds_local.shape[1]
        aligned_dist = _align_dist_to_seq(
            entity_dist_vectors,
            seq_len=seq_len,
            batch_size=inputs_embeds_local.shape[0],
            target_device=inputs_embeds_local.device,
            target_dtype=inputs_embeds_local.dtype,
        )

        projector, fusion_gate = _get_edef_modules(model)
        projected = projector(aligned_dist)
        fused = fusion_gate(inputs_embeds_local, projected)

        return original_forward(inputs_embeds=fused, **kwargs)

    model.forward = edef_forward
    return model


class EDEFWrapper(nn.Module):
    def __init__(
        self, model: nn.Module, dist_dim: int = 45, hidden_dim: int = 2560
    ) -> None:
        super().__init__()
        self.model = model
        self.entity_projector = EntityDistProjector(dist_dim, hidden_dim)
        self.fusion_gate = GatedFusion(hidden_dim)
        self._dist_vectors = None
        embed_tokens = _get_embed_tokens_module(self.model)
        self._hook_handle = embed_tokens.register_forward_hook(self._embedding_hook)

    def set_dist_vectors(self, dist_vectors: torch.Tensor) -> None:
        self._dist_vectors = dist_vectors

    def clear_dist_vectors(self) -> None:
        self._dist_vectors = None

    def get_trainable_params(self):
        return list(self.entity_projector.parameters()) + list(
            self.fusion_gate.parameters()
        )

    def _embedding_hook(self, module, inputs, output):
        if self._dist_vectors is None:
            return output

        aligned_dist = _align_dist_to_seq(
            self._dist_vectors,
            seq_len=output.shape[1],
            batch_size=output.shape[0],
            target_device=output.device,
            target_dtype=output.dtype,
        )
        projected = self.entity_projector(aligned_dist)
        return self.fusion_gate(output, projected)

    def forward(self, *args, entity_dist_vectors=None, **kwargs):
        if entity_dist_vectors is not None:
            self.set_dist_vectors(entity_dist_vectors)
        try:
            return self.model(*args, **kwargs)
        finally:
            self.clear_dist_vectors()

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


def load_edef_checkpoint(model, checkpoint_path):
    projector_path = os.path.join(checkpoint_path, "entity_projector.pt")
    fusion_path = os.path.join(checkpoint_path, "fusion_gate.pt")

    projector_module, fusion_module = _get_edef_modules(model)
    projector_module = resolve_trainable_module(projector_module)
    fusion_module = resolve_trainable_module(fusion_module)

    projector_state = _normalize_saved_module_state(
        torch.load(projector_path, map_location="cpu")
    )
    fusion_state = _normalize_saved_module_state(
        torch.load(fusion_path, map_location="cpu")
    )

    projector_module.load_state_dict(projector_state)
    fusion_module.load_state_dict(fusion_state)

    device = _get_model_device(model)
    projector_module.to(device)
    fusion_module.to(device)

    return model


def save_edef_checkpoint(model, save_path):
    projector_module, fusion_module = _get_edef_modules(model)
    projector_module = resolve_trainable_module(projector_module)
    fusion_module = resolve_trainable_module(fusion_module)

    os.makedirs(save_path, exist_ok=True)
    torch.save(
        projector_module.state_dict(), os.path.join(save_path, "entity_projector.pt")
    )
    torch.save(fusion_module.state_dict(), os.path.join(save_path, "fusion_gate.pt"))


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
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided."
                    )
                inputs_embeds = self.model.embed_tokens(input_ids)
            logits = self.lm_head(inputs_embeds)
            return {"logits": logits}

    torch.manual_seed(7)
    batch_size = 2
    seq_len = 16
    dist_dim = 45
    hidden_dim = 2560
    vocab_size = 1000

    mock_model = MockCausalLM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    attach_edef_to_model(mock_model, dist_dim=dist_dim, hidden_dim=hidden_dim)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    entity_dist_vectors = torch.randn(batch_size, seq_len - 3, dist_dim)

    outputs = mock_model(input_ids=input_ids, entity_dist_vectors=entity_dist_vectors)
    logits = outputs["logits"]
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, (
        f"Unexpected logits shape: got {tuple(logits.shape)}, expected {expected_shape}"
    )

    attached_projector, attached_gate = _get_edef_modules(mock_model)
    projector_params = sum(p.numel() for p in attached_projector.parameters())
    gate_params = sum(p.numel() for p in attached_gate.parameters())
    print(f"Projector parameters: {projector_params:,}")
    print(f"Fusion gate parameters: {gate_params:,}")
    print(f"Total EDEF parameters: {projector_params + gate_params:,}")

    with tempfile.TemporaryDirectory(prefix="edef_ckpt_") as ckpt_dir:
        save_edef_checkpoint(mock_model, ckpt_dir)

        projector_before_state = {
            key: value.detach().clone()
            for key, value in attached_projector.state_dict().items()
        }
        with torch.no_grad():
            first_param = next(attached_projector.parameters())
            first_param.add_(1.0)

        load_edef_checkpoint(mock_model, ckpt_dir)
        for key, value in attached_projector.state_dict().items():
            assert torch.allclose(projector_before_state[key], value), (
                "Checkpoint reload failed: entity_projector weights did not restore correctly."
            )

    print("Smoke test passed: forward path, shape check, and save/load all succeeded.")
