"""Late-correction EDEF modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class EntityDistProjector(nn.Module):
    """Project 45-d entity distributions into the model hidden space."""

    def __init__(
        self,
        dist_dim: int = 45,
        hidden_dim: int = 2560,
        bottleneck_dim: int | None = None,
        use_temperature_scaling: bool = False,
    ) -> None:
        super().__init__()
        self.bottleneck_dim = (
            int(bottleneck_dim) if bottleneck_dim is not None else None
        )
        if self.bottleneck_dim is not None and self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive when provided")

        self.use_temperature_scaling = bool(use_temperature_scaling)
        if self.use_temperature_scaling:
            self.log_temperature = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("log_temperature", None)

        if self.bottleneck_dim is None:
            self.projector = nn.Sequential(
                nn.Linear(dist_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(dist_dim, self.bottleneck_dim),
                nn.GELU(),
                nn.Linear(self.bottleneck_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def _scale_distributions(self, dist_vectors: torch.Tensor) -> torch.Tensor:
        if self.log_temperature is None:
            return dist_vectors

        safe_dist = dist_vectors.float().clamp_min(1e-6)
        temperature = self.log_temperature.float().exp().clamp(0.25, 4.0)
        scaled = torch.softmax(torch.log(safe_dist) / temperature, dim=-1)
        return scaled.to(dtype=dist_vectors.dtype)

    def forward(self, dist_vectors: torch.Tensor) -> torch.Tensor:
        scaled = self._scale_distributions(dist_vectors)
        return self.projector(scaled)


class GatedFusion(nn.Module):
    """Late gated residual correction.

    The gate starts near identity so the late correction path can warm up
    without destabilizing the frozen backbone.
    """

    def __init__(
        self, hidden_dim: int = 2560, normalize_projected: bool = False
    ) -> None:
        super().__init__()
        self.projected_norm = (
            nn.LayerNorm(hidden_dim) if normalize_projected else nn.Identity()
        )
        self.gate_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gate_net.weight, std=0.01)
        nn.init.constant_(self.gate_net.bias, -2.0)

    def forward(
        self,
        token_states: torch.Tensor,
        projected_features: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        normalized_features = self.projected_norm(projected_features)
        combined = torch.cat([token_states, normalized_features], dim=-1)
        gate = torch.sigmoid(self.gate_net(combined))
        if prompt_mask is not None:
            gate = gate * prompt_mask.to(gate.dtype).unsqueeze(-1)
        fused = token_states + gate * normalized_features
        if return_gate:
            return fused, gate
        return fused


class _CorrectionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.normal_(self.attn.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=1e-3)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_in = self.norm1(hidden_states)
        attn_out, _ = self.attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LateCorrectionTransformer(nn.Module):
    """Small bottleneck transformer that refines late hidden states."""

    def __init__(
        self,
        hidden_dim: int = 2560,
        bottleneck_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.layers = nn.ModuleList(
            _CorrectionBlock(bottleneck_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        )
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.output_gate = nn.Parameter(torch.full((hidden_dim,), -2.0))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.up_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if prompt_mask is not None and not prompt_mask.any():
            if return_gate:
                gate = torch.sigmoid(self.output_gate).view(1, 1, -1)
                return hidden_states, gate
            return hidden_states

        safe_mask = None
        if prompt_mask is not None:
            safe_mask = prompt_mask.bool().clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True

        hidden = self.down_proj(self.input_norm(hidden_states))
        key_padding_mask = None if safe_mask is None else ~safe_mask
        for layer in self.layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)

        delta = self.up_proj(hidden)
        gate = torch.sigmoid(self.output_gate).view(1, 1, -1)
        if prompt_mask is not None:
            delta = delta * prompt_mask.to(delta.dtype).unsqueeze(-1)

        corrected = hidden_states + gate * delta
        if return_gate:
            return corrected, gate
        return corrected


if __name__ == "__main__":
    print("=== Late Correction EDEF Modules Smoke Test ===\n")
    torch.manual_seed(7)

    batch, seq_len, dist_dim, hidden_dim = 2, 12, 45, 128
    projector = EntityDistProjector(
        dist_dim=dist_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=32,
        use_temperature_scaling=True,
    )
    gate = GatedFusion(hidden_dim=hidden_dim, normalize_projected=True)
    corrector = LateCorrectionTransformer(
        hidden_dim=hidden_dim,
        bottleneck_dim=32,
        num_layers=2,
        num_heads=4,
    )

    dist_vectors = torch.randn(batch, seq_len, dist_dim)
    hidden_states = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)
    prompt_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    prompt_mask[:, :8] = True

    projected = projector(dist_vectors)
    fused, gate_values = gate(
        hidden_states, projected, prompt_mask=prompt_mask, return_gate=True
    )
    corrected, corrector_gate = corrector(
        fused, prompt_mask=prompt_mask, return_gate=True
    )

    assert projected.shape == (batch, seq_len, hidden_dim)
    assert fused.shape == hidden_states.shape
    assert corrected.shape == hidden_states.shape
    assert gate_values.mean().item() > 0.0
    assert 0.05 < torch.sigmoid(gate.gate_net.bias).mean().item() < 0.25
    assert 0.05 < corrector_gate.mean().item() < 0.25
    assert torch.exp(projector.log_temperature).item() == 1.0

    loss = corrected.square().mean()
    loss.backward()
    grad_checks = {
        "projector": all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in projector.parameters()
        ),
        "gate": all(
            p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters()
        ),
        "corrector": all(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in corrector.parameters()
        ),
    }
    print("Gradient flow:", grad_checks)
    assert all(grad_checks.values())
    print("\n✓ Late correction module smoke tests passed!")
