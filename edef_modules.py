"""EDEF neural modules: EntityDistProjector and GatedFusion.

EntityDistProjector: 2-layer MLP that projects 45-dim entity type distribution
vectors into the LLM's 2560-dim embedding space (inspired by LLaVA).

GatedFusion: Learned per-token gate that controls how much distribution
information to inject into each token's embedding (inspired by GEMNET).
"""

import torch
import torch.nn as nn


class EntityDistProjector(nn.Module):
    """Projects entity distribution vectors (dim=45) into LLM embedding space (dim=2560).

    Architecture inspired by LLaVA's multimodal projector:
    2-layer MLP with GELU activation.

    Parameters: 45*2560 + 2560 + 2560*2560 + 2560 ≈ 6.67M
    """

    def __init__(self, dist_dim: int = 45, hidden_dim: int = 2560) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(dist_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.projector:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, dist_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_vectors: (batch, seq_len, 45) — entity type distributions per token
        Returns:
            (batch, seq_len, 2560) — projected features in LLM embedding space
        """
        compute_dtype = self.projector[0].weight.dtype
        with torch.autocast(device_type=dist_vectors.device.type, enabled=False):
            return self.projector(dist_vectors.to(dtype=compute_dtype))


class GatedFusion(nn.Module):
    """Gated fusion module inspired by GEMNET (NAACL 2021).

    Learns per-token gate values that control how much entity distribution
    information to inject into each token's embedding.

    Formula: h' = h + gate * projected_features
    where gate = σ(W · [h; projected_features])

    Parameters: (2560+2560)*2560 + 2560 ≈ 13.1M
    """

    def __init__(self, hidden_dim: int = 2560) -> None:
        super().__init__()
        self.gate_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gate_net.weight, std=0.01)
        nn.init.constant_(self.gate_net.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self, token_embeddings: torch.Tensor, projected_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch, seq_len, 2560) — from Qwen3 embed_tokens
            projected_features: (batch, seq_len, 2560) — from EntityDistProjector
        Returns:
            (batch, seq_len, 2560) — fused embeddings
        """
        output_dtype = token_embeddings.dtype
        compute_dtype = self.gate_net.weight.dtype
        with torch.autocast(device_type=token_embeddings.device.type, enabled=False):
            token_embeddings_fp = token_embeddings.to(dtype=compute_dtype)
            projected_features_fp = projected_features.to(dtype=compute_dtype)
            combined = torch.cat([token_embeddings_fp, projected_features_fp], dim=-1)
            gate = torch.sigmoid(self.gate_net(combined))
            fused = token_embeddings_fp + gate * projected_features_fp
        return fused.to(dtype=output_dtype)


if __name__ == "__main__":
    # Smoke test: verify shapes, init values, gradient flow
    print("=== EDEF Modules Smoke Test ===\n")

    batch, seq_len, dist_dim, hidden_dim = 4, 128, 45, 2560

    projector = EntityDistProjector(dist_dim, hidden_dim)
    gate = GatedFusion(hidden_dim)

    proj_params = sum(p.numel() for p in projector.parameters())
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"Projector params: {proj_params:,} ({proj_params / 1e6:.2f}M)")
    print(f"Gate params:      {gate_params:,} ({gate_params / 1e6:.2f}M)")
    print(
        f"Total EDEF params: {(proj_params + gate_params):,} ({(proj_params + gate_params) / 1e6:.2f}M)\n"
    )

    # Forward pass
    dist_vectors = torch.randn(batch, seq_len, dist_dim)
    token_embeds = torch.randn(batch, seq_len, hidden_dim)

    projected = projector(dist_vectors)
    print(f"Projector input:  {dist_vectors.shape}")
    print(f"Projector output: {projected.shape}")
    assert projected.shape == (batch, seq_len, hidden_dim), "Projector shape mismatch!"

    fused = gate(token_embeds, projected)
    print(f"Gate input embeds: {token_embeds.shape}")
    print(f"Gate output:       {fused.shape}")
    assert fused.shape == (batch, seq_len, hidden_dim), "Gate shape mismatch!"

    # Verify initial gate values near 0.12
    with torch.no_grad():
        combined = torch.cat([token_embeds, projected], dim=-1)
        gate_vals = torch.sigmoid(gate.gate_net(combined))
        print(
            f"\nInitial gate stats: mean={gate_vals.mean():.4f}, "
            f"std={gate_vals.std():.4f}, min={gate_vals.min():.4f}, max={gate_vals.max():.4f}"
        )
        assert 0.05 < gate_vals.mean().item() < 0.25, (
            f"Gate mean {gate_vals.mean():.4f} not near 0.12!"
        )

    # Verify gradient flow
    loss = fused.sum()
    loss.backward()
    proj_has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters()
    )
    gate_has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters()
    )
    print(f"\nProjector gradients flow: {proj_has_grad}")
    print(f"Gate gradients flow:      {gate_has_grad}")
    assert proj_has_grad, "No gradients in projector!"
    assert gate_has_grad, "No gradients in gate!"

    # Verify near-identity at init (fused ≈ token_embeds)
    projector.zero_grad()
    gate.zero_grad()
    with torch.no_grad():
        diff = (fused - token_embeds).abs().mean()
        rel_diff = diff / token_embeds.abs().mean()
        print(f"\nFusion contribution (abs mean diff): {diff:.6f}")
        print(f"Relative contribution: {rel_diff:.6f} (should be small)")

    print("\n✓ All smoke tests passed!")
