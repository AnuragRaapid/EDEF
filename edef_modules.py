"""EDEF neural modules: EntityDistProjector, GatedFusion, ContextualDistRefiner.

EntityDistProjector: 2-layer MLP that projects entity type distribution
vectors into the LLM's embedding space (inspired by LLaVA).  Includes a
learnable default embedding for unknown tokens and distribution dropout
for regularisation.

GatedFusion: Learned per-token gate that controls how much distribution
information to inject into each token's embedding (inspired by GEMNET).

ContextualDistRefiner: Lightweight 1-layer BiLSTM that refines raw
distribution vectors using surrounding token context before projection.
"""

import torch
import torch.nn as nn


class EntityDistProjector(nn.Module):
    """Projects entity distribution vectors into LLM embedding space.

    Full internal pipeline:
      1. Replace near-zero vectors with a **learnable default** embedding.
      2. (optional) Contextually refine distributions with a lightweight BiLSTM.
      3. Apply **distribution dropout** during training.
      4. Project through a 2-layer MLP into the LLM hidden dimension.
    """

    def __init__(
        self,
        dist_dim: int = 45,
        hidden_dim: int = 2560,
        dist_dropout: float = 0.2,
        use_refiner: bool = True,
        refiner_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.dist_dim = dist_dim
        self.hidden_dim = hidden_dim
        self.dist_dropout = dist_dropout

        self.learned_default = nn.Parameter(torch.zeros(dist_dim))

        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = ContextualDistRefiner(dist_dim, refiner_hidden)
        else:
            self.refiner = None

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
        with torch.no_grad():
            self.learned_default[-1] = 1.0

    def forward(self, dist_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist_vectors: (batch, seq_len, dist_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        is_default = dist_vectors.abs().sum(dim=-1) < 1e-6
        if is_default.any():
            expanded = self.learned_default.expand_as(dist_vectors)
            dist_vectors = torch.where(is_default.unsqueeze(-1), expanded, dist_vectors)

        if self.refiner is not None:
            dist_vectors = self.refiner(dist_vectors)

        if self.training and self.dist_dropout > 0:
            keep = torch.bernoulli(
                torch.full(
                    (dist_vectors.shape[0], dist_vectors.shape[1], 1),
                    1.0 - self.dist_dropout,
                    device=dist_vectors.device,
                    dtype=dist_vectors.dtype,
                )
            )
            dist_vectors = dist_vectors * keep

        return self.projector(dist_vectors)


class GatedFusion(nn.Module):
    """Gated fusion module inspired by GEMNET (NAACL 2021).

    Formula: h' = h + gate * projected_features
    where gate = σ(W · [h; projected_features])
    """

    def __init__(self, hidden_dim: int = 2560) -> None:
        super().__init__()
        self.gate_net = nn.Linear(hidden_dim * 2, hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gate_net.weight, std=0.01)
        nn.init.constant_(self.gate_net.bias, -2.0)

    def forward(
        self, token_embeddings: torch.Tensor, projected_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch, seq_len, hidden_dim) — from embed_tokens
            projected_features: (batch, seq_len, hidden_dim) — from EntityDistProjector
        Returns:
            (batch, seq_len, hidden_dim) — fused embeddings
        """
        combined = torch.cat([token_embeddings, projected_features], dim=-1)
        gate = torch.sigmoid(self.gate_net(combined))
        return token_embeddings + gate * projected_features


class ContextualDistRefiner(nn.Module):
    """Lightweight BiLSTM that refines raw distribution vectors using context.

    Addresses the context-free limitation of raw gazetteer distributions:
    if an unseen word sits near known entity tokens the BiLSTM can propagate
    entity signal to it.  Uses a gated residual so the model can choose
    between the raw and refined distribution per position.
    """

    def __init__(self, dist_dim: int = 45, refiner_hidden: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            dist_dim, refiner_hidden, batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(refiner_hidden * 2, dist_dim)
        self.res_gate = nn.Linear(dist_dim * 2, dist_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.proj.bias)
        nn.init.constant_(self.res_gate.bias, -1.0)

    def forward(self, raw_dists: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_dists: (batch, seq_len, dist_dim)
        Returns:
            (batch, seq_len, dist_dim) — context-refined distributions
        """
        refined, _ = self.lstm(raw_dists)
        refined = self.proj(refined)
        gate = torch.sigmoid(
            self.res_gate(torch.cat([raw_dists, refined], dim=-1))
        )
        return gate * refined + (1.0 - gate) * raw_dists


if __name__ == "__main__":
    print("=== EDEF Modules Smoke Test ===\n")

    batch, seq_len, dist_dim, hidden_dim = 4, 128, 45, 2560

    projector = EntityDistProjector(
        dist_dim, hidden_dim, dist_dropout=0.2, use_refiner=True, refiner_hidden=64,
    )
    gate = GatedFusion(hidden_dim)

    proj_params = sum(p.numel() for p in projector.parameters())
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"Projector params (incl. refiner): {proj_params:,} ({proj_params/1e6:.2f}M)")
    print(f"Gate params:                      {gate_params:,} ({gate_params/1e6:.2f}M)")
    total_edef = proj_params + gate_params
    print(f"Total EDEF params:                {total_edef:,} ({total_edef/1e6:.2f}M)\n")

    dist_vectors = torch.randn(batch, seq_len, dist_dim)
    dist_vectors[:, :5, :] = 0.0  # first 5 tokens are "unknown" (zero → learned default)
    token_embeds = torch.randn(batch, seq_len, hidden_dim)

    # Test projector (training mode — dropout + refiner active)
    projector.train()
    projected = projector(dist_vectors)
    print(f"Projector input:  {dist_vectors.shape}")
    print(f"Projector output: {projected.shape}")
    assert projected.shape == (batch, seq_len, hidden_dim), "Projector shape mismatch!"

    # Test learnable default in eval mode
    projector.eval()
    zero_dist = torch.zeros(2, 10, dist_dim)
    proj_default = projector(zero_dist)
    print(f"Learned default forward OK: {proj_default.shape}")
    assert proj_default.shape == (2, 10, hidden_dim)

    # Test gate
    projector.train()
    projected = projector(dist_vectors)
    fused = gate(token_embeds, projected)
    print(f"Gate output: {fused.shape}")
    assert fused.shape == (batch, seq_len, hidden_dim), "Gate shape mismatch!"

    with torch.no_grad():
        combined = torch.cat([token_embeds, projected], dim=-1)
        gate_vals = torch.sigmoid(gate.gate_net(combined))
        print(f"\nInitial gate stats: mean={gate_vals.mean():.4f}, "
              f"std={gate_vals.std():.4f}, min={gate_vals.min():.4f}, max={gate_vals.max():.4f}")
        assert 0.05 < gate_vals.mean().item() < 0.25, f"Gate mean {gate_vals.mean():.4f} not near 0.12!"

    # Gradient flow
    loss = fused.sum()
    loss.backward()
    proj_has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in projector.parameters())
    gate_has_grad = all(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())
    print(f"\nProjector (incl. refiner) gradients: {proj_has_grad}")
    print(f"Gate gradients:                      {gate_has_grad}")
    assert proj_has_grad, "No gradients in projector!"
    assert gate_has_grad, "No gradients in gate!"

    # Verify learned_default specifically got gradients
    assert projector.learned_default.grad is not None, "learned_default has no gradient!"
    print(f"learned_default grad norm: {projector.learned_default.grad.norm():.6f}")

    # Also test without refiner
    proj_no_ref = EntityDistProjector(dist_dim, hidden_dim, dist_dropout=0.0, use_refiner=False)
    proj_no_ref.eval()
    out_no_ref = proj_no_ref(torch.randn(2, 10, dist_dim))
    assert out_no_ref.shape == (2, 10, hidden_dim), "No-refiner projector shape mismatch!"
    print(f"No-refiner projector OK: {out_no_ref.shape}")

    print("\n✓ All smoke tests passed!")
