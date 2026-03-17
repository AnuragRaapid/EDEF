from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

NEG_INF = -10000.0


def _split_bio_label(label: str) -> tuple[str, str]:
    if label == "O":
        return "O", ""
    prefix, _, entity_type = label.partition("-")
    return prefix, entity_type


def _is_bio_start_allowed(label: str) -> bool:
    prefix, _ = _split_bio_label(label)
    return prefix in {"O", "B"}


def _is_bio_transition_allowed(prev_label: str, next_label: str) -> bool:
    prev_prefix, prev_type = _split_bio_label(prev_label)
    next_prefix, next_type = _split_bio_label(next_label)

    if next_prefix == "O":
        return True
    if next_prefix == "B":
        return True
    if next_prefix != "I":
        return False
    if prev_prefix in {"B", "I"} and prev_type == next_type:
        return True
    return False


def build_bio_constraints(
    label_list: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_labels = len(label_list)
    start_constraints = torch.full((num_labels,), NEG_INF, dtype=torch.float32)
    end_constraints = torch.zeros((num_labels,), dtype=torch.float32)
    transition_constraints = torch.full(
        (num_labels, num_labels), NEG_INF, dtype=torch.float32
    )

    for idx, label in enumerate(label_list):
        if _is_bio_start_allowed(str(label)):
            start_constraints[idx] = 0.0
        end_constraints[idx] = 0.0

    for prev_idx, prev_label in enumerate(label_list):
        for next_idx, next_label in enumerate(label_list):
            if _is_bio_transition_allowed(str(prev_label), str(next_label)):
                transition_constraints[prev_idx, next_idx] = 0.0

    return start_constraints, end_constraints, transition_constraints


class ConstrainedLinearChainCRF(nn.Module):
    def __init__(self, num_tags: int, label_list: Sequence[str]) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        start_constraints, end_constraints, transition_constraints = (
            build_bio_constraints(label_list)
        )
        self.register_buffer("start_constraints", start_constraints)
        self.register_buffer("end_constraints", end_constraints)
        self.register_buffer("transition_constraints", transition_constraints)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _constrained_start(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return self.start_transitions.to(
            dtype=dtype, device=device
        ) + self.start_constraints.to(dtype=dtype, device=device)

    def _constrained_end(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return self.end_transitions.to(
            dtype=dtype, device=device
        ) + self.end_constraints.to(dtype=dtype, device=device)

    def _constrained_transitions(
        self, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        return self.transitions.to(
            dtype=dtype, device=device
        ) + self.transition_constraints.to(dtype=dtype, device=device)

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape (batch, seq_len, num_tags)")
        if tags.shape != emissions.shape[:2]:
            raise ValueError("tags must have shape (batch, seq_len)")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask must have shape (batch, seq_len)")

        safe_mask = mask.bool()
        if not torch.all(safe_mask[:, 0]):
            raise ValueError(
                "CRF requires the first timestep of every sequence to be valid"
            )

        numerator = self._compute_score(emissions, tags, safe_mask)
        denominator = self._compute_log_partition(emissions, safe_mask)
        return (denominator - numerator).mean()

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        dtype = emissions.dtype
        device = emissions.device
        start = self._constrained_start(dtype, device)
        end = self._constrained_end(dtype, device)
        transitions = self._constrained_transitions(dtype, device)

        batch_index = torch.arange(batch_size, device=device)
        first_tags = tags[:, 0]
        score = start[first_tags] + emissions[batch_index, 0, first_tags]

        for timestep in range(1, seq_len):
            current_mask = mask[:, timestep]
            prev_tags = tags[:, timestep - 1]
            current_tags = tags[:, timestep]
            transition_score = transitions[prev_tags, current_tags]
            emission_score = emissions[batch_index, timestep, current_tags]
            score += (transition_score + emission_score) * current_mask.to(dtype)

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[batch_index, lengths]
        score += end[last_tags]
        return score

    def _compute_log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        del batch_size
        dtype = emissions.dtype
        device = emissions.device
        start = self._constrained_start(dtype, device)
        end = self._constrained_end(dtype, device)
        transitions = self._constrained_transitions(dtype, device)

        score = start.unsqueeze(0) + emissions[:, 0]
        for timestep in range(1, seq_len):
            next_score = (
                score.unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, timestep].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            current_mask = mask[:, timestep].unsqueeze(1)
            score = torch.where(current_mask, next_score, score)

        score = score + end.unsqueeze(0)
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape (batch, seq_len, num_tags)")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask must have shape (batch, seq_len)")

        safe_mask = mask.bool()
        if not torch.all(safe_mask[:, 0]):
            raise ValueError(
                "CRF requires the first timestep of every sequence to be valid"
            )

        batch_size, seq_len, num_tags = emissions.shape
        del num_tags
        dtype = emissions.dtype
        device = emissions.device
        start = self._constrained_start(dtype, device)
        end = self._constrained_end(dtype, device)
        transitions = self._constrained_transitions(dtype, device)

        score = start.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []

        for timestep in range(1, seq_len):
            next_score = score.unsqueeze(2) + transitions.unsqueeze(0)
            next_score, indices = next_score.max(dim=1)
            next_score = next_score + emissions[:, timestep]
            current_mask = safe_mask[:, timestep].unsqueeze(1)
            score = torch.where(current_mask, next_score, score)
            history.append(indices)

        score = score + end.unsqueeze(0)
        best_last_tags = score.argmax(dim=1)
        lengths = safe_mask.long().sum(dim=1)

        decoded: list[list[int]] = []
        for batch_idx in range(batch_size):
            seq_length = int(lengths[batch_idx].item())
            best_tag = int(best_last_tags[batch_idx].item())
            best_path = [best_tag]
            for history_t in reversed(history[: max(seq_length - 1, 0)]):
                best_tag = int(history_t[batch_idx, best_tag].item())
                best_path.append(best_tag)
            best_path.reverse()
            decoded.append(best_path)
        return decoded
