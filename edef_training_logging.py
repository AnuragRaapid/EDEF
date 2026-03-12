from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from edef_model import get_edef_host


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _group_named_params(
    model: Any, include_lora: bool
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "projector": [],
        "gate": [],
        "corrector": [],
    }
    if include_lora:
        grouped["lora"] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "entity_projector" in name:
            grouped["projector"].append((name, param))
        elif "fusion_gate" in name:
            grouped["gate"].append((name, param))
        elif "late_corrector" in name:
            grouped["corrector"].append((name, param))
        elif include_lora and "lora_" in name:
            grouped["lora"].append((name, param))

    return {group: params for group, params in grouped.items() if params}


def _summarize_param_group(
    named_params: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    tensor_count = len(named_params)
    param_count = 0
    grad_param_count = 0
    tensors_with_grad = 0
    param_sumsq = 0.0
    grad_sumsq = 0.0
    grad_abs_max = 0.0

    for _, param in named_params:
        param_data = param.detach().float()
        param_count += int(param_data.numel())
        param_sumsq += float(param_data.square().sum().item())

        if param.grad is None:
            continue

        grad_data = param.grad.detach().float()
        tensors_with_grad += 1
        grad_param_count += int(grad_data.numel())
        grad_sumsq += float(grad_data.square().sum().item())
        grad_abs_max = max(grad_abs_max, float(grad_data.abs().max().item()))

    stats: dict[str, Any] = {
        "tensor_count": tensor_count,
        "param_count": param_count,
        "tensors_with_grad": tensors_with_grad,
        "param_rms": math.sqrt(param_sumsq / param_count) if param_count > 0 else None,
    }
    if tensors_with_grad > 0 and param_count > 0:
        stats["grad_rms"] = math.sqrt(grad_sumsq / grad_param_count)
        stats["grad_abs_max"] = grad_abs_max
    else:
        stats["grad_rms"] = None
        stats["grad_abs_max"] = None
    return stats


def _snapshot_params(
    named_params: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in named_params}


def _summarize_updates(
    named_params: list[tuple[str, torch.nn.Parameter]],
    before_snapshot: dict[str, torch.Tensor],
    pre_step_stats: dict[str, Any],
) -> dict[str, Any]:
    if not before_snapshot:
        return {}

    update_sumsq = 0.0
    update_abs_max = 0.0
    param_count = 0

    for name, param in named_params:
        before = before_snapshot.get(name)
        if before is None:
            continue
        delta = param.detach().float() - before.float()
        param_count += int(delta.numel())
        update_sumsq += float(delta.square().sum().item())
        update_abs_max = max(update_abs_max, float(delta.abs().max().item()))

    if param_count == 0:
        return {}

    update_rms = math.sqrt(update_sumsq / param_count)
    param_rms = pre_step_stats.get("param_rms")
    return {
        "update_rms": update_rms,
        "update_abs_max": update_abs_max,
        "update_to_param_rms": _safe_ratio(update_rms, param_rms),
    }


def _extract_optimizer_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lr_map: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("group_name", f"group_{idx}"))
        lr_map[group_name] = float(group["lr"])
    return lr_map


def _format_ratio(value: float | None, precision: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}x"


class EDEFTrainingDiagnosticsCallback(TrainerCallback):
    def __init__(
        self,
        *,
        stage_name: str,
        log_every_steps: int = 100,
        include_lora: bool = False,
        model: Any | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.log_every_steps = log_every_steps
        self.include_lora = include_lora
        self.bound_model = model
        self.log_path: Path | None = None
        self._pending_step: int | None = None
        self._pending_stats: dict[str, Any] | None = None
        self._pending_snapshots: dict[str, dict[str, torch.Tensor]] = {}

    def _should_log(self, state: Any) -> bool:
        if self.log_every_steps <= 0:
            return False
        if not getattr(state, "is_local_process_zero", True):
            return False
        next_step = int(state.global_step) + 1
        return next_step > 0 and next_step % self.log_every_steps == 0

    def _resolve_model(self, kwargs_model: Any | None) -> Any | None:
        return kwargs_model if kwargs_model is not None else self.bound_model

    def _build_ratio_block(
        self, group_stats: dict[str, dict[str, Any]]
    ) -> dict[str, float | None]:
        projector = group_stats.get("projector", {})
        gate = group_stats.get("gate", {})
        ratios: dict[str, float | None] = {
            "projector_to_gate_grad_rms_ratio": _safe_ratio(
                projector.get("grad_rms"),
                gate.get("grad_rms"),
            ),
            "projector_to_gate_grad_abs_max_ratio": _safe_ratio(
                projector.get("grad_abs_max"),
                gate.get("grad_abs_max"),
            ),
        }

        gate_update = gate.get("update_rms")
        projector_update = projector.get("update_rms")
        ratios["projector_to_gate_update_rms_ratio"] = _safe_ratio(
            projector_update,
            gate_update,
        )

        if "corrector" in group_stats:
            corrector = group_stats["corrector"]
            ratios["corrector_to_gate_grad_rms_ratio"] = _safe_ratio(
                corrector.get("grad_rms"),
                gate.get("grad_rms"),
            )
            ratios["corrector_to_gate_update_rms_ratio"] = _safe_ratio(
                corrector.get("update_rms"),
                gate_update,
            )

        if "lora" in group_stats:
            lora = group_stats["lora"]
            ratios["lora_to_gate_grad_rms_ratio"] = _safe_ratio(
                lora.get("grad_rms"),
                gate.get("grad_rms"),
            )

        return ratios

    def _build_aux_metrics(self, model: Any) -> dict[str, Any]:
        host = get_edef_host(model)
        gate_sigmoid = torch.sigmoid(host.fusion_gate.gate_net.bias.detach().float())
        metrics: dict[str, Any] = {
            "gate_sigmoid_mean": float(gate_sigmoid.mean().item()),
            "gate_sigmoid_min": float(gate_sigmoid.min().item()),
            "gate_sigmoid_max": float(gate_sigmoid.max().item()),
        }

        log_temperature = getattr(host.entity_projector, "log_temperature", None)
        if isinstance(log_temperature, torch.nn.Parameter):
            metrics["projector_temperature"] = float(
                log_temperature.detach().float().exp().item()
            )

        corrector = getattr(host, "late_corrector", None)
        if corrector is not None and hasattr(corrector, "output_gate"):
            corrector_gate = torch.sigmoid(corrector.output_gate.detach().float())
            metrics["corrector_gate_mean"] = float(corrector_gate.mean().item())
            metrics["corrector_gate_min"] = float(corrector_gate.min().item())
            metrics["corrector_gate_max"] = float(corrector_gate.max().item())

        return metrics

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        del state, control, kwargs
        if self.log_every_steps <= 0 or not getattr(args, "process_index", 0) == 0:
            return
        self.log_path = (
            Path(args.output_dir) / f"{self.stage_name}_training_diagnostics.jsonl"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{self.stage_name} diagnostics will be written to {self.log_path}")

    def on_pre_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        del args, control
        model = self._resolve_model(kwargs.get("model"))
        optimizer = kwargs.get("optimizer")
        if model is None or optimizer is None or not self._should_log(state):
            return

        next_step = int(state.global_step) + 1
        grouped_params = _group_named_params(model, include_lora=self.include_lora)
        group_stats = {
            group_name: _summarize_param_group(named_params)
            for group_name, named_params in grouped_params.items()
        }
        self._pending_snapshots = {
            group_name: _snapshot_params(named_params)
            for group_name, named_params in grouped_params.items()
            if group_name != "lora"
        }
        self._pending_step = next_step
        self._pending_stats = {
            "stage": self.stage_name,
            "step": next_step,
            "optimizer_lrs": _extract_optimizer_lrs(optimizer),
            "aux_metrics": self._build_aux_metrics(model),
            "groups": group_stats,
        }

    def on_optimizer_step(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        del args, control
        model = self._resolve_model(kwargs.get("model"))
        if model is None or self._pending_step is None or self._pending_stats is None:
            return

        expected_step = int(state.global_step) + 1
        if expected_step != self._pending_step:
            return

        grouped_params = _group_named_params(model, include_lora=self.include_lora)
        for group_name, named_params in grouped_params.items():
            if group_name not in self._pending_snapshots:
                continue
            update_stats = _summarize_updates(
                named_params,
                self._pending_snapshots[group_name],
                self._pending_stats["groups"].get(group_name, {}),
            )
            self._pending_stats["groups"][group_name].update(update_stats)

        self._pending_stats["ratios"] = self._build_ratio_block(
            self._pending_stats["groups"]
        )
        self._emit_payload()
        self._pending_step = None
        self._pending_stats = None
        self._pending_snapshots = {}

    def _emit_payload(self) -> None:
        if self._pending_stats is None:
            return

        payload = self._pending_stats
        aux = payload["aux_metrics"]
        ratios = payload["ratios"]
        message_parts = [
            f"Step {payload['step']}",
            f"gate sigmoid mean={aux['gate_sigmoid_mean']:.4f}",
            f"proj/gate grad rms={_format_ratio(ratios.get('projector_to_gate_grad_rms_ratio'))}",
            f"proj/gate update rms={_format_ratio(ratios.get('projector_to_gate_update_rms_ratio'), precision=2)}",
        ]
        if "projector_temperature" in aux:
            message_parts.append(f"temp={aux['projector_temperature']:.4f}")
        if "corrector_gate_mean" in aux:
            message_parts.append(
                f"corrector gate mean={aux['corrector_gate_mean']:.4f}"
            )
        print(", ".join(message_parts))

        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


class Stage1DiagnosticsCallback(EDEFTrainingDiagnosticsCallback):
    def __init__(self, model: Any | None = None, log_every_steps: int = 100) -> None:
        super().__init__(
            stage_name="stage1",
            log_every_steps=log_every_steps,
            include_lora=False,
            model=model,
        )


class Stage2DiagnosticsCallback(EDEFTrainingDiagnosticsCallback):
    def __init__(
        self,
        model: Any | None = None,
        log_every_steps: int = 100,
        stage_name: str = "stage2",
    ) -> None:
        super().__init__(
            stage_name=stage_name,
            log_every_steps=log_every_steps,
            include_lora=True,
            model=model,
        )
