from __future__ import annotations

import importlib
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM

if __package__:
    _crf_mod = importlib.import_module(".crf", package=__package__)
else:
    _crf_mod = importlib.import_module("encoder_ner.crf")

ConstrainedLinearChainCRF = _crf_mod.ConstrainedLinearChainCRF

if __package__:
    _edef_mod = importlib.import_module("edef_model")
else:
    _edef_mod = importlib.import_module("edef_model")

attach_edef_to_model = _edef_mod.attach_edef_to_model
load_edef_checkpoint = _edef_mod.load_edef_checkpoint
resolve_trainable_module = _edef_mod.resolve_trainable_module


def resolve_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if torch.cuda.is_available() and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _single_device_map() -> dict[str, str] | None:
    if torch.cuda.is_available():
        return {"": "cuda:0"}
    return None


def _get_embed_tokens_module(model: nn.Module) -> nn.Module:
    candidates = [
        getattr(getattr(model, "model", None), "embed_tokens", None),
        getattr(
            getattr(getattr(model, "base_model", None), "model", None),
            "embed_tokens",
            None,
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return candidate
    raise AttributeError("Could not locate embed_tokens on backbone model")


def _get_decoder_module(model: nn.Module) -> nn.Module:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            return candidate
    raise AttributeError("Could not locate decoder module on backbone model")


def _get_edef_modules(model: nn.Module) -> tuple[nn.Module, nn.Module] | None:
    projector = getattr(model, "entity_projector", None)
    gate = getattr(model, "fusion_gate", None)
    if not isinstance(projector, nn.Module) or not isinstance(gate, nn.Module):
        return None
    return resolve_trainable_module(projector), resolve_trainable_module(gate)


def _align_dist_to_seq(
    dist_vectors: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dist_vectors.dim() != 3:
        raise ValueError(
            "entity_dist_vectors must have shape (batch_size, seq_len, dist_dim)"
        )
    if dist_vectors.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch: dist batch={dist_vectors.shape[0]} vs hidden batch={batch_size}"
        )

    aligned = dist_vectors.to(device=device, dtype=dtype)
    if aligned.shape[1] > seq_len:
        return aligned[:, :seq_len, :]
    if aligned.shape[1] < seq_len:
        padding = torch.zeros(
            aligned.shape[0],
            seq_len - aligned.shape[1],
            aligned.shape[2],
            device=device,
            dtype=dtype,
        )
        return torch.cat([aligned, padding], dim=1)
    return aligned


class TokenClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        head_type: str = "bilstm",
        head_hidden_dim: int = 512,
        dropout: float = 0.1,
        label_list: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_labels = num_labels
        self.use_crf = head_type.endswith("_crf")

        if head_type == "linear":
            self.projection = None
            self.sequence_encoder = None
            classifier_input_dim = input_dim
        elif head_type in {"bilstm", "bilstm_crf"}:
            sequence_dim = max(2, int(head_hidden_dim))
            if sequence_dim % 2 != 0:
                sequence_dim += 1
            self.projection = nn.Linear(input_dim, sequence_dim)
            self.sequence_encoder = nn.LSTM(
                input_size=sequence_dim,
                hidden_size=sequence_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            classifier_input_dim = sequence_dim
        else:
            raise ValueError(
                f"Unsupported head_type={head_type!r}. Expected 'linear', 'bilstm', or 'bilstm_crf'."
            )

        if self.use_crf and label_list is None:
            raise ValueError("label_list must be provided when using a CRF head")
        self.output = nn.Linear(classifier_input_dim, num_labels)
        self.crf = (
            ConstrainedLinearChainCRF(num_labels, label_list) if self.use_crf else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        if self.projection is not None:
            x = self.projection(x)
        if self.sequence_encoder is not None:
            x, _ = self.sequence_encoder(x)
        x = self.dropout(x)
        return self.output(x)

    def loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.crf is None:
            return F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        if attention_mask is None:
            active_mask = labels.ne(-100)
        else:
            active_mask = attention_mask.bool() & labels.ne(-100)

        safe_labels = labels.masked_fill(~active_mask, 0)
        return self.crf.neg_log_likelihood(
            logits.float(), safe_labels.long(), active_mask
        )

    def decode(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if attention_mask is None:
            attention_mask = torch.ones(
                logits.shape[:2], device=logits.device, dtype=torch.bool
            )
        else:
            attention_mask = attention_mask.bool()

        if self.crf is not None:
            return self.crf.decode(logits.float(), attention_mask)

        argmax_predictions = logits.argmax(dim=-1)
        decoded: list[list[int]] = []
        lengths = attention_mask.long().sum(dim=1)
        for row, length in zip(argmax_predictions, lengths):
            decoded.append(row[: int(length.item())].tolist())
        return decoded


class DecoderBackboneTokenClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_labels: int,
        head_type: str = "bilstm",
        head_hidden_dim: int = 512,
        dropout: float = 0.1,
        label_list: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = getattr(backbone, "config", None)
        self.num_labels = num_labels
        hidden_size = int(getattr(self.config, "hidden_size", 2560))
        self.classifier = TokenClassificationHead(
            input_dim=hidden_size,
            num_labels=num_labels,
            head_type=head_type,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            label_list=label_list,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        entity_dist_vectors: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        decoder = _get_decoder_module(self.backbone)
        edef_modules = _get_edef_modules(self.backbone)
        model_kwargs = {key: value for key, value in kwargs.items() if key != "labels"}
        model_kwargs.setdefault("return_dict", True)
        model_kwargs.setdefault("use_cache", False)

        if entity_dist_vectors is None or edef_modules is None:
            outputs = decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_kwargs,
            )
            return outputs.last_hidden_state

        embed_tokens = _get_embed_tokens_module(self.backbone)
        inputs_embeds = embed_tokens(input_ids)
        projector, fusion_gate = edef_modules
        aligned_dist = _align_dist_to_seq(
            entity_dist_vectors,
            seq_len=inputs_embeds.shape[1],
            batch_size=inputs_embeds.shape[0],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        projected = projector(aligned_dist)
        fused = fusion_gate(inputs_embeds, projected)
        outputs = decoder(
            inputs_embeds=fused,
            attention_mask=attention_mask,
            **model_kwargs,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        entity_dist_vectors: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        hidden_states = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_dist_vectors=entity_dist_vectors,
            **kwargs,
        )
        logits = self.classifier(hidden_states)
        output = {"logits": logits}

        if labels is not None:
            output["loss"] = self.classifier.loss(logits, labels, attention_mask)
        return output

    def decode_predictions(
        self,
        logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[list[int]]:
        return self.classifier.decode(logits, attention_mask)


def load_phase1_model(
    phase1_model: str,
    phase1_adapter: str | None = None,
    base_model: str = "unsloth/Qwen3-4B-Instruct-2507",
    torch_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
) -> nn.Module:
    dtype = resolve_dtype() if torch_dtype is None else torch_dtype
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": _single_device_map(),
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if phase1_adapter:
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(model, phase1_adapter)
        model = model.merge_and_unload()
        return model

    return AutoModelForCausalLM.from_pretrained(phase1_model, **model_kwargs)


def load_stage2_backbone(
    phase1_model: str,
    phase1_adapter: str | None = None,
    base_model: str = "unsloth/Qwen3-4B-Instruct-2507",
    stage2_adapter: str | None = None,
    stage2_edef_checkpoint: str | None = None,
    dist_dim: int | None = None,
    attn_implementation: str | None = None,
    use_edef: bool = True,
) -> nn.Module:
    backbone = load_phase1_model(
        phase1_model=phase1_model,
        phase1_adapter=phase1_adapter,
        base_model=base_model,
        attn_implementation=attn_implementation,
    )

    if use_edef:
        if dist_dim is None:
            raise ValueError("dist_dim must be provided when use_edef=True")
        hidden_dim = int(getattr(backbone.config, "hidden_size", 2560))
        backbone = attach_edef_to_model(
            backbone,
            dist_dim=dist_dim,
            hidden_dim=hidden_dim,
        )

    if stage2_adapter:
        backbone = PeftModel.from_pretrained(backbone, stage2_adapter)
        resolved_edef_ckpt = stage2_edef_checkpoint
        if resolved_edef_ckpt is None:
            candidate = os.path.join(stage2_adapter, "edef_checkpoint")
            if os.path.exists(candidate):
                resolved_edef_ckpt = candidate
        if use_edef and resolved_edef_ckpt and os.path.exists(resolved_edef_ckpt):
            load_edef_checkpoint(backbone.base_model.model, resolved_edef_ckpt)
        backbone = backbone.merge_and_unload()
        return backbone

    if use_edef and stage2_edef_checkpoint and os.path.exists(stage2_edef_checkpoint):
        load_edef_checkpoint(backbone, stage2_edef_checkpoint)
    return backbone


def freeze_model_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def enable_gradient_checkpointing(model: nn.Module) -> None:
    checkpointing_fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(checkpointing_fn):
        checkpointing_fn()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        setattr(config, "use_cache", False)
