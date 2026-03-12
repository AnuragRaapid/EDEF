# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedImport=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportImplicitRelativeImport=false, reportUnannotatedClassAttribute=false, reportMissingParameterType=false, reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    _dist_mod = importlib.import_module(".distribution_alignment", package=__package__)
    _edef_mod = importlib.import_module(".edef_model", package=__package__)
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _edef_mod = importlib.import_module("edef_model")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
attach_edef_to_model = _edef_mod.attach_edef_to_model
edef_runtime_context = _edef_mod.edef_runtime_context
get_edef_host = _edef_mod.get_edef_host
load_edef_checkpoint = _edef_mod.load_edef_checkpoint
load_edef_config = _edef_mod.load_edef_config


NER_INSTRUCTION = (
    "You are an expert medical Named Entity Recognition (NER) assistant. "
    "Your task is to extract and classify entities from the provided medical text. "
    "Output format should be {'ner': [['entity', 'type'], ['entity', 'type'],...]}"
)


def parse_ner_output(text: str) -> dict[str, Any]:
    import re

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*"ner".*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"ner": []}


def _build_chat_prompt(tokenizer: Any, text: str) -> str:
    messages = [
        {"role": "system", "content": NER_INSTRUCTION},
        {"role": "user", "content": text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{NER_INSTRUCTION}\n\n{text}"


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    value = dtype_name.strip().lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _build_prompt_features(
    prompt_text: str,
    tokenizer: Any,
    word_entity_dist: dict[str, list[float]],
    default_dist: list[float],
    dist_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoding = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    dist_vectors = (
        get_token_distributions(
            text=prompt_text,
            tokenizer=tokenizer,
            word_entity_dist=word_entity_dist,
            default_dist=default_dist,
            dist_dim=dist_dim,
        )
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    prompt_mask = torch.ones(
        (1, dist_vectors.shape[1]), dtype=torch.bool, device=device
    )
    return input_ids, attention_mask, dist_vectors, prompt_mask


class EDEFInferencePipeline:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        word_entity_dist: dict[str, list[float]],
        default_dist: list[float],
        dist_dim: int = 45,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.word_entity_dist = word_entity_dist
        self.default_dist = default_dist
        self.dist_dim = dist_dim
        self._edef_host = get_edef_host(model)
        self.device = next(self.model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        phase1_model: str,
        dist_path: str,
        base_model: str = "Qwen/Qwen3-4B-Instruct",
        dist_dim: int = 45,
        hidden_dim: int | None = None,
        insertion_layer: int | None = None,
        corrector_layers: int = 2,
        corrector_dim: int = 512,
        corrector_heads: int = 8,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> "EDEFInferencePipeline":
        dtype = _resolve_torch_dtype(torch_dtype)
        model_source = phase1_model if phase1_model else base_model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        edef_ckpt = os.path.join(model_path, "edef_checkpoint")
        saved_cfg = load_edef_config(edef_ckpt) if os.path.isdir(edef_ckpt) else {}
        hidden = (
            hidden_dim
            if hidden_dim is not None
            else int(getattr(model.config, "hidden_size", 2560))
        )
        model = attach_edef_to_model(
            model,
            dist_dim=int(saved_cfg.get("dist_dim", dist_dim)),
            hidden_dim=hidden,
            insertion_layer=int(
                saved_cfg.get(
                    "insertion_layer",
                    insertion_layer if insertion_layer is not None else 28,
                )
            ),
            projector_bottleneck_dim=saved_cfg.get("projector_bottleneck_dim"),
            projector_use_temperature=bool(
                saved_cfg.get("projector_use_temperature", False)
            ),
            fusion_projected_norm=bool(saved_cfg.get("fusion_projected_norm", False)),
            corrector_layers=int(saved_cfg.get("corrector_layers", corrector_layers)),
            corrector_dim=int(saved_cfg.get("corrector_dim", corrector_dim)),
            corrector_heads=int(saved_cfg.get("corrector_heads", corrector_heads)),
        )
        model = PeftModel.from_pretrained(model, model_path)
        if os.path.isdir(edef_ckpt):
            load_edef_checkpoint(model, edef_ckpt)

        tokenizer_source = (
            model_path
            if os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
            else model_source
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        word_entity_dist, default_dist = load_distributions(dist_path)
        model.eval()
        return cls(model, tokenizer, word_entity_dist, default_dist, dist_dim=dist_dim)

    def predict(
        self, text: str, max_new_tokens: int = 2048, temperature: float = 0.0
    ) -> dict[str, Any]:
        prompt_text = _build_chat_prompt(self.tokenizer, text)
        input_ids, attention_mask, dist_vectors, prompt_mask = _build_prompt_features(
            prompt_text,
            self.tokenizer,
            self.word_entity_dist,
            self.default_dist,
            self.dist_dim,
            self.device,
        )

        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            with edef_runtime_context(
                self.model,
                entity_dist_vectors=dist_vectors,
                entity_prompt_mask=prompt_mask,
                prefill_only=True,
            ):
                output_ids = self.model.generate(**generate_kwargs)

        prompt_len = input_ids.shape[1]
        completion_ids = output_ids[0, prompt_len:]
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return parse_ner_output(decoded)

    def predict_batch(
        self,
        texts: list[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> list[dict[str, Any]]:
        return [
            self.predict(text, max_new_tokens=max_new_tokens, temperature=temperature)
            for text in texts
        ]


def _load_texts_from_json(input_file: str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "--input_file JSON must be a list of strings or list of {input: text} objects."
        )

    texts: list[str] = []
    for item in data:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "input" in item:
            texts.append(str(item["input"]))
        else:
            raise ValueError(
                "Invalid JSON entry. Expected string or object with an 'input' field."
            )
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Late-correction EDEF clinical NER inference"
    )
    parser.add_argument(
        "--model_path", required=True, help="Stage 2 model path (LoRA + late EDEF)"
    )
    parser.add_argument(
        "--phase1_model", required=True, help="Phase 1 merged model path"
    )
    parser.add_argument(
        "--base_model", default="Qwen/Qwen3-4B-Instruct", help="Base model name"
    )
    parser.add_argument("--dist_path", required=True, help="Entity distributions JSON")

    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument("--input", help="Single text to process")
    io_group.add_argument("--input_file", help="JSON file with texts to process")

    parser.add_argument("--output_file", help="Save predictions JSON")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--torch_dtype", default="auto", help="auto|bfloat16|float16|float32"
    )
    parser.add_argument(
        "--trust_remote_code", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pipeline = EDEFInferencePipeline.from_pretrained(
        model_path=args.model_path,
        phase1_model=args.phase1_model,
        dist_path=args.dist_path,
        base_model=args.base_model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.input is not None:
        output_payload: Any = pipeline.predict(
            args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        texts = _load_texts_from_json(args.input_file)
        predictions = pipeline.predict_batch(
            texts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_payload = [
            {"input": text, "prediction": pred}
            for text, pred in zip(texts, predictions)
        ]

    output_text = json.dumps(output_payload, ensure_ascii=False, indent=2)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
