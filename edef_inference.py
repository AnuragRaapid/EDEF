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
    _medical_mod = importlib.import_module(".medical_alignment", package=__package__)
    _edef_mod = importlib.import_module(".edef_model", package=__package__)
    _dataset_utils_mod = importlib.import_module(
        ".ner_dataset_utils", package=__package__
    )
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _medical_mod = importlib.import_module("medical_alignment")
    _edef_mod = importlib.import_module("edef_model")
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
build_medical_alignment_features = _medical_mod.build_medical_alignment_features
DEFAULT_DIST_PATH = _dataset_utils_mod.DEFAULT_DIST_PATH
DEFAULT_PHASE1_MODEL_PATH = _dataset_utils_mod.DEFAULT_PHASE1_MODEL_PATH
load_task_metadata_from_dist_path = _dataset_utils_mod.load_task_metadata_from_dist_path
DEFAULT_MEDICAL_CHUNK_OVERLAP = _edef_mod.DEFAULT_MEDICAL_CHUNK_OVERLAP
DEFAULT_MEDICAL_CHUNK_SIZE = _edef_mod.DEFAULT_MEDICAL_CHUNK_SIZE
DEFAULT_MEDICAL_ENCODER_MODEL = _edef_mod.DEFAULT_MEDICAL_ENCODER_MODEL
DEFAULT_MAX_PROMPT_MEDICAL_TOKENS = _edef_mod.DEFAULT_MAX_PROMPT_MEDICAL_TOKENS
DEFAULT_SIGNAL_SOURCE = _edef_mod.DEFAULT_SIGNAL_SOURCE
attach_edef_to_model = _edef_mod.attach_edef_to_model
build_fused_input_embeddings = _edef_mod.build_fused_input_embeddings
get_edef_config = _edef_mod.get_edef_config
load_edef_checkpoint = _edef_mod.load_edef_checkpoint
load_edef_config = _edef_mod.load_edef_config


def parse_ner_output(text):
    import ast
    import json
    import re

    cleaned = re.sub(r"^assistant\s*", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
        return data
    except json.JSONDecodeError:
        pass

    try:
        data = ast.literal_eval(cleaned)
        if isinstance(data, dict):
            return data
    except (ValueError, SyntaxError):
        pass

    match = re.search(r"\{.*['\"]ner['\"].*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(match.group())
                if isinstance(data, dict):
                    return data
            except (ValueError, SyntaxError):
                pass
    return {"ner": []}


def _build_chat_prompt(tokenizer: Any, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
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
    return f"{instruction}\n\n{text}"


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


def _resolve_runtime_config(
    *,
    model_path: str,
    signal_source: str | None,
    medical_encoder_model: str | None,
    medical_chunk_size: int | None,
    medical_chunk_overlap: int | None,
    max_prompt_medical_tokens: int | None,
) -> dict[str, Any]:
    ckpt_config = load_edef_config(os.path.join(model_path, "edef_checkpoint"))
    resolved_signal_source = signal_source or ckpt_config.get(
        "signal_source", DEFAULT_SIGNAL_SOURCE
    )
    return {
        "signal_source": str(resolved_signal_source),
        "medical_encoder_model_name": medical_encoder_model
        or ckpt_config.get("medical_encoder_model_name", DEFAULT_MEDICAL_ENCODER_MODEL),
        "medical_chunk_size": int(
            medical_chunk_size
            if medical_chunk_size is not None
            else ckpt_config.get("medical_chunk_size", DEFAULT_MEDICAL_CHUNK_SIZE)
        ),
        "medical_chunk_overlap": int(
            medical_chunk_overlap
            if medical_chunk_overlap is not None
            else ckpt_config.get("medical_chunk_overlap", DEFAULT_MEDICAL_CHUNK_OVERLAP)
        ),
        "max_prompt_medical_tokens": int(
            max_prompt_medical_tokens
            if max_prompt_medical_tokens is not None
            else ckpt_config.get(
                "max_prompt_medical_tokens",
                DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
            )
        ),
    }


class EDEFInferencePipeline:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        instruction: str,
        signal_source: str,
        dist_dim: int = 45,
        word_entity_dist: dict[str, list[float]] | None = None,
        default_dist: list[float] | None = None,
        medical_tokenizer: Any | None = None,
        medical_chunk_size: int = DEFAULT_MEDICAL_CHUNK_SIZE,
        medical_chunk_overlap: int = DEFAULT_MEDICAL_CHUNK_OVERLAP,
        max_prompt_medical_tokens: int = DEFAULT_MAX_PROMPT_MEDICAL_TOKENS,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.signal_source = signal_source
        self.dist_dim = dist_dim
        self.word_entity_dist = word_entity_dist or {}
        self.default_dist = default_dist or []
        self.medical_tokenizer = medical_tokenizer
        self.medical_chunk_size = medical_chunk_size
        self.medical_chunk_overlap = medical_chunk_overlap
        self.max_prompt_medical_tokens = max_prompt_medical_tokens
        self.device = next(self.model.parameters()).device
        self._edef_config = get_edef_config(model)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        phase1_model: str,
        dist_path: str | None,
        *,
        instruction: str | None = None,
        signal_source: str | None = None,
        medical_encoder_model: str | None = None,
        medical_chunk_size: int | None = None,
        medical_chunk_overlap: int | None = None,
        max_prompt_medical_tokens: int | None = None,
        base_model: str = "Qwen/Qwen3-4B-Instruct",
        dist_dim: int | None = None,
        hidden_dim: int | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        runtime_config = _resolve_runtime_config(
            model_path=model_path,
            signal_source=signal_source,
            medical_encoder_model=medical_encoder_model,
            medical_chunk_size=medical_chunk_size,
            medical_chunk_overlap=medical_chunk_overlap,
            max_prompt_medical_tokens=max_prompt_medical_tokens,
        )

        task_metadata: dict[str, Any] = {"instruction": instruction or ""}
        effective_dist_dim = dist_dim if dist_dim is not None else 45
        if dist_path:
            task_metadata = load_task_metadata_from_dist_path(dist_path)
            metadata_dist_dim = int(task_metadata["dist_dim"])
            if dist_dim is not None and int(dist_dim) != metadata_dist_dim:
                raise ValueError(
                    f"dist_dim={dist_dim} does not match metadata dist_dim={metadata_dist_dim} in {dist_path}"
                )
            effective_dist_dim = metadata_dist_dim
        if not task_metadata.get("instruction"):
            if not instruction:
                raise ValueError(
                    "Provide dist_path or instruction for prompt construction."
                )
            task_metadata["instruction"] = instruction

        dtype = _resolve_torch_dtype(torch_dtype)
        model_source = phase1_model if phase1_model else base_model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        hidden = (
            hidden_dim
            if hidden_dim is not None
            else getattr(model.config, "hidden_size", 2560)
        )
        model = attach_edef_to_model(
            model,
            dist_dim=effective_dist_dim,
            hidden_dim=hidden,
            signal_source=runtime_config["signal_source"],
            medical_encoder_model_name=runtime_config["medical_encoder_model_name"],
            medical_chunk_size=runtime_config["medical_chunk_size"],
            medical_chunk_overlap=runtime_config["medical_chunk_overlap"],
            max_prompt_medical_tokens=runtime_config["max_prompt_medical_tokens"],
        )
        model = PeftModel.from_pretrained(model, model_path)

        edef_ckpt = os.path.join(model_path, "edef_checkpoint")
        if os.path.isdir(edef_ckpt):
            load_edef_checkpoint(model, edef_ckpt, medical_encoder_trainable=False)

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

        word_entity_dist: dict[str, list[float]] | None = None
        default_dist: list[float] | None = None
        medical_tokenizer_obj = None
        if runtime_config["signal_source"] == "distribution":
            if not dist_path:
                raise ValueError("Distribution mode requires dist_path.")
            word_entity_dist, default_dist = load_distributions(dist_path)
        else:
            medical_tokenizer_obj = AutoTokenizer.from_pretrained(
                runtime_config["medical_encoder_model_name"],
                trust_remote_code=trust_remote_code,
            )

        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            instruction=str(task_metadata["instruction"]),
            signal_source=runtime_config["signal_source"],
            dist_dim=effective_dist_dim,
            word_entity_dist=word_entity_dist,
            default_dist=default_dist,
            medical_tokenizer=medical_tokenizer_obj,
            medical_chunk_size=runtime_config["medical_chunk_size"],
            medical_chunk_overlap=runtime_config["medical_chunk_overlap"],
            max_prompt_medical_tokens=runtime_config["max_prompt_medical_tokens"],
        )

    def _build_signal_inputs(
        self,
        *,
        prompt_text: str,
        clinical_text: str,
    ) -> dict[str, torch.Tensor]:
        if self.signal_source == "distribution":
            dist_vectors = get_token_distributions(
                text=prompt_text,
                tokenizer=self.tokenizer,
                word_entity_dist=self.word_entity_dist,
                default_dist=self.default_dist,
                dist_dim=self.dist_dim,
            ).unsqueeze(0)
            return {"entity_dist_vectors": dist_vectors.to(self.device)}

        if self.medical_tokenizer is None:
            raise ValueError("Medical encoder mode requires a medical tokenizer.")
        alignment = build_medical_alignment_features(
            prompt_text=prompt_text,
            clinical_text=clinical_text,
            prompt_tokenizer=self.tokenizer,
            medical_tokenizer=self.medical_tokenizer,
            prompt_max_length=4096,
            chunk_size=self.medical_chunk_size,
            chunk_overlap=self.medical_chunk_overlap,
            max_prompt_medical_tokens=self.max_prompt_medical_tokens,
        )
        batched: dict[str, torch.Tensor] = {}
        for key, value in alignment.items():
            tensor_value = value.unsqueeze(0) if value.dim() >= 1 else value.reshape(1)
            batched[key] = tensor_value.to(self.device)
        return batched

    def _build_fused_embeddings(
        self,
        *,
        prompt_text: str,
        clinical_text: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        signal_inputs = self._build_signal_inputs(
            prompt_text=prompt_text,
            clinical_text=clinical_text,
        )
        fused_embeds, _ = build_fused_input_embeddings(
            self.model,
            input_ids=input_ids,
            **signal_inputs,
        )
        return input_ids, attention_mask, fused_embeds

    def predict(self, text: str, max_new_tokens: int = 2048, temperature: float = 0.0):
        prompt_text = _build_chat_prompt(self.tokenizer, self.instruction, text)
        with torch.no_grad():
            input_ids, attention_mask, fused_embeds = self._build_fused_embeddings(
                prompt_text=prompt_text,
                clinical_text=text,
            )
            do_sample = temperature > 0
            generate_kwargs: dict[str, Any] = {
                "inputs_embeds": fused_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature
            output_ids = self.model.generate(**generate_kwargs)

        prompt_len = input_ids.shape[1]
        generated_ids = output_ids[0]
        completion_ids = (
            generated_ids[prompt_len:]
            if generated_ids.shape[0] > prompt_len
            else generated_ids
        )
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return parse_ner_output(decoded)

    def predict_batch(
        self, texts, max_new_tokens: int = 2048, temperature: float = 0.0
    ):
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
    parser = argparse.ArgumentParser(description="EDEF-enhanced clinical NER inference")
    parser.add_argument(
        "--model_path", required=True, help="Stage 2 model path (LoRA + EDEF)"
    )
    parser.add_argument(
        "--phase1_model",
        default=DEFAULT_PHASE1_MODEL_PATH,
        help="Phase 1 merged model path",
    )
    parser.add_argument(
        "--base_model", default="Qwen/Qwen3-4B-Instruct", help="Base model name"
    )
    parser.add_argument(
        "--dist_path", default=DEFAULT_DIST_PATH, help="Entity distributions JSON"
    )
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--signal_source", default=None, choices=["distribution", "medical_encoder"]
    )
    parser.add_argument("--medical_encoder_model", default=None)
    parser.add_argument("--medical_chunk_size", type=int, default=None)
    parser.add_argument("--medical_chunk_overlap", type=int, default=None)
    parser.add_argument("--max_prompt_medical_tokens", type=int, default=None)

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
        instruction=args.instruction,
        signal_source=args.signal_source,
        medical_encoder_model=args.medical_encoder_model,
        medical_chunk_size=args.medical_chunk_size,
        medical_chunk_overlap=args.medical_chunk_overlap,
        max_prompt_medical_tokens=args.max_prompt_medical_tokens,
        base_model=args.base_model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.input is not None:
        prediction = pipeline.predict(
            args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_payload: Any = prediction
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
