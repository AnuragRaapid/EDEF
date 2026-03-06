# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportAny=false, reportExplicitAny=false, reportUnusedImport=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportImplicitRelativeImport=false, reportUnannotatedClassAttribute=false, reportMissingParameterType=false, reportUnknownArgumentType=false

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
    _dataset_utils_mod = importlib.import_module(".ner_dataset_utils", package=__package__)
else:
    _dist_mod = importlib.import_module("distribution_alignment")
    _edef_mod = importlib.import_module("edef_model")
    _dataset_utils_mod = importlib.import_module("ner_dataset_utils")

get_token_distributions = _dist_mod.get_token_distributions
load_distributions = _dist_mod.load_distributions
attach_edef_to_model = _edef_mod.attach_edef_to_model
load_edef_checkpoint = _edef_mod.load_edef_checkpoint
DEFAULT_DIST_PATH = _dataset_utils_mod.DEFAULT_DIST_PATH
DEFAULT_PHASE1_MODEL_PATH = _dataset_utils_mod.DEFAULT_PHASE1_MODEL_PATH
load_task_metadata_from_dist_path = _dataset_utils_mod.load_task_metadata_from_dist_path


def parse_ner_output(text):
    """Parse model output to extract NER entities."""
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


def _find_edef_host(model: Any) -> Any:
    if hasattr(model, "entity_projector") and hasattr(model, "fusion_gate"):
        return model
    if hasattr(model, "base_model"):
        found = _find_edef_host(model.base_model)
        if found is not None:
            return found
    if hasattr(model, "model"):
        found = _find_edef_host(model.model)
        if found is not None:
            return found
    return None


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


class EDEFInferencePipeline:
    """EDEF-enhanced NER inference pipeline.

    Usage:
        pipeline = EDEFInferencePipeline.from_pretrained(
            model_path="saves/edef-stage2",
            phase1_model="saves/phase1_merged",
            dist_path="artifacts/ncbi_disease/entity_distributions.json",
        )
        result = pipeline.predict("Familial Mediterranean fever was diagnosed in the patient.")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        word_entity_dist: dict[str, list[float]],
        default_dist: list[float],
        instruction: str,
        dist_dim: int = 45,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.word_entity_dist = word_entity_dist
        self.default_dist = default_dist
        self.instruction = instruction
        self.dist_dim = dist_dim

        host = _find_edef_host(model)
        if host is None:
            raise ValueError("Could not locate attached EDEF modules in model.")
        self._edef_host = host

        self.device = next(self.model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        phase1_model: str,
        dist_path: str,
        base_model: str = "Qwen/Qwen3-4B-Instruct",
        dist_dim: int | None = None,
        hidden_dim: int | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        task_metadata = load_task_metadata_from_dist_path(dist_path)
        metadata_dist_dim = int(task_metadata["dist_dim"])
        if dist_dim is not None and int(dist_dim) != metadata_dist_dim:
            raise ValueError(
                f"dist_dim={dist_dim} does not match metadata dist_dim={metadata_dist_dim} in {dist_path}"
            )
        effective_dist_dim = metadata_dist_dim
        dtype = _resolve_torch_dtype(torch_dtype)

        model_source = phase1_model if phase1_model else base_model
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        hidden = hidden_dim if hidden_dim is not None else getattr(model.config, "hidden_size", 2560)
        model = attach_edef_to_model(model, dist_dim=effective_dist_dim, hidden_dim=hidden)
        model = PeftModel.from_pretrained(model, model_path)

        edef_ckpt = os.path.join(model_path, "edef_checkpoint")
        if os.path.isdir(edef_ckpt):
            host = _find_edef_host(model)
            if host is None:
                raise RuntimeError("EDEF modules are not attached after loading LoRA adapter.")
            load_edef_checkpoint(host, edef_ckpt)

        tokenizer_source = model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else model_source
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        word_entity_dist, default_dist = load_distributions(dist_path)

        model.eval()
        return cls(
            model,
            tokenizer,
            word_entity_dist,
            default_dist,
            instruction=str(task_metadata["instruction"]),
            dist_dim=effective_dist_dim,
        )

    def _build_fused_embeddings(self, prompt_text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        token_embed_layer = self.model.get_input_embeddings()
        token_embeds = token_embed_layer(input_ids)

        dist_vectors = get_token_distributions(
            text=prompt_text,
            tokenizer=self.tokenizer,
            word_entity_dist=self.word_entity_dist,
            default_dist=self.default_dist,
            dist_dim=self.dist_dim,
        )
        dist_vectors = dist_vectors.to(device=token_embeds.device, dtype=token_embeds.dtype)
        dist_vectors = dist_vectors.unsqueeze(0)

        seq_len = token_embeds.shape[1]
        if dist_vectors.shape[1] > seq_len:
            dist_vectors = dist_vectors[:, :seq_len, :]
        elif dist_vectors.shape[1] < seq_len:
            pad_rows = seq_len - dist_vectors.shape[1]
            padding = torch.zeros(
                (1, pad_rows, self.dist_dim),
                device=token_embeds.device,
                dtype=token_embeds.dtype,
            )
            dist_vectors = torch.cat([dist_vectors, padding], dim=1)

        projected = self._edef_host.entity_projector(dist_vectors)
        fused_embeds = self._edef_host.fusion_gate(token_embeds, projected)
        return input_ids, attention_mask, fused_embeds

    def predict(self, text: str, max_new_tokens: int = 2048, temperature: float = 0.0):
        prompt_text = _build_chat_prompt(self.tokenizer, self.instruction, text)

        with torch.no_grad():
            input_ids, attention_mask, fused_embeds = self._build_fused_embeddings(prompt_text)

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
        completion_ids = generated_ids[prompt_len:] if generated_ids.shape[0] > prompt_len else generated_ids
        decoded = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return parse_ner_output(decoded)

    def predict_batch(self, texts, max_new_tokens: int = 2048, temperature: float = 0.0):
        return [self.predict(text, max_new_tokens=max_new_tokens, temperature=temperature) for text in texts]


def _load_texts_from_json(input_file: str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("--input_file JSON must be a list of strings or list of {input: text} objects.")

    texts: list[str] = []
    for item in data:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and "input" in item:
            texts.append(str(item["input"]))
        else:
            raise ValueError("Invalid JSON entry. Expected string or object with an 'input' field.")
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDEF-enhanced clinical NER inference")
    parser.add_argument("--model_path", required=True, help="Stage 2 model path (LoRA + EDEF)")
    parser.add_argument("--phase1_model", default=DEFAULT_PHASE1_MODEL_PATH, help="Phase 1 merged model path")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct", help="Base model name")
    parser.add_argument("--dist_path", default=DEFAULT_DIST_PATH, help="Entity distributions JSON")

    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument("--input", help="Single text to process")
    io_group.add_argument("--input_file", help="JSON file with texts to process")

    parser.add_argument("--output_file", help="Save predictions JSON")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto", help="auto|bfloat16|float16|float32")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
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
