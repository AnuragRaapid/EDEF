from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _baseline_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_baseline.py"),
        "--model_path",
        args.baseline_model_path,
        "--test_data",
        args.test_data,
        "--test_split",
        args.test_split,
        "--dist_path",
        args.dist_path,
        "--output_dir",
        str(output_dir),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    return command


def _edef_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    model_path: str,
    signal_source: str,
    medical_encoder_model: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate_edef.py"),
        "--model_path",
        model_path,
        "--phase1_model",
        args.phase1_model,
        "--test_data",
        args.test_data,
        "--test_split",
        args.test_split,
        "--dist_path",
        args.dist_path,
        "--output_dir",
        str(output_dir),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch_size",
        str(args.batch_size),
        "--signal_source",
        signal_source,
    ]
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    if medical_encoder_model:
        command.extend(["--medical_encoder_model", medical_encoder_model])
    if args.skip_ablation:
        command.append("--no_run_ablation")
    return command


def _extract_summary(result: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "baseline":
        exact = result.get("exact_match", {})
        relaxed = result.get("relaxed_match", {})
        return {
            "exact_f1": float(exact.get("f1", 0.0)) * 100.0,
            "relaxed_f1": float(relaxed.get("f1", 0.0)) * 100.0,
            "avg_inference_time": float(result.get("avg_inference_time", 0.0)),
        }

    return {
        "exact_f1": float(result.get("edef_exact_f1", 0.0)),
        "relaxed_f1": float(result.get("edef_relaxed_f1", 0.0)),
        "avg_inference_time": float(result.get("avg_inference_time", 0.0)),
        "signal_source": result.get("signal_source"),
        "ablation_exact_f1": result.get("ablation_exact_f1"),
        "edef_minus_ablation_f1": result.get("edef_minus_ablation_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the baseline/distribution/medical comparison matrix."
    )
    parser.add_argument("--phase1_model", required=True)
    parser.add_argument("--baseline_model_path", required=True)
    parser.add_argument("--distribution_model_path", required=True)
    parser.add_argument("--medical_frozen_model_path", required=True)
    parser.add_argument("--medical_lora_model_path", required=True)
    parser.add_argument("--pubmed_lora_model_path", default=None)
    parser.add_argument(
        "--medical_encoder_model", default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument(
        "--pubmed_encoder_model",
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--dist_path", required=True)
    parser.add_argument(
        "--output_file", default="evaluation_results/fusion_comparison_matrix.json"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_ablation", action="store_true")
    args = parser.parse_args()

    variants: list[dict[str, Any]] = [
        {
            "name": "baseline_qwen",
            "kind": "baseline",
            "command_builder": lambda out_dir: _baseline_command(args, out_dir),
            "result_file": "baseline_results.json",
        },
        {
            "name": "distribution_edef",
            "kind": "edef",
            "command_builder": lambda out_dir: _edef_command(
                args=args,
                output_dir=out_dir,
                model_path=args.distribution_model_path,
                signal_source="distribution",
                medical_encoder_model=None,
            ),
            "result_file": "edef_results.json",
        },
        {
            "name": "medical_frozen",
            "kind": "edef",
            "command_builder": lambda out_dir: _edef_command(
                args=args,
                output_dir=out_dir,
                model_path=args.medical_frozen_model_path,
                signal_source="medical_encoder",
                medical_encoder_model=args.medical_encoder_model,
            ),
            "result_file": "edef_results.json",
        },
        {
            "name": "medical_lora",
            "kind": "edef",
            "command_builder": lambda out_dir: _edef_command(
                args=args,
                output_dir=out_dir,
                model_path=args.medical_lora_model_path,
                signal_source="medical_encoder",
                medical_encoder_model=args.medical_encoder_model,
            ),
            "result_file": "edef_results.json",
        },
    ]

    if args.pubmed_lora_model_path:
        variants.append(
            {
                "name": "pubmed_lora",
                "kind": "edef",
                "command_builder": lambda out_dir: _edef_command(
                    args=args,
                    output_dir=out_dir,
                    model_path=args.pubmed_lora_model_path,
                    signal_source="medical_encoder",
                    medical_encoder_model=args.pubmed_encoder_model,
                ),
                "result_file": "edef_results.json",
            }
        )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix: dict[str, Any] = {"variants": {}}
    with tempfile.TemporaryDirectory(prefix="fusion-matrix-") as tmpdir:
        tmp_root = Path(tmpdir)
        for variant in variants:
            variant_dir = tmp_root / str(variant["name"])
            variant_dir.mkdir(parents=True, exist_ok=True)
            command = variant["command_builder"](variant_dir)
            print(f"Running {variant['name']}...")
            _run_command(command)
            result = _read_json(variant_dir / str(variant["result_file"]))
            matrix["variants"][str(variant["name"])] = {
                "kind": variant["kind"],
                "result": result,
                "summary": _extract_summary(result, kind=str(variant["kind"])),
            }

    baseline_exact = matrix["variants"]["baseline_qwen"]["summary"]["exact_f1"]
    for name, payload in matrix["variants"].items():
        payload["summary"]["delta_vs_baseline"] = (
            payload["summary"]["exact_f1"] - baseline_exact
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    print(f"Saved fusion comparison matrix to: {output_path}")
    for name, payload in matrix["variants"].items():
        summary = payload["summary"]
        print(
            f"{name}: exact_f1={summary['exact_f1']:.2f}, "
            f"relaxed_f1={summary['relaxed_f1']:.2f}, "
            f"delta_vs_baseline={summary['delta_vs_baseline']:+.2f}"
        )


if __name__ == "__main__":
    main()
