from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_PHASE2_DATA_DIR = Path(
    "/home/anurag/NER/Multi-task Finetuning/Multitask Finetuning Phase 2 Dataset"
)


def resolve_local_first_path(relative_path: str, fallback_path: str | Path) -> Path:
    local_path = PROJECT_ROOT / relative_path
    if local_path.is_file():
        return local_path
    return Path(fallback_path)


def resolve_phase2_split_path(filename: str) -> str:
    return str(
        resolve_local_first_path(f"data/{filename}", LEGACY_PHASE2_DATA_DIR / filename)
    )


def resolve_eval_helper_path() -> Path:
    candidates = [
        PROJECT_ROOT / "evaluate_ner.py",
        PROJECT_ROOT.parent / "Soft Prompt Tuning" / "evaluate_ner.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]
