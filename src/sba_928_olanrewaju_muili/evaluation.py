"""Requirement 4: comparing the original and fine-tuned models."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def normalize_answer(answer) -> str:
    """Lowercase, drop thousands separators and collapse whitespace."""
    return " ".join(
        str(answer).lower().replace(",", "").replace("£", "gbp").strip().split()
    )


def is_correct(prediction, expected) -> bool:
    return normalize_answer(prediction) == normalize_answer(expected)


def compare_models(
    examples: Sequence[Dict[str, str]],
    original_predictions: Sequence[str],
    fine_tuned_predictions: Sequence[str],
) -> pd.DataFrame:
    """One row per evaluation prompt with both models' outputs and scores."""
    rows = []
    for example, original, fine_tuned in zip(
        examples, original_predictions, fine_tuned_predictions
    ):
        expected = example["answer"]
        rows.append({
            "scenario": example["scenario"],
            "prompt": example["prompt"],
            "expected": expected,
            "original_output": original,
            "fine_tuned_output": fine_tuned,
            "original_correct": is_correct(original, expected),
            "fine_tuned_correct": is_correct(fine_tuned, expected),
        })
    return pd.DataFrame(rows)


def scenario_accuracy(comparison: pd.DataFrame) -> pd.DataFrame:
    """Exact-match accuracy (%) per research scenario."""
    return (
        comparison.groupby("scenario")[["original_correct", "fine_tuned_correct"]]
        .mean() * 100
    )
