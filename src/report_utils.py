"""LaTeX formatting helpers for result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def latex_float(x: float, digits: int = 4) -> str:
    """Format a number for LaTeX tables: exponential for very small/large."""
    if pd.isna(x):
        return ""
    if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def write_latex_table(
    df: pd.DataFrame, path: Path | str, caption: str | None = None
) -> None:
    """Write a DataFrame to a LaTeX file with human-readable column names."""
    table = df.copy()
    rename = {
        "dataset": "data",
        "parameter": "param.",
        "method": "method",
        "mean_estimate": "mean",
        "numerical_sd": "sd across runs",
        "rmse_to_reference": "rmse/ref.",
        "variance_reduction": "var. red.",
        "mean_selected": "avg. selected",
    }
    cols = [c for c in rename if c in table.columns]
    table = table[cols].rename(columns=rename)
    for col in ["mean", "sd across runs", "rmse/ref.", "var. red.", "avg. selected"]:
        if col in table.columns:
            table[col] = table[col].map(lambda x: latex_float(x, 4))
    latex = table.to_latex(index=False, escape=True)
    Path(path).write_text(latex, encoding="utf-8")


def write_compact_latex_table(df: pd.DataFrame, path: Path | str) -> None:
    """Write a DataFrame to a LaTeX file, formatting all numeric columns."""
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda x: latex_float(x, 4))
    Path(path).write_text(table.to_latex(index=False, escape=True), encoding="utf-8")
