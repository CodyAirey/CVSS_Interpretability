#!/usr/bin/env python3
"""
Compute aggregate Likert statistics across all justification responses,
and also output the primary Justification Value (JV) distribution table.

- Strips whitespace from CSV headers before processing.
- Outputs mean, standard deviation, and n for each Likert dimension.
- Outputs JV counts mapped from numeric codes to textual labels.

Notes:
- By default, this script looks for a JV column called "JV".
  If yours is named differently, pass --jv_col.
- JV mapping assumes:
    1 Strongly contradicts
    2 Contradicts
    3 Leans against
    4 No evidence
    5 Leans toward
    6 Supports
    7 Explicitly supports
    8 Unclear
"""

from __future__ import annotations

import argparse
import pandas as pd

LIKERT_COLUMNS = {
    "Explicitness": "Explicitness",
    "Inference": "Inference",
    "Sufficiency of information": "Completeness",
    "Vague wording": "Ambiguity",
    "Conflicting writing": "Contradiction",
}

JV_MAP = {
    1: "Strongly contradicts",
    2: "Contradicts",
    3: "Leans against",
    4: "No evidence",
    5: "Leans toward",
    6: "Supports",
    7: "Explicitly supports",
    8: "Unclear",
}

JV_ORDER = [JV_MAP[i] for i in range(1, 9)]


def _coerce_int_series(s: pd.Series) -> pd.Series:
    """
    Convert a series to Int (nullable), coercing non-numeric to NaN.
    """
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def build_likert_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, col in LIKERT_COLUMNS.items():
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in CSV")

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append(
            {
                "Measure": label,
                "Mean": series.mean(),
                "Std": series.std(ddof=1),
                "n": int(series.count()),
            }
        )
    return pd.DataFrame(rows)


def build_jv_table(df: pd.DataFrame, jv_col: str) -> pd.DataFrame:
    if jv_col not in df.columns:
        raise KeyError(
            f"Expected JV column '{jv_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    jv = _coerce_int_series(df[jv_col])

    # Keep only expected range
    jv_valid = jv.where(jv.between(1, 8))

    labels = jv_valid.map(JV_MAP)
    counts = labels.value_counts(dropna=False)

    rows = []
    for label in JV_ORDER:
        rows.append({"Justification value": label, "Count": int(counts.get(label, 0))})

    missing = int(labels.isna().sum())
    if missing:
        rows.append({"Justification value": "Missing/invalid", "Count": missing})

    return pd.DataFrame(rows)


def main(csv_path: str, jv_col: str):
    df = pd.read_csv(csv_path)

    # ---- Normalise column names ----
    df.columns = [c.strip() for c in df.columns]

    # ---- Likert aggregate table ----
    likert_out = build_likert_table(df)

    print("\nAggregate Likert statistics (all CVEs):\n")
    print(
        likert_out.to_string(
            index=False,
            formatters={"Mean": "{:.2f}".format, "Std": "{:.2f}".format},
        )
    )

    print("\nLaTeX table (Likert):\n")
    print(
        likert_out.to_latex(
            index=False,
            float_format="%.2f",
            caption="Aggregate follow-up Likert ratings across all CVE justification responses.",
            label="tab:justification_likert_overall",
            column_format="lrrr",
            escape=True,
        )
    )

    # ---- JV distribution table ----
    jv_out = build_jv_table(df, jv_col=jv_col)

    print("\nPrimary justification value (JV) distribution:\n")
    print(jv_out.to_string(index=False))

    print("\nLaTeX table (JV distribution):\n")
    print(
        jv_out.to_latex(
            index=False,
            caption="Distribution of primary justification responses across all CVE evaluations.",
            label="tab:justification_value_distribution",
            column_format="lr",
            escape=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to justification responses CSV",
    )
    parser.add_argument(
        "--jv_col",
        default="JustificationValue",
        help="Column name for the primary justification value (numeric 1-8). Default: JV",
    )
    args = parser.parse_args()
    main(args.csv, args.jv_col)
