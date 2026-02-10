#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import pandas as pd


# ----------------------------
# Label definitions
# ----------------------------

age_arr = [
    "18--24",
    "24--35",
    "35--44",
    "45--54",
    "55--64",
    "65 or older",
]

education_arr = [
    "High school or equivalent",
    "Vocational training / certificate",
    "Bachelor’s degree",
    "Master’s degree",
    "Doctorate",
    "Other (Please specify)",
]

confidence_arr = [
    "Not confident",
    "Somewhat confident",
    "Neutral",
    "Confident",
    "Very confident",
]

roles_map: Dict[str, str] = {
    "Roles01": "Security analyst or researcher",
    "Roles02": "Software developer or engineer",
    "Roles03": "Vulnerability coordinator or CNA",
    "Roles04": "Academic",
    "Roles05": "Student",
    "Roles06": "Penetration tester / red team / bug bounty",
    "Roles07": "Infrastructure or operations (DevOps / SysAdmin)",
    "Roles08": "Policy / compliance / governance",
    "Roles09": "Other",
}

versions_map: Dict[str, str] = {
    "Versions01": "CVSS 2.0",
    "Versions02": "CVSS 3.0",
    "Versions03": "CVSS 3.1",
    "Versions04": "CVSS 4.0",
    "Versions05": "None / Not sure",
}

training_map: Dict[str, str] = {
    "Training01": "Yes, CVSS 4.0",
    "Training02": "Yes, CVSS 3.1 / 3.0",
    "Training03": "Yes, CVSS 2.0",
    "Training04": "No",
}


# ----------------------------
# Helpers
# ----------------------------

def latex_escape(s: str) -> str:
    """
    Minimal LaTeX escaping for table cells.
    """
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def code_to_label(series: pd.Series, labels: List[str]) -> pd.Series:
    """
    Map 1-based integer codes to labels.
    """
    def _map(x):
        if pd.isna(x):
            return pd.NA
        try:
            xi = int(x)
        except Exception:
            return pd.NA
        if 1 <= xi <= len(labels):
            return labels[xi - 1]
        return pd.NA

    return series.map(_map)


def single_choice_rows(
    series: pd.Series,
    order: List[str] | None,
    n_total: int,
) -> List[Tuple[str, int, float]]:
    """
    Return (label, count, percent_of_total) rows for a single-choice question.

    NOTE: Percent is computed over n_total (all participants), so tables may not
    sum to 100% if there are missing responses. That is intentional.
    """
    s = series.dropna()
    counts = s.value_counts()

    if order is not None:
        # Keep full order, including zeros
        counts = counts.reindex(order).fillna(0).astype(int)
    else:
        counts = counts.sort_index()

    rows: List[Tuple[str, int, float]] = []
    for label, count in counts.items():
        pct = (int(count) / n_total) * 100.0 if n_total > 0 else 0.0
        rows.append((str(label), int(count), float(pct)))
    return rows


def is_selected(x) -> bool:
    """
    Robust multi-select selection detection.

    Handles:
      - booleans
      - numeric 0/1
      - strings like "0"/"1", "yes"/"no", "true"/"false"
      - option text stored in the cell when selected
    """
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x) != 0.0

    s = str(x).strip()
    if s == "":
        return False

    s_low = s.lower()

    # Common non-selections
    if s_low in {"0", "false", "no", "n", "off", "unchecked", "unselected", "nan", "none"}:
        return False

    # Common selections
    if s_low in {"1", "true", "yes", "y", "on", "checked", "selected"}:
        return True

    # Fallback: many exports store the option text when selected
    return True


def multi_select_count_rows(df: pd.DataFrame, col_map: Dict[str, str]) -> List[Tuple[str, int]]:
    """
    For a multi-select block stored as one column per option (e.g., Roles01..Roles09),
    return counts for each option and a Skip count.
    """
    cols_present = [c for c in col_map.keys() if c in df.columns]
    if not cols_present:
        raise ValueError(f"None of the expected columns exist: {list(col_map.keys())}")

    block = pd.DataFrame({c: df[c].map(is_selected) for c in cols_present})

    rows: List[Tuple[str, int]] = []
    for col in cols_present:
        label = col_map[col]
        rows.append((label, int(block[col].sum())))

    skip_count = int((block.sum(axis=1) == 0).sum())
    # Sort descending by count, keep Skip last
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    rows_sorted.append(("Skip", skip_count))
    return rows_sorted


def print_single_choice_table(
    caption: str,
    label: str,
    rows: List[Tuple[str, int, float]],
) -> None:
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(fr"\caption{{{caption}}}")
    print(fr"\label{{{label}}}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{6pt}")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Count} & \textbf{Percent} \\")
    print(r"\midrule")
    for cat, count, pct in rows:
        cat_e = latex_escape(cat)
        print(f"{cat_e} & {count} & {pct:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def print_multi_select_table_counts_only(
    caption: str,
    label: str,
    rows: List[Tuple[str, int]],
    note: str | None = None,
) -> None:
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(fr"\caption{{{caption}}}")
    print(fr"\label{{{label}}}")
    if note:
        print(fr"\small \emph{{{latex_escape(note)}}}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{6pt}")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"\textbf{Option} & \textbf{Count} \\")
    print(r"\midrule")
    for opt, count in rows:
        opt_e = latex_escape(opt)
        print(f"{opt_e} & {count} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Print LaTeX demographics tables.")
    ap.add_argument("--csv", required=True, help="Demographics CSV exported from the survey platform.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).convert_dtypes()
    df.columns = df.columns.str.strip()
    n = len(df)

    # Single-choice label columns (if present)
    df["AgeLabel"] = code_to_label(df["Age"], age_arr) if "Age" in df.columns else pd.NA
    df["EducationLabel"] = code_to_label(df["Education"], education_arr) if "Education" in df.columns else pd.NA
    df["ConfidenceLabel"] = code_to_label(df["Confidence"], confidence_arr) if "Confidence" in df.columns else pd.NA

    print(f"% n = {n}")
    print()

    # Age
    if "AgeLabel" in df.columns:
        print_single_choice_table(
            caption=fr"Participant age distribution, $n={n}$.",
            label="tab:survey_age",
            rows=single_choice_rows(df["AgeLabel"], age_arr, n),
        )

    # Education
    if "EducationLabel" in df.columns:
        print_single_choice_table(
            caption=fr"Highest completed education, $n={n}$.",
            label="tab:survey_education",
            rows=single_choice_rows(df["EducationLabel"], education_arr, n),
        )

    # Confidence
    if "ConfidenceLabel" in df.columns:
        print_single_choice_table(
            caption=fr"Self-reported confidence reading or writing technical security documentation (English), $n={n}$.",
            label="tab:survey_confidence",
            rows=single_choice_rows(df["ConfidenceLabel"], confidence_arr, n),
        )

    # Roles (multi-select, counts only)
    print_multi_select_table_counts_only(
        caption=fr"Participant roles (multi-select), $n={n}$.",
        label="tab:survey_roles",
        note="Counts are reported because participants could select multiple options or skip the question.",
        rows=multi_select_count_rows(df, roles_map),
    )

    # CVSS versions (multi-select, counts only)
    print_multi_select_table_counts_only(
        caption=fr"Reported CVSS version familiarity (multi-select), $n={n}$.",
        label="tab:survey_cvss_versions",
        note="Counts are reported because participants could select multiple options or skip the question.",
        rows=multi_select_count_rows(df, versions_map),
    )

    # Training (multi-select, counts only)
    print_multi_select_table_counts_only(
        caption=fr"Formal CVSS training (multi-select), $n={n}$.",
        label="tab:survey_cvss_training",
        note="Counts are reported because participants could select multiple options or skip the question.",
        rows=multi_select_count_rows(df, training_map),
    )


if __name__ == "__main__":
    main()
