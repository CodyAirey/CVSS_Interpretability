#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
from typing import Dict


# ----------------------------
# JV labels
# ----------------------------

JV_LABELS: Dict[int, str] = {
    1: "Strongly contradicts",
    2: "Contradicts",
    3: "Leans against",
    4: "No evidence",
    5: "Leans toward",
    6: "Supports",
    7: "Explicitly supports",
    8: "Unclear",
}


# ----------------------------
# Helpers
# ----------------------------

def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cve", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()

    sub = df[df["CVEID"] == args.cve].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for {args.cve}")

    sub = sub.reset_index(drop=True)
    sub["PID"] = [f"P{i+1:02d}" for i in range(len(sub))]

    n = len(sub)
    label_cve = args.cve.replace("-", "_").lower()

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(
        rf"\caption{{Individual justification outcomes and follow-up Likert ratings for {args.cve} ($n={n}$).}}"
    )
    print(rf"\label{{tab:{label_cve}_justification_individual}}")
    print(r"\small")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(
        r"\textbf{ID} & "
        r"\textbf{Justification (JV)} & "
        r"\textbf{Explicitness} & "
        r"\textbf{Inference} & "
        r"\textbf{Completeness} & "
        r"\textbf{Ambiguity} & "
        r"\textbf{Contradiction} \\"
    )
    print(r"\midrule")

    for _, r in sub.iterrows():
        jv_val = int(r["JustificationValue"])
        jv_text = f"{JV_LABELS[jv_val]} ({jv_val})"

        print(
            f"{r['PID']} & "
            f"{latex_escape(jv_text)} & "
            f"{int(r['Explicitness'])} & "
            f"{int(r['Inference'])} & "
            f"{int(r['Completeness'])} & "
            f"{int(r['Ambiguity'])} & "
            f"{int(r['Contradiction'])} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
