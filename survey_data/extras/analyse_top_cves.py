#!/usr/bin/env python3

import argparse
import pandas as pd

LIKERT_COLS = {
    "Explicitness": "Explicitness",
    "Inference": "Inference",
    "Completeness": "Completeness",
    "Ambiguity": "Ambiguity",
    "Contradiction": "Contradiction",
}

# 8-point scale (codes 1..8)
JV_LABELS = {
    1: "Strongly contradicts",
    2: "Contradicts",
    3: "Leans against",
    4: "No evidence",
    5: "Leans toward",
    6: "Supports",
    7: "Explicitly supports",
    8: "Unclear",
}

PRIMARY_MAP = {
    1: "contradict",
    2: "contradict",
    3: "contradict",
    4: "no-evidence",
    5: "support",
    6: "support",
    7: "support",
    8: "unclear",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # normalise headers
    df.columns = [c.strip() for c in df.columns]

    if "CVEID" not in df.columns:
        raise KeyError("Expected column 'CVEID' not found in CSV")
    if "JustificationValue" not in df.columns:
        raise KeyError("Expected column 'JustificationValue' not found in CSV")

    # coerce JV to numeric
    df["JustificationValue"] = pd.to_numeric(df["JustificationValue"], errors="coerce")

    # ---- NEW: Top-N CVEs x 8-point JV count table ----
    top_cves = (
        df.dropna(subset=["JustificationValue"])
          .groupby("CVEID")
          .size()
          .sort_values(ascending=False)
          .head(args.top_n)
          .index
          .tolist()
    )

    sub = df[df["CVEID"].isin(top_cves)].copy()
    sub["JV_Label"] = (
        sub["JustificationValue"]
        .dropna()
        .astype(int)
        .map(JV_LABELS)
    )

    # Make sure all 8 columns exist even if zero
    jv_col_order = [JV_LABELS[i] for i in range(1, 9)]

    jv_counts = (
        pd.crosstab(sub["CVEID"], sub["JV_Label"])
          .reindex(index=top_cves, columns=jv_col_order, fill_value=0)
    )

    # Add n for sanity / sorting visibility (optional but useful)
    jv_counts.insert(0, "n", jv_counts.sum(axis=1))

    print("\n% ---- Table: Top CVEs by n (rows) x JV (8-point) counts ----\n")
    print(jv_counts.to_latex(
        index=True,
        caption=f"Counts of primary justification responses (8-point scale) for the top {args.top_n} most-rated CVEs.",
        label="tab:justification_top_cves_by_jv_counts",
        column_format="l" + "r" * (1 + len(jv_col_order)),
        escape=True,
    ))

    # ---- Existing behaviour below (keep your old outputs) ----
    # map primary judgement (3-bin)
    df["primary"] = df["JustificationValue"].map(PRIMARY_MAP)

    rows = []
    for cve, g in df.groupby("CVEID"):
        n = len(g)

        counts = g["primary"].value_counts(normalize=True) * 100
        row = {
            "CVE": cve,
            "n": n,
            "Support\\%": counts.get("support", 0.0),
            "NoEvidence\\%": counts.get("no-evidence", 0.0),
            "Contradict\\%": counts.get("contradict", 0.0),
        }
        for out_name, col in LIKERT_COLS.items():
            row[out_name] = g[col].mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("n", ascending=False).head(args.top_n)

    # -------- Table A: primary judgements --------
    primary = out[["CVE", "n", "Support\\%", "NoEvidence\\%", "Contradict\\%"]]

    print("\n% ---- Table A: Primary judgements (3-bin) ----\n")
    print(primary.to_latex(
        index=False,
        float_format="%.1f",
        caption="Primary justification outcomes for selected CVEs.",
        label="tab:justification_selected_primary",
        column_format="lrrrr",
    ))

    # -------- Table B: Likert means --------
    likert = out[["CVE", "n"] + list(LIKERT_COLS.keys())]

    print("\n% ---- Table B: Likert means ----\n")
    print(likert.to_latex(
        index=False,
        float_format="%.2f",
        caption="Mean follow-up Likert ratings for selected CVEs (with response counts).",
        label="tab:justification_selected_likert",
        column_format="lrrrrrr",
    ))

if __name__ == "__main__":
    main()
