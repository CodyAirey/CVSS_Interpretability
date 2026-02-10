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

# Collapsed mapping used for Table A (kept as-is, but now includes 8)
PRIMARY_MAP = {
    1: "contradict",
    2: "contradict",
    3: "no-evidence",
    4: "no-evidence",
    5: "support",
    6: "support",
    7: "support",
    8: "unclear",
}

# Full 8-point labels for the new figure (matches your table wording)
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

JV_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]


def emit_full_jv_distribution_figure(df: pd.DataFrame) -> None:
    """
    Emits a compact pgfplots bar chart for the full 8-point JV distribution.
    This does not replace any existing tables.
    """
    if "JustificationValue" not in df.columns:
        raise KeyError("Expected column 'JustificationValue' not found in CSV")

    s = pd.to_numeric(df["JustificationValue"], errors="coerce").dropna().astype(int)
    counts = s.value_counts().to_dict()

    # Build coordinates in a stable order
    coords = []
    for k in JV_ORDER:
        coords.append((k, int(counts.get(k, 0))))

    # Short symbolic x-values for pgfplots (avoid spaces in coords)
    sym = {
        1: "SC",
        2: "C",
        3: "LA",
        4: "NE",
        5: "LT",
        6: "S",
        7: "ES",
        8: "U",
    }

    xticklabels = [
        JV_LABELS[1],
        JV_LABELS[2],
        JV_LABELS[3],
        JV_LABELS[4],
        JV_LABELS[5],
        JV_LABELS[6],
        JV_LABELS[7],
        JV_LABELS[8],
    ]

    coord_str = " ".join(f"({sym[k]},{v})" for k, v in coords)

    print("\n% ---- Figure: Full 8-point justification value distribution ----\n")
    print(r"\begin{figure}[t]")
    print(r"    \centering")
    print(r"    \begin{tikzpicture}")
    print(r"    \begin{axis}[")
    print(r"        ybar,")
    print(r"        width=\linewidth,")
    print(r"        height=5.0cm,")
    print(r"        ymin=0,")
    print(r"        ymax=55,")
    print(r"        ymajorgrids=true,")
    print(r"        ylabel={Count},")
    print(r"        symbolic x coords={SC,C,LA,NE,LT,S,ES,U},")
    print(r"        xtick=data,")
    print(r"        xticklabels={%s}," % (",".join(xticklabels)))
    print(r"        xticklabel style={rotate=35, anchor=east, font=\scriptsize},")
    print(r"        nodes near coords,")
    print(r"        nodes near coords align={vertical},")
    print(r"        every node near coord/.append style={font=\scriptsize},")
    print(r"        enlarge x limits=0.05,")
    print(r"        bar width=8pt,")
    print(r"    ]")
    print(r"        \addplot coordinates {%s};" % coord_str)
    print(r"    \end{axis}")
    print(r"    \end{tikzpicture}")
    print(r"    \caption{Distribution of primary justification responses across all CVE evaluations (full 8-point scale).}")
    print(r"    \label{fig:justification_value_distribution_full}")
    print(r"\end{figure}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # normalise headers
    df.columns = [c.strip() for c in df.columns]

    # map primary judgement (collapsed, for Table A only)
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

    print("\n% ---- Table A: Primary judgements ----\n")
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

    # -------- Figure: full 8-point JV distribution --------
    emit_full_jv_distribution_figure(df)


if __name__ == "__main__":
    main()
