#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

LIKERT_COLS = ["Explicitness", "Inference", "Completeness", "Ambiguity", "Contradiction"]

SUPPORT = {5, 6, 7}
CONTRADICT = {1, 2, 3}
NO_EVIDENCE = {4}
UNCLEAR = {8}


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash ")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("^", r"\^{}")
        .replace("~", r"\~{}")
    )


def latex_table(caption: str, label: str, headers: list[str], rows: list[list[str]], colspec: str) -> str:
    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(fr"\caption{{{caption}}}")
    out.append(fr"\label{{{label}}}")
    out.append(r"\small")
    out.append(r"\setlength{\tabcolsep}{6pt}")
    out.append(r"\renewcommand{\arraystretch}{1.1}")
    out.append(fr"\begin{{tabular}}{{{colspec}}}")
    out.append(r"\toprule")
    out.append(" & ".join([fr"\textbf{{{latex_escape(h)}}}" for h in headers]) + r" \\")
    out.append(r"\midrule")
    for r in rows:
        out.append(" & ".join(r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def entropy_of_categories(values: pd.Series) -> float:
    p = values.value_counts(normalize=True, dropna=True)
    ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    # clamp -0.0 and tiny negative rounding artefacts
    if abs(ent) < 1e-12:
        ent = 0.0
    return ent


def rate(values: pd.Series, allowed: set[int]) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.isin(allowed).sum() / len(values))


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse top-N CVEs by response count.")
    ap.add_argument("--csv", required=True, help="Justification CSV export")
    ap.add_argument("--top_n", type=int, default=3, help="Number of CVEs to analyse by highest n")
    ap.add_argument("--out_dir", default="tables/justification", help="Where to write LaTeX output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    required = ["CVEID", "JustificationValue"] + LIKERT_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for c in ["JustificationValue"] + LIKERT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["CVEID", "JustificationValue"])

    # per-CVE summary
    grp = df.groupby("CVEID", dropna=True)

    per_cve = grp.apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "support_rate": rate(g["JustificationValue"], SUPPORT),
                "contradict_rate": rate(g["JustificationValue"], CONTRADICT),
                "no_evidence_rate": rate(g["JustificationValue"], NO_EVIDENCE),
                "unclear_rate": rate(g["JustificationValue"], UNCLEAR),
                "explicit_mean": float(g["Explicitness"].mean()),
                "inference_mean": float(g["Inference"].mean()),
                "completeness_mean": float(g["Completeness"].mean()),
                "ambiguity_mean": float(g["Ambiguity"].mean()),
                "contradiction_mean": float(g["Contradiction"].mean()),
                "disagreement_entropy": entropy_of_categories(g["JustificationValue"]),
                "participants": int(g["QuestionSetID"].nunique()) if "QuestionSetID" in g.columns else int(g.shape[0]),
            }
        )
    ).reset_index()

    # top-N by n (tie-break by participants then CVEID)
    top = per_cve.sort_values(["n", "participants", "CVEID"], ascending=[False, False, True]).head(args.top_n)

    # ---- Console analysis blocks ----
    print(f"Top {args.top_n} CVEs by response count (n):")
    print()

    for _, r in top.iterrows():
        cve = r["CVEID"]
        n = int(r["n"])

        support = r["support_rate"]
        contradict = r["contradict_rate"]
        noe = r["no_evidence_rate"]
        unclear = r["unclear_rate"]

        exp = r["explicit_mean"]
        inf = r["inference_mean"]
        comp = r["completeness_mean"]
        amb = r["ambiguity_mean"]
        contra = r["contradiction_mean"]
        ent = r["disagreement_entropy"]

        # simple flags to make the analysis useful
        infer_heavy = (inf - exp) >= 1.0
        incomplete = comp <= 2.5
        vague = amb >= 3.5
        conflicting = contra >= 3.5
        split = ent >= 1.5  # heuristic, not a claim

        print(f"{cve} (n={n})")
        print(f"  Primary judgement mix: "
              f"support {fmt_pct(support)}, contradict {fmt_pct(contradict)}, "
              f"no-evidence {fmt_pct(noe)}, unclear {fmt_pct(unclear)}")
        print(f"  Likert means: explicit {exp:.2f}, inference {inf:.2f}, completeness {comp:.2f}, "
              f"ambiguity {amb:.2f}, contradiction {contra:.2f}")
        print(f"  Disagreement (entropy): {ent:.2f}")
        flags = []
        if infer_heavy:
            flags.append("inference-heavy")
        if incomplete:
            flags.append("low-completeness")
        if vague:
            flags.append("high-ambiguity")
        if conflicting:
            flags.append("high-contradiction")
        if split:
            flags.append("high-disagreement")
        if flags:
            print(f"  Flags: {', '.join(flags)}")
        print()

    # ---- LaTeX table for top-N ----
    rows = []
    for _, r in top.iterrows():
        rows.append(
            [
                latex_escape(str(r["CVEID"])),
                str(int(r["n"])),
                f"{r['support_rate']:.2f}",
                f"{r['contradict_rate']:.2f}",
                f"{r['unclear_rate']:.2f}",
                f"{r['explicit_mean']:.2f}",
                f"{r['inference_mean']:.2f}",
                f"{r['completeness_mean']:.2f}",
                f"{r['ambiguity_mean']:.2f}",
                f"{r['contradiction_mean']:.2f}",
                f"{r['disagreement_entropy']:.2f}",
            ]
        )

    tex = latex_table(
        caption=f"Top {args.top_n} CVEs by number of justification responses, with primary judgement rates and mean follow-up Likert ratings.",
        label="tab:justification_topn",
        headers=["CVE", "n", "Sup.", "Con.", "Unc.", "Exp.", "Inf.", "Comp.", "Amb.", "Contr.", "Disagr."],
        rows=rows,
        colspec="lrrrrrrrrrr",
    )
    (out_dir / "justification_topn.tex").write_text(tex, encoding="utf-8")

    # Also write a CSV for convenience
    top.to_csv(out_dir / "justification_topn.csv", index=False)

    print(f"Wrote: {out_dir / 'justification_topn.tex'}")
    print(f"Wrote: {out_dir / 'justification_topn.csv'}")


if __name__ == "__main__":
    main()
