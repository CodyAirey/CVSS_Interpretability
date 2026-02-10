#!/usr/bin/env python3
"""
utils/generate_top_tokens_table.py

Emit a LaTeX table of top-K tokens per CVSS metric for each model.
Tokens common to ALL models for a metric are bolded.
If the bolding is due only to surface-form normalisation (not exact in all),
append a small superscript '~' to flag it.

Normalisation rules:
- lowercase
- strip leading '▁' and WordPiece '##'
- drop spaces, hyphens, underscores, punctuation
- (optional) alias map hook for domain synonyms (disabled by default)
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict, Set

CVSS_METRICS = ["av", "ac", "pr", "ui", "s", "c", "i", "a"]

# ---------------------------
# Helpers
# ---------------------------
def parse_model_arg(arg: str) -> Tuple[str, str]:
    if ":" in arg:
        f, l = arg.split(":", 1)
        return f.strip(), l.strip()
    return arg.strip(), arg.strip()

def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)

_ws_dash_underscore = re.compile(r"[ \-_]+")
_non_alnum = re.compile(r"[^a-z0-9]+")

def normalise_token(raw: str) -> str:
    """Conservative normalisation: casefold + strip BPE/WordPiece + remove separators/punct."""
    s = raw
    # Remove common subword markers
    s = s.lstrip("▁")        # SentencePiece
    if s.startswith("##"):   # WordPiece continuation
        s = s[2:]
    s = s.lower()
    # Collapse separators, then strip non-alnum (keeps digits)
    s = _ws_dash_underscore.sub("", s)
    s = _non_alnum.sub("", s)
    return s

# Optional aliasing hook (disabled for correctness)
ALIAS_MAP: Dict[str, str] = {
    # Example (if you decide this is acceptable):
    # "crosssitescripting": "xss",
    # "crosssite": "xss",
}
def canonical_form(raw: str) -> str:
    n = normalise_token(raw)
    return ALIAS_MAP.get(n, n)

def read_top_tokens(csv_path: Path, k: int) -> List[str]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Return LaTeX-escaped display tokens
    return [latex_escape(r["token"]) for r in rows[:k] if r.get("token")]

# ---------------------------
# Table builder
# ---------------------------
def build_table(base: Path, dataset: str, seed: str,
                models: List[Tuple[str, str]], k: int) -> str:
    # grid[model_folder][metric] -> list of escaped display tokens
    grid: Dict[str, Dict[str, List[str]]] = {}
    for model_folder, _ in models:
        per_metric = {}
        for metric in CVSS_METRICS:
            csv_path = base / dataset / seed / model_folder / metric / "analysis" / "tokens_predictive_doc.csv"
            per_metric[metric] = read_top_tokens(csv_path, k)
        grid[model_folder] = per_metric

    # Build intersections
    model_ids = [mf for mf, _ in models]
    exact_common: Dict[str, Set[str]] = {}
    norm_common: Dict[str, Set[str]] = {}

    for metric in CVSS_METRICS:
        # Exact intersection on displayed (escaped) tokens
        sets_exact = [set(grid[mf].get(metric, [])) for mf in model_ids]
        exact = set.intersection(*sets_exact) if sets_exact else set()

        # Normalised intersection on canonical forms; keep canonical strings
        sets_norm = []
        for mf in model_ids:
            toks = grid[mf].get(metric, [])
            toks_canon = {canonical_form(_strip_latex(t)) for t in toks}
            sets_norm.append(toks_canon)
        norm = set.intersection(*sets_norm) if sets_norm else set()

        exact_common[metric] = exact
        norm_common[metric] = norm

    header = (
        "\\begin{table*}[!t]\n"
        "  \\caption{Top "
        + "{k}".format(k=k)
        + " tokens per CVSS base metric (from token-level TP/FP analysis). "
          "Tokens appearing in all models are \\textbf{bold}; "
          "a superscript \\textsuperscript{$\\sim$} indicates a form-variant match (case/segmentation).}\n"
        "  \\label{tab:top_tokens_per_metric}\n"
        "  \\centering\n"
        "  \\footnotesize\n"
        "  \\setlength{\\tabcolsep}{4pt}\n"
        "  \\renewcommand{\\arraystretch}{1.15}\n"
        "  \\begin{tabularx}{\\textwidth}{l*{8}{>{\\raggedright\\arraybackslash}X}}\n"
        "    \\toprule\n"
        "    Model & " + " & ".join(m.upper() for m in CVSS_METRICS) + " \\\\\n"
        "    \\midrule\n"
    )

    body_lines = []
    for idx, (model_folder, model_label) in enumerate(models):
        cells = []
        for metric in CVSS_METRICS:
            toks = grid[model_folder].get(metric, [])
            ex = exact_common[metric]
            nc = norm_common[metric]

            rendered = []
            # For normalised comparison, we need per-token canonical forms
            canon_list = [canonical_form(_strip_latex(t)) for t in toks]

            for t, canon in zip(toks, canon_list):
                if t in ex:
                    # Exact in all models: bold, no marker
                    rendered.append(f"\\textbf{{{t}}}")
                elif canon in nc and len(nc) > 0:
                    # Only via normalisation: bold + ~ marker
                    rendered.append(f"\\textbf{{{t}}}\\textsuperscript{{$\\sim$}}")
                else:
                    rendered.append(t)
            cells.append(", ".join(rendered))
        body_lines.append(f"    {model_label} & " + " & ".join(cells) + " \\\\")
        if idx < len(models) - 1:
            body_lines.append("    \\midrule")

    footer = (
        "    \\bottomrule\n"
        "  \\end{tabularx}\n"
        "\\end{table*}\n"
    )

    return "\n".join([header] + body_lines + [footer])

# Utility: remove LaTeX escaping for canonicalisation checks
# (We keep display escaped, but match on raw-ish content.)
def _strip_latex(s: str) -> str:
    # Quick inverse for the escapes we add; good enough for matching forms
    s = s.replace(r"\textbackslash{}", "\\")
    s = s.replace(r"\&", "&").replace(r"\%", "%").replace(r"\$", "$")
    s = s.replace(r"\#", "#").replace(r"\_", "_").replace(r"\{", "{").replace(r"\}", "}")
    s = s.replace(r"\textasciitilde{}", "~").replace(r"\textasciicircum{}", "^")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="../saved_models")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model folders, optionally with labels as 'folder[:label]'")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    models = [parse_model_arg(m) for m in args.models]
    table = build_table(Path(args.base_dir), args.dataset, str(args.seed), models, args.k)
    print(table)

if __name__ == "__main__":
    main()
