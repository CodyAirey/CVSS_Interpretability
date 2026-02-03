from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _require_svgwrite():
    try:
        import svgwrite  # type: ignore
    except Exception as e:
        raise RuntimeError("svgwrite not installed. Install inside your env: pip install svgwrite") from e
    return svgwrite


def _approx_text_w_px(s: str, font_px: int) -> int:
    return int(math.ceil(len(s) * font_px * 0.56))


def _wrap_text_to_lines(text: str, max_width_px: int, font_px: int) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return [""]

    words = text.split(" ")
    lines: List[str] = []
    cur = ""

    for w in words:
        candidate = w if not cur else (cur + " " + w)
        if _approx_text_w_px(candidate, font_px) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _measure_wrapped_paragraph_height(
    text: str, max_width_px: int, font_px: int, line_h_px: Optional[int] = None
) -> int:
    if line_h_px is None:
        line_h_px = int(font_px * 1.35)
    lines = _wrap_text_to_lines(text, max_width_px, font_px)
    return max(1, len(lines)) * line_h_px


def _draw_wrapped_paragraph(
    dwg,
    x: int,
    y: int,
    text: str,
    max_width_px: int,
    *,
    font_px: int,
    line_h_px: Optional[int] = None,
    fill: str = "#111827",
    font_family: str = "system-ui, -apple-system, Segoe UI, sans-serif",
) -> int:
    if line_h_px is None:
        line_h_px = int(font_px * 1.35)

    lines = _wrap_text_to_lines(text, max_width_px, font_px)
    for i, ln in enumerate(lines):
        dwg.add(
            dwg.text(
                ln,
                insert=(x, y + i * line_h_px),
                font_size=font_px,
                fill=fill,
                font_family=font_family,
            )
        )
    return len(lines) * line_h_px


def _norm_signed(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    m = float(np.max(np.abs(w))) if w.size else 0.0
    return w / m if m > 0 else w


def _norm_nonneg(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    m = float(np.max(w)) if w.size else 0.0
    return w / m if m > 0 else w


def _colour_signed(w: float):
    w = max(min(w, 1.0), -1.0)
    opacity = 0.15 + 0.75 * abs(w)
    if w >= 0:
        return "rgb(255,0,0)", opacity
    return "rgb(0,0,255)", opacity


def _colour_orange(w: float):
    w = max(min(w, 1.0), 0.0)
    opacity = 0.15 + 0.75 * w
    return "rgb(255,165,0)", opacity


def _measure_tokens_height(
    tokens: List[str],
    font_px: int,
    cell_w_px: int,
    *,
    pad_px: int = 10,
    line_gap_px: int = 8,
) -> int:
    if not tokens:
        return pad_px * 2 + int(font_px * 1.35) + 8

    toks = [t.replace("▁", " ") for t in tokens]
    x = pad_px
    max_x = cell_w_px - pad_px
    token_h = int(font_px * 1.35)
    space_px = int(font_px * 0.55)

    lines = 1
    for t in toks:
        if not t.strip():
            continue
        tw = _approx_text_w_px(t, font_px)
        box_w = tw + int(font_px * 0.9)
        if x + box_w > max_x:
            lines += 1
            x = pad_px
        x += box_w + space_px

    return pad_px * 2 + lines * token_h + (lines - 1) * line_gap_px + 8


def _draw_wrapped_tokens(
    dwg,
    x0: int,
    y0: int,
    w_px: int,
    h_px: int,
    tokens: List[str],
    weights: np.ndarray,
    *,
    signed: bool,
    font_px: int,
    pad_px: int = 10,
    line_gap_px: int = 8,
) -> None:
    dwg.add(
        dwg.rect(
            insert=(x0, y0),
            size=(w_px, h_px),
            fill="white",
            stroke="#E5E7EB",
            stroke_width=1,
            rx=8,
            ry=8,
        )
    )

    if not tokens or weights is None or len(tokens) == 0:
        dwg.add(
            dwg.text(
                "(no tokens)",
                insert=(x0 + pad_px, y0 + pad_px + font_px),
                font_size=font_px,
                fill="#6B7280",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )
        return

    toks = [t.replace("▁", " ") for t in tokens]
    w = np.asarray(weights, dtype=float)

    if signed:
        wn = _norm_signed(w)
        colour = _colour_signed
    else:
        wn = _norm_nonneg(w)
        colour = _colour_orange

    x = x0 + pad_px
    y = y0 + pad_px + font_px + 2
    token_h = int(font_px * 1.35)
    space_px = int(font_px * 0.55)
    max_x = x0 + w_px - pad_px

    for t, ww in zip(toks, wn):
        if not t.strip():
            continue

        tw = _approx_text_w_px(t, font_px)
        box_w = tw + int(font_px * 0.9)

        if x + box_w > max_x:
            x = x0 + pad_px
            y += token_h + line_gap_px

        if y > y0 + h_px - pad_px:
            dwg.add(
                dwg.text(
                    "…",
                    insert=(x0 + w_px - pad_px - font_px, y0 + h_px - pad_px),
                    font_size=font_px + 6,
                    fill="#6B7280",
                    font_family="system-ui, -apple-system, Segoe UI, sans-serif",
                )
            )
            break

        fill_col, fill_opacity = colour(float(ww))

        dwg.add(
            dwg.rect(
                insert=(x, y - token_h + 4),
                size=(box_w, token_h),
                rx=4,
                ry=4,
                fill=fill_col,
                fill_opacity=fill_opacity,
                stroke="rgb(0,0,0)",
                stroke_opacity=0.08,
                stroke_width=1,
            )
        )
        dwg.add(
            dwg.text(
                t,
                insert=(x + int(font_px * 0.45), y),
                font_size=font_px,
                fill="#111827",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )
        x += box_w + space_px


def _draw_cell_title(dwg, x: int, y: int, text: str, font_px: int) -> None:
    dwg.add(
        dwg.text(
            text,
            insert=(x, y),
            font_size=font_px,
            fill="#111827",
            font_weight="600",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )


def render_case_svg(
    out_svg: Path,
    *,
    cve_id: str,
    metric: str,
    gt_idx: int,
    gt_label: str,
    description: str,
    model_infos: List[Dict],
    methods: List[str],
    attribs: Dict[Tuple[str, str], Tuple[List[str], np.ndarray, bool]],
    width_px: int,
    font_px: int,
) -> None:
    svgwrite = _require_svgwrite()

    # Layout constants
    margin = 24
    col_header_h = 72
    row_header_w = 150
    row_gap = 16
    col_gap = 16

    title_font = font_px + 14
    desc_font = font_px + 4
    gap1 = 14
    gap3 = 18
    desc_line_h = int(desc_font * 1.35)

    cell_w = int(
        (width_px - (margin * 2) - row_header_w - (len(model_infos) - 1) * col_gap)
        / max(1, len(model_infos))
    )

    desc_max_w = width_px - (2 * margin)
    title_y = margin + title_font
    desc_label_y = title_y + gap1 + desc_font
    desc_text_y = desc_label_y + desc_line_h + 6
    desc_h = _measure_wrapped_paragraph_height(description, desc_max_w, desc_font)
    header_h = (desc_text_y - margin) + desc_h + gap3

    pad_px = 10
    line_gap_px = 8
    max_cell_h = 0
    for mi in model_infos:
        for m in methods:
            key = (mi["label"], m)
            toks = attribs.get(key, ([], np.asarray([], dtype=float), True))[0]
            h = _measure_tokens_height(toks, font_px, cell_w, pad_px=pad_px, line_gap_px=line_gap_px)
            if h > max_cell_h:
                max_cell_h = h
    cell_h = max_cell_h + 10

    height_px = (
        margin * 2
        + header_h
        + col_header_h
        + len(methods) * cell_h
        + (len(methods) - 1) * row_gap
    )

    dwg = svgwrite.Drawing(str(out_svg), size=(width_px, height_px))
    dwg.add(dwg.rect(insert=(0, 0), size=(width_px, height_px), fill="white"))

    x = margin
    dwg.add(
        dwg.text(
            f"{cve_id}  |  Metric {metric.upper()}  |  GT: {gt_label} ({gt_idx})",
            insert=(x, title_y),
            font_size=title_font,
            font_weight="800",
            fill="#111827",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )

    dwg.add(
        dwg.text(
            "Description:",
            insert=(x, desc_label_y),
            font_size=desc_font,
            font_weight="700",
            fill="#111827",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )

    _draw_wrapped_paragraph(
        dwg,
        x=x,
        y=desc_text_y,
        text=description,
        max_width_px=desc_max_w,
        font_px=desc_font,
    )

    top = margin + header_h
    left = margin + row_header_w

    for j, mi in enumerate(model_infos):
        cx = left + j * (cell_w + col_gap)
        cy = top

        dwg.add(
            dwg.rect(
                insert=(cx, cy),
                size=(cell_w, col_header_h),
                fill="white",
                stroke="#E5E7EB",
                stroke_width=1,
                rx=8,
                ry=8,
            )
        )

        _draw_cell_title(dwg, cx + 10, cy + 26, mi["label"], font_px)
        dwg.add(
            dwg.text(
                f"Pred: {mi['pred_label']} ({mi['pred_idx']})  p={mi['prob']:.3f}",
                insert=(cx + 10, cy + 52),
                font_size=font_px - 2,
                fill="#374151",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )

    grid_top = top + col_header_h + 16
    for i, method in enumerate(methods):
        ry = grid_top + i * (cell_h + row_gap)

        dwg.add(
            dwg.rect(
                insert=(margin, ry),
                size=(row_header_w - 16, cell_h),
                fill="white",
                stroke="#E5E7EB",
                stroke_width=1,
                rx=8,
                ry=8,
            )
        )
        dwg.add(
            dwg.text(
                method,
                insert=(margin + 16, ry + 38),
                font_size=font_px + 4,
                font_weight="800",
                fill="#111827",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )

        for j, mi in enumerate(model_infos):
            cx = left + j * (cell_w + col_gap)
            key = (mi["label"], method)

            if key not in attribs:
                dwg.add(
                    dwg.rect(
                        insert=(cx, ry),
                        size=(cell_w, cell_h),
                        fill="white",
                        stroke="#E5E7EB",
                        stroke_width=1,
                        rx=8,
                        ry=8,
                    )
                )
                dwg.add(
                    dwg.text(
                        "(not available)",
                        insert=(cx + 10, ry + 28),
                        font_size=font_px,
                        fill="#6B7280",
                        font_family="system-ui, -apple-system, Segoe UI, sans-serif",
                    )
                )
                continue

            toks, wts, signed = attribs[key]
            _draw_wrapped_tokens(
                dwg,
                cx,
                ry,
                cell_w,
                cell_h,
                toks,
                wts,
                signed=signed,
                font_px=font_px,
                pad_px=pad_px,
                line_gap_px=line_gap_px,
            )

    dwg.save()
