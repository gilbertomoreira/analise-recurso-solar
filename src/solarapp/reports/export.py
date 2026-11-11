"""CSV/Excel/HTML exports."""

from __future__ import annotations

import html
import io
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV (UTF-8 with BOM, ';' separator, decimal comma) friendly to Brazilian Excel."""
    return df.to_csv(sep=";", decimal=",", encoding="utf-8-sig").encode("utf-8-sig")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Write several DataFrames to one .xlsx (sheet names truncated to 31 chars)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            out = df.copy()
            if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
                out.index = out.index.tz_localize(None)
            out.to_excel(writer, sheet_name=name[:31])
    return buf.getvalue()


def build_html_report(
    title: str,
    inputs: dict[str, str],
    metrics: dict[str, str],
    tables: dict[str, pd.DataFrame],
    figures: list[go.Figure],
    notes: list[str],
) -> str:
    """Self-contained HTML report (Plotly JS from CDN). Print to PDF from the browser."""
    esc = html.escape
    parts = [
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>",
        f"<title>{esc(title)}</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:2rem auto;"
        "padding:0 1rem;color:#222}table{border-collapse:collapse;font-size:.85rem;margin:1rem 0}"
        "td,th{border:1px solid #ccc;padding:.25rem .5rem;text-align:right}th{background:#eee}"
        ".cards{display:flex;flex-wrap:wrap;gap:.6rem}"
        ".card{border:1px solid #ccc;border-radius:6px;"
        "padding:.5rem .9rem;min-width:170px}.card b{display:block;font-size:1.2rem}"
        "@media print{.pagebreak{page-break-before:always}}</style></head><body>",
        f"<h1>{esc(title)}</h1><p>Gerado em {datetime.now():%d/%m/%Y %H:%M}</p>",
        "<h2>Parâmetros</h2><ul>",
        *[f"<li><b>{esc(k)}:</b> {esc(v)}</li>" for k, v in inputs.items()],
        "</ul><h2>Indicadores</h2><div class='cards'>",
        *[f"<div class='card'>{esc(k)}<b>{esc(v)}</b></div>" for k, v in metrics.items()],
        "</div>",
    ]
    for i, fig in enumerate(figures):
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    for name, df in tables.items():
        parts.append(f"<h2>{esc(name)}</h2>" + df.to_html(float_format=lambda x: f"{x:.3f}"))
    parts.append("<h2>Hipóteses e limitações</h2><ul>")
    parts += [f"<li>{esc(n)}</li>" for n in notes]
    parts.append("</ul></body></html>")
    return "\n".join(parts)