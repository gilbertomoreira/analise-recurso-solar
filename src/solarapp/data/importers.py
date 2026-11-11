"""Import CSV/Excel files (INPE/LABREN Atlas or any user series) with column mapping.

The INPE/LABREN Atlas Brasileiro de Energia Solar is distributed as downloadable files
(CSV/SHP/NetCDF, values in Wh/m2/day); no public API was found, so only file import exists.
"""

from __future__ import annotations

import io
import math
import unicodedata
from typing import BinaryIO

import pandas as pd

MONTH_ALIASES = [
    ("jan", "january", "janeiro"),
    ("feb", "fev", "february", "fevereiro"),
    ("mar", "march", "marco"),
    ("apr", "abr", "april", "abril"),
    ("may", "mai", "maio"),
    ("jun", "june", "junho"),
    ("jul", "july", "julho"),
    ("aug", "ago", "august", "agosto"),
    ("sep", "set", "september", "setembro"),
    ("oct", "out", "october", "outubro"),
    ("nov", "november", "novembro"),
    ("dec", "dez", "december", "dezembro"),
]
# Multiplier that converts the file unit to kWh/m2/day.
UNIT_FACTORS = {"kWh/m²/dia": 1.0, "Wh/m²/dia": 1e-3, "MJ/m²/dia": 1.0 / 3.6}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return text.strip().lower()


def read_table(file: BinaryIO | str, filename: str = "") -> pd.DataFrame:
    """Read CSV (auto separator/decimal) or Excel into a DataFrame."""
    name = (filename or str(getattr(file, "name", file))).lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file)
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()  # noqa: SIM115
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Não foi possível interpretar o CSV: {exc}") from exc
    # Brazilian files often use decimal comma: retry numeric parse of object columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        text_col = df[col].astype(str).str.replace(",", ".", regex=False)
        conv = pd.to_numeric(text_col, errors="coerce")
        if conv.notna().mean() > 0.9:
            df[col] = conv
    return df


def guess_month_columns(columns: list[str]) -> list[str] | None:
    """Find 12 columns named after the months (jan..dec / jan..dez); None if not found."""
    found: list[str] = []
    for aliases in MONTH_ALIASES:
        match = next((c for c in columns if _norm(c).split("_")[0].split(" ")[0] in aliases), None)
        if match is None:
            return None
        found.append(match)
    return found


def wide_to_monthly(
    df: pd.DataFrame, month_columns: list[str], row: int = 0, unit: str = "kWh/m²/dia"
) -> pd.Series:
    """Take one row with 12 month columns -> Series indexed 1..12 in kWh/m2/day."""
    if len(month_columns) != 12:
        raise ValueError("São necessárias exatamente 12 colunas (jan–dez).")
    values = pd.to_numeric(df.iloc[row][month_columns], errors="coerce").to_numpy(float)
    s = pd.Series(values * UNIT_FACTORS[unit], index=range(1, 13), name="ghi")
    s.index.name = "month"
    return s


def nearest_row(df: pd.DataFrame, lat_col: str, lon_col: str, lat: float, lon: float) -> int:
    """Positional index of the row closest to (lat, lon) (planar degrees, adequate for grids)."""
    d2 = (df[lat_col] - lat) ** 2 + ((df[lon_col] - lon) * _cos_deg(lat)) ** 2
    return int(d2.to_numpy().argmin())


def _cos_deg(lat: float) -> float:
    return math.cos(math.radians(lat))


def daily_series(
    df: pd.DataFrame, date_col: str, value_col: str, unit: str = "kWh/m²/dia"
) -> pd.Series:
    """Daily series in kWh/m2/day indexed by date; invalid dates dropped, -999 -> NaN."""
    idx = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    vals = pd.to_numeric(df[value_col], errors="coerce").where(lambda s: s > -990)
    s = pd.Series(vals.to_numpy() * UNIT_FACTORS[unit], index=idx, name="ghi")
    s = s[s.index.notna()].sort_index()
    s.index.name = "date"
    return s