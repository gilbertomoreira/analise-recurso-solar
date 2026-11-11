"""Glue between data and physics: builds the tables that the UI, plots and reports consume.

All energy columns are kWh/m2/day unless the name says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from solarapp.config import DEFAULT_ALBEDO, KWH_TO_MJ
from solarapp.physics import diffuse, tilted
from solarapp.physics.clearness import clearness_index
from solarapp.physics.extraterrestrial import h0_daily_kwh, h0_monthly_mean_kwh
from solarapp.physics.geometry import (
    DAYS_IN_MONTH,
    MEAN_DAYS,
    day_length,
    declination,
    sunset_hour_angle,
)

MONTHS_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def climatology(monthly: pd.DataFrame) -> pd.DataFrame:
    """Mean over years for each month (index 1..12) of every column of the monthly table."""
    return monthly.groupby(level="month").mean(numeric_only=True).reindex(range(1, 13))


def enrich_monthly(clim: pd.DataFrame, lat: float, diffuse_model: str = "Erbs") -> pd.DataFrame:
    """Add H0 (own computation), Kt, day length and a complete diffuse column.

    `dhi_model` is always computed from the chosen correlation; `dhi` keeps the NASA value and is
    filled with the model only where missing. `dhi_source` flags what was used.
    """
    out = clim.copy()
    out["h0_calc"] = h0_monthly_mean_kwh(lat)
    out["kt_calc"] = clearness_index(out["ghi"], out["h0_calc"])
    ws = sunset_hour_angle(lat, declination(MEAN_DAYS))
    out["day_length_h"] = day_length(lat, MEAN_DAYS)
    out["dhi_model"] = out["ghi"] * diffuse.diffuse_fraction(
        out["kt_calc"].to_numpy(), ws, diffuse_model, monthly=True
    )
    if "dhi" not in out:
        out["dhi"] = np.nan
    out["dhi_source"] = np.where(out["dhi"].isna(), "correlação", "NASA")
    out["dhi"] = out["dhi"].fillna(out["dhi_model"])
    return out


def diffuse_comparison_monthly(enriched: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Diffuse fraction Hd/H: NASA data vs the three correlations (monthly)."""
    ws = sunset_hour_angle(lat, declination(MEAN_DAYS))
    kt = enriched["kt_calc"].to_numpy()
    table = {"NASA": (enriched["dhi"] / enriched["ghi"]).where(enriched["dhi_source"] == "NASA")}
    for name in diffuse.MONTHLY_MODELS:
        table[name] = pd.Series(
            diffuse.diffuse_fraction(kt, ws, name, monthly=True), index=enriched.index
        )
    return pd.DataFrame(table)


def enrich_daily(daily: pd.DataFrame, lat: float) -> pd.DataFrame:
    """Add H0, Kt and diffuse fractions from the 3 daily correlations to a daily table."""
    out = daily.copy()
    n = out.index.day_of_year.to_numpy()
    n = np.where((out.index.month == 2) & (out.index.day == 29), 59, n)
    n = np.where(out.index.is_leap_year & (out.index.month > 2), n - 1, n)
    out["n"] = n
    out["h0_calc"] = h0_daily_kwh(lat, n)
    out["kt_calc"] = clearness_index(out["ghi"], out["h0_calc"])
    ws = sunset_hour_angle(lat, declination(n))
    for name in diffuse.DAILY_MODELS:
        out[f"fd_{name}"] = diffuse.diffuse_fraction(out["kt_calc"].to_numpy(), ws, name, False)
    if "dhi" in out:
        out["fd_NASA"] = out["dhi"] / out["ghi"]
    return out


@dataclass(frozen=True)
class Surface:
    """Tilted surface definition. azimuth: 0 = south, east negative, west positive [deg]."""

    label: str
    beta: float
    azimuth: float


def surfaces_for_comparison(lat: float, user_beta: float, user_azimuth: float, beta_opt: float):
    """Horizontal, tilt = |latitude|, annual optimum and user-defined surfaces."""
    eq = tilted.equator_facing_azimuth(lat)
    return [
        Surface("Horizontal", 0.0, eq),
        Surface(f"Inclinação = latitude ({abs(lat):.0f}°)", min(abs(lat), 90.0), eq),
        Surface(f"Ótima anual ({beta_opt:.0f}°)", beta_opt, eq),
        Surface(f"Definida ({user_beta:.0f}°, γ={user_azimuth:.0f}°)", user_beta, user_azimuth),
    ]


def compare_surfaces(
    lat: float, enriched: pd.DataFrame, surfaces: list[Surface], albedo: float, model: str
) -> pd.DataFrame:
    """Monthly HT [kWh/m2/day] for each surface (columns), index 1..12."""
    cols = {}
    for s in surfaces:
        res = tilted.monthly_tilted(
            lat, enriched["ghi"], s.beta, s.azimuth, albedo, model, hd_monthly=enriched["dhi"]
        )
        cols[s.label] = res["HT"]
    return pd.DataFrame(cols)


def optimal_tilt_info(
    lat: float, enriched: pd.DataFrame, azimuth: float | None, albedo: float, model: str
) -> tuple[float, pd.Series, pd.DataFrame]:
    """Annual optimal tilt, optimal tilt per month and the sweep table."""
    sweep = tilted.tilt_sweep(
        lat, enriched["ghi"], azimuth, albedo, model, hd_monthly=enriched["dhi"]
    )
    best, monthly_best = tilted.optimal_tilts(sweep)
    return best, monthly_best, sweep


def key_metrics(
    enriched: pd.DataFrame, beta_opt: float, ht_opt_daily: float
) -> dict[str, float | str]:
    """Headline metrics shown on the summary cards."""
    ghi = enriched["ghi"]
    ghi_mean = tilted.annual_mean_daily(ghi)
    kt = tilted.annual_mean_daily(enriched["kt_calc"])
    return {
        "ghi_mean_kwh": ghi_mean,
        "ghi_mean_mj": ghi_mean * KWH_TO_MJ,
        "ghi_year_kwh": ghi_mean * 365.0,
        "best_month": MONTHS_PT[int(ghi.idxmax()) - 1],
        "best_value": float(ghi.max()),
        "worst_month": MONTHS_PT[int(ghi.idxmin()) - 1],
        "worst_value": float(ghi.min()),
        "kt_mean": kt,
        "beta_opt": beta_opt,
        "ht_opt_mean_kwh": ht_opt_daily,
        "sun_hours_mean": float(np.sum(enriched["day_length_h"] * DAYS_IN_MONTH) / 365.0),
    }


__all__ = [
    "DEFAULT_ALBEDO",
    "MONTHS_PT",
    "Surface",
    "climatology",
    "compare_surfaces",
    "diffuse_comparison_monthly",
    "enrich_daily",
    "enrich_monthly",
    "key_metrics",
    "optimal_tilt_info",
    "surfaces_for_comparison",
]