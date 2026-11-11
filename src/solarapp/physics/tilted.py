"""Radiation on tilted surfaces (D&B Ch. 1.6, 1.8, 2.15-2.19).

Surface orientation: slope beta in [0, 90+] deg from horizontal; surface azimuth gamma with
0 = south, east negative, west positive. For the southern hemisphere the equator-facing surface
has gamma = 180 deg (north).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from solarapp.config import DEFAULT_ALBEDO
from solarapp.physics.clearness import clearness_index
from solarapp.physics.diffuse import diffuse_fraction
from solarapp.physics.extraterrestrial import h0_monthly_mean_kwh
from solarapp.physics.geometry import (
    DAYS_IN_MONTH,
    MEAN_DAYS,
    ArrayLike,
    cos_zenith,
    declination,
    sunset_hour_angle,
)


def equator_facing_azimuth(lat: float) -> float:
    """Azimuth of the equator-facing direction: 0 (south) in the N hemisphere, 180 in the S."""
    return 0.0 if lat >= 0 else 180.0


def cos_incidence(
    lat: ArrayLike, delta: ArrayLike, beta: ArrayLike, gamma: ArrayLike, omega: ArrayLike
) -> ArrayLike:
    """cos(theta), angle of incidence on a tilted surface, D&B Eq. 1.6.2.

    cos(th) = sin d sin p cos b - sin d cos p sin b cos g + cos d cos p cos b cos w
              + cos d sin p sin b cos g cos w + cos d sin b sin g sin w
    """
    d, p, b, g, w = (np.radians(x) for x in (delta, lat, beta, gamma, omega))
    return (
        np.sin(d) * np.sin(p) * np.cos(b)
        - np.sin(d) * np.cos(p) * np.sin(b) * np.cos(g)
        + np.cos(d) * np.cos(p) * np.cos(b) * np.cos(w)
        + np.cos(d) * np.sin(p) * np.sin(b) * np.cos(g) * np.cos(w)
        + np.cos(d) * np.sin(b) * np.sin(g) * np.sin(w)
    )


def rb_instantaneous(
    lat: float, delta: float, beta: float, gamma: float, omega: ArrayLike
) -> ArrayLike:
    """Geometric factor Rb = cos(theta) / cos(theta_z), D&B Eq. 1.8.1 (0 when sun is down)."""
    cz = cos_zenith(lat, delta, omega)
    ct = np.maximum(cos_incidence(lat, delta, beta, gamma, omega), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cz > 1e-6, ct / cz, 0.0)


def rb_daily(
    lat: float, n: ArrayLike, beta: float, gamma: float | None = None, n_steps: int = 360
) -> ArrayLike:
    """Daily-integrated Rb = integral(cos theta+ dw) / integral(cos theta_z+ dw), D&B Sec. 1.8.

    Numerical (trapezoidal) integration over sunrise-sunset: valid for any azimuth. For
    gamma = 0 (N hemisphere) or 180 (S hemisphere) it matches the closed forms Eq. 1.8.2-1.8.3
    (see rb_monthly_closed_form). NaN during polar night.
    """
    gamma = equator_facing_azimuth(lat) if gamma is None else gamma
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    delta = declination(n_arr)
    ws = sunset_hour_angle(lat, delta)
    frac = np.linspace(-1.0, 1.0, n_steps + 1)
    omega = ws[:, None] * frac[None, :]
    ct = np.maximum(cos_incidence(lat, delta[:, None], beta, gamma, omega), 0.0)
    cz = np.maximum(cos_zenith(lat, delta[:, None], omega), 0.0)
    num, den = np.trapezoid(ct, omega, axis=1), np.trapezoid(cz, omega, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = np.where(den > 1e-9, num / den, np.nan)
    return rb if np.ndim(n) else float(rb[0])


def rb_monthly_closed_form(lat: float, n: float, beta: float) -> float:
    """Closed-form Rb for a surface facing the equator, D&B Eq. 1.8.2 (N) / 1.8.3 (S).

    N hemisphere (gamma = 0): uses (phi - beta); S hemisphere (gamma = 180): (phi + beta).
    Rb = [cos(p') cos d sin ws' + pi ws'/180 sin(p') sin d] /
         [cos p cos d sin ws + pi ws/180 sin p sin d],  ws' = min(ws, arccos(-tan p' tan d)).
    """
    d = float(declination(n))
    ws = float(sunset_hour_angle(lat, d))
    pp = lat - beta if lat >= 0 else lat + beta
    wsp = min(ws, float(sunset_hour_angle(pp, d)))
    r = np.radians
    num = np.cos(r(pp)) * np.cos(r(d)) * np.sin(r(wsp)) + r(wsp) * np.sin(r(pp)) * np.sin(r(d))
    den = np.cos(r(lat)) * np.cos(r(d)) * np.sin(r(ws)) + r(ws) * np.sin(r(lat)) * np.sin(r(d))
    return float(num / den)


def tilted_isotropic(
    h: ArrayLike, hd: ArrayLike, rb: ArrayLike, beta: float, albedo: float = DEFAULT_ALBEDO
) -> ArrayLike:
    """Total radiation on a tilted plane, isotropic sky (Liu & Jordan), D&B Eq. 2.15.1 / 2.19.1.

    HT = (H - Hd) Rb + Hd (1 + cos b)/2 + H rho_g (1 - cos b)/2
    """
    b = np.radians(beta)
    return (h - hd) * rb + hd * (1 + np.cos(b)) / 2 + h * albedo * (1 - np.cos(b)) / 2


def tilted_hay_davies(
    h: ArrayLike,
    hd: ArrayLike,
    h0: ArrayLike,
    rb: ArrayLike,
    beta: float,
    albedo: float = DEFAULT_ALBEDO,
) -> ArrayLike:
    """Total radiation on a tilted plane, Hay-Davies model, D&B Eq. 2.16.7 (anisotropy index).

    Ai = (H - Hd)/H0;  HT = (H - Hd + Ai Hd) Rb + Hd (1 - Ai)(1 + cos b)/2 + H rho_g (1 - cos b)/2
    """
    b = np.radians(beta)
    ai = np.clip(np.asarray((h - hd) / np.where(np.asarray(h0) > 0, h0, np.nan)), 0.0, 1.0)
    return (
        (h - hd + ai * hd) * rb
        + hd * (1 - ai) * (1 + np.cos(b)) / 2
        + h * albedo * (1 - np.cos(b)) / 2
    )


def transpose_daily(
    h: ArrayLike,
    hd: ArrayLike,
    h0: ArrayLike,
    rb: ArrayLike,
    beta: float,
    albedo: float,
    model: str,
) -> ArrayLike:
    """Dispatch 'Isotrópico (Liu-Jordan)' or 'Hay-Davies' transposition."""
    if model.startswith("Iso"):
        return tilted_isotropic(h, hd, rb, beta, albedo)
    if model.startswith("Hay"):
        return tilted_hay_davies(h, hd, h0, rb, beta, albedo)
    raise ValueError(f"Modelo de transposição desconhecido: {model}")


TRANSPOSITION_MODELS = ["Isotrópico (Liu-Jordan)", "Hay-Davies"]


def monthly_tilted(
    lat: float,
    h_monthly: ArrayLike,
    beta: float,
    gamma: float | None = None,
    albedo: float = DEFAULT_ALBEDO,
    model: str = TRANSPOSITION_MODELS[0],
    hd_monthly: ArrayLike | None = None,
    diffuse_model: str = "Erbs",
) -> pd.DataFrame:
    """Monthly mean daily radiation on a tilted plane [kWh/m2/day], Klein method.

    `h_monthly` (and `hd_monthly`) must be in kWh/m2/day, as H0 is computed in those units.

    Uses the mean day of each month (Table 1.6.1). If hd_monthly is None the diffuse part is
    estimated with `diffuse_model` (monthly Erbs by default). Returns DataFrame indexed 1..12
    with columns H, Hd, H0, Kt, Rb, HT.
    """
    h = np.asarray(h_monthly, dtype=float)
    h0 = h0_monthly_mean_kwh(lat)  # kWh/m2/day; scale to input units via ratio below
    kt = clearness_index(h, h0)
    ws = sunset_hour_angle(lat, declination(MEAN_DAYS))
    hd = (
        np.asarray(hd_monthly, dtype=float)
        if hd_monthly is not None
        else h * diffuse_fraction(kt, ws, diffuse_model, monthly=True)
    )
    rb = np.asarray(rb_daily(lat, MEAN_DAYS, beta, gamma))
    ht = transpose_daily(h, hd, h0, rb, beta, albedo, model)
    return pd.DataFrame(
        {"H": h, "Hd": hd, "H0": h0, "Kt": kt, "Rb": rb, "HT": ht},
        index=pd.RangeIndex(1, 13, name="month"),
    )


def annual_mean_daily(monthly_values: ArrayLike) -> float:
    """Annual mean daily value [same unit] weighted by days in each month."""
    v = np.asarray(monthly_values, dtype=float)
    return float(np.nansum(v * DAYS_IN_MONTH) / np.sum(DAYS_IN_MONTH[~np.isnan(v)]))


def tilt_sweep(
    lat: float,
    h_monthly: ArrayLike,
    gamma: float | None = None,
    albedo: float = DEFAULT_ALBEDO,
    model: str = TRANSPOSITION_MODELS[0],
    hd_monthly: ArrayLike | None = None,
    betas: ArrayLike | None = None,
) -> pd.DataFrame:
    """Monthly HT for each slope in `betas` (default 0..90 step 1). Index beta, columns 1..12."""
    betas = np.arange(0, 91, 1.0) if betas is None else np.asarray(betas, dtype=float)
    rows = [
        monthly_tilted(lat, h_monthly, float(b), gamma, albedo, model, hd_monthly)["HT"].to_numpy()
        for b in betas
    ]
    return pd.DataFrame(rows, index=pd.Index(betas, name="beta"), columns=pd.RangeIndex(1, 13))


def optimal_tilts(sweep: pd.DataFrame) -> tuple[float, pd.Series]:
    """Optimal annual slope and optimal slope per month from a tilt_sweep table."""
    annual = sweep.apply(annual_mean_daily, axis=1)
    return float(annual.idxmax()), sweep.idxmax(axis=0)