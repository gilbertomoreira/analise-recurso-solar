"""Diffuse fraction correlations Hd/H = f(Kt).

References: Erbs, Klein & Duffie (1982); Orgill & Hollands (1977);
Collares-Pereira & Rabl (1979); all discussed in D&B Sec. 2.10 and 2.12.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from solarapp.physics.geometry import ArrayLike

OMEGA_S_LIMIT_DEG = 81.4  # D&B: winter/summer split of the Erbs correlations


def _clip01(x: ArrayLike) -> ArrayLike:
    return np.clip(x, 0.0, 1.0)


def erbs_daily(kt: ArrayLike, omega_s: ArrayLike) -> ArrayLike:
    """Daily diffuse fraction Hd/H, Erbs et al. (1982), D&B Eq. 2.10.1.

    omega_s <= 81.4 deg: 1 - 0.2727 Kt + 2.4495 Kt^2 - 11.9514 Kt^3 + 9.3879 Kt^4  (Kt < 0.715)
                          0.143                                                    (Kt >= 0.715)
    omega_s >  81.4 deg: 1 + 0.2832 Kt - 2.5557 Kt^2 + 0.8448 Kt^3                 (Kt < 0.722)
                          0.175                                                    (Kt >= 0.722)
    """
    kt = np.asarray(kt, dtype=float)
    summer = np.asarray(omega_s, dtype=float) <= OMEGA_S_LIMIT_DEG
    a = np.where(
        kt < 0.715,
        1 - 0.2727 * kt + 2.4495 * kt**2 - 11.9514 * kt**3 + 9.3879 * kt**4,
        0.143,
    )
    b = np.where(kt < 0.722, 1 + 0.2832 * kt - 2.5557 * kt**2 + 0.8448 * kt**3, 0.175)
    return _clip01(np.where(summer, a, b))


def orgill_hollands(kt: ArrayLike) -> ArrayLike:
    """Diffuse fraction, Orgill & Hollands (1977).

    1.0 - 0.249 Kt (Kt < 0.35); 1.557 - 1.84 Kt (0.35 <= Kt < 0.75); 0.177 (Kt >= 0.75).
    Originally derived for hourly data; applied here to daily Kt as a common approximation.
    """
    kt = np.asarray(kt, dtype=float)
    mid = np.where(kt < 0.75, 1.557 - 1.84 * kt, 0.177)
    return _clip01(np.where(kt < 0.35, 1 - 0.249 * kt, mid))


def collares_pereira_rabl_daily(kt: ArrayLike) -> ArrayLike:
    """Daily diffuse fraction, Collares-Pereira & Rabl (1979).

    0.99 (Kt < 0.17); 1.188 - 2.272 Kt + 9.473 Kt^2 - 21.856 Kt^3 + 14.648 Kt^4 (0.17-0.75);
    -0.54 Kt + 0.632 (0.75-0.80); 0.2 (Kt >= 0.80).
    """
    kt = np.asarray(kt, dtype=float)
    mid = 1.188 - 2.272 * kt + 9.473 * kt**2 - 21.856 * kt**3 + 14.648 * kt**4
    high = np.where(kt < 0.80, -0.54 * kt + 0.632, 0.2)
    return _clip01(np.where(kt < 0.17, 0.99, np.where(kt < 0.75, mid, high)))


def erbs_monthly(kt_mean: ArrayLike, omega_s: ArrayLike) -> ArrayLike:
    """Monthly-mean diffuse fraction, Erbs et al. (1982), D&B Eq. 2.12.1 (0.3 <= Kt <= 0.8).

    omega_s <= 81.4: 1.391 - 3.560 K + 4.189 K^2 - 2.137 K^3
    omega_s >  81.4: 1.311 - 3.022 K + 3.427 K^2 - 1.821 K^3
    Outside 0.3-0.8 the correlation is extrapolated with Kt clipped to that range.
    """
    k = np.clip(np.asarray(kt_mean, dtype=float), 0.3, 0.8)
    summer = np.asarray(omega_s, dtype=float) <= OMEGA_S_LIMIT_DEG
    a = 1.391 - 3.560 * k + 4.189 * k**2 - 2.137 * k**3
    b = 1.311 - 3.022 * k + 3.427 * k**2 - 1.821 * k**3
    return _clip01(np.where(summer, a, b))


def collares_pereira_rabl_monthly(kt_mean: ArrayLike, omega_s: ArrayLike) -> ArrayLike:
    """Monthly-mean diffuse fraction, Collares-Pereira & Rabl (1979), D&B Eq. 2.12.2.

    Hd/H = 0.775 + 0.00606 (ws - 90) - [0.505 + 0.00455 (ws - 90)] cos(115 K - 103)
    (K in degrees argument), valid for 0.3 <= K <= 0.8 (clipped outside).
    """
    k = np.clip(np.asarray(kt_mean, dtype=float), 0.3, 0.8)
    ws = np.asarray(omega_s, dtype=float)
    return _clip01(
        0.775
        + 0.00606 * (ws - 90.0)
        - (0.505 + 0.00455 * (ws - 90.0)) * np.cos(np.radians(115.0 * k - 103.0))
    )


DAILY_MODELS: dict[str, Callable[..., ArrayLike]] = {
    "Erbs": lambda kt, ws: erbs_daily(kt, ws),
    "Orgill-Hollands": lambda kt, ws: orgill_hollands(kt),
    "Collares-Pereira–Rabl": lambda kt, ws: collares_pereira_rabl_daily(kt),
}
MONTHLY_MODELS: dict[str, Callable[..., ArrayLike]] = {
    "Erbs": erbs_monthly,
    "Orgill-Hollands": lambda kt, ws: orgill_hollands(kt),
    "Collares-Pereira–Rabl": collares_pereira_rabl_monthly,
}


def diffuse_fraction(kt: ArrayLike, omega_s: ArrayLike, model: str, monthly: bool) -> ArrayLike:
    """Dispatch to a named correlation (keys of DAILY_MODELS / MONTHLY_MODELS)."""
    table = MONTHLY_MODELS if monthly else DAILY_MODELS
    if model not in table:
        raise ValueError(f"Modelo desconhecido: {model}. Opções: {list(table)}")
    return table[model](kt, omega_s)