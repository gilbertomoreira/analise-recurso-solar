"""Simple energy indicators (not detailed system simulations)."""

from __future__ import annotations

G_STC_KW_M2 = 1.0  # reference irradiance at STC [kW/m2]


def pv_energy_kwh(ht_kwh_m2: float, p_kwp: float, performance_ratio: float = 0.80) -> float:
    """PV energy E [kWh] = P_kWp * (HT / G_STC) * PR for incident irradiation HT [kWh/m2].

    PR (performance ratio, typically 0.75-0.85) lumps temperature, soiling, wiring and inverter
    losses. With HT in kWh/m2/day the result is kWh/day. Reference: IEC 61724 definition of PR.
    """
    if p_kwp < 0 or not 0 <= performance_ratio <= 1:
        raise ValueError("Potência deve ser >= 0 e PR em [0, 1].")
    return p_kwp * ht_kwh_m2 / G_STC_KW_M2 * performance_ratio


def collector_energy_kwh(ht_kwh_m2: float, area_m2: float, efficiency: float = 0.50) -> float:
    """Thermal energy incident on a collector and converted: Q [kWh] = A * eta * HT.

    `efficiency` is a constant mean thermal efficiency (0-1); Duffie & Beckman Ch. 6 gives the
    Hottel-Whillier-Bliss model for a performance dependent on (Ti - Ta)/GT, not used here.
    """
    if area_m2 < 0 or not 0 <= efficiency <= 1:
        raise ValueError("Área deve ser >= 0 e eficiência em [0, 1].")
    return area_m2 * efficiency * ht_kwh_m2