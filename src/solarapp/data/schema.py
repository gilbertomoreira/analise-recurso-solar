"""Standard column names used internally (units in the comment)."""

# Daily/monthly energy columns are in kWh/m2/day; hourly ones in W/m2 (hourly mean).
NASA_TO_INTERNAL = {
    "ALLSKY_SFC_SW_DWN": "ghi",  # global horizontal irradiation
    "ALLSKY_SFC_SW_DIFF": "dhi",  # diffuse horizontal
    "ALLSKY_SFC_SW_DNI": "dni",  # direct normal
    "ALLSKY_KT": "kt",  # clearness index [-]
    "CLRSKY_SFC_SW_DWN": "ghi_clear",  # clear-sky GHI
    "TOA_SW_DWN": "h0",  # extraterrestrial horizontal
    "T2M": "t2m",  # air temperature at 2 m [degC]
    "WS10M": "ws10m",  # wind speed at 10 m [m/s]
}
ENERGY_COLUMNS = ["ghi", "dhi", "dni", "ghi_clear", "h0"]
LABELS_PT = {
    "ghi": "GHI (kWh/m²/dia)",
    "dhi": "Difusa DHI (kWh/m²/dia)",
    "dni": "DNI (kWh/m²/dia)",
    "kt": "Kt (–)",
    "ghi_clear": "GHI céu claro (kWh/m²/dia)",
    "h0": "H0 topo da atmosfera (kWh/m²/dia)",
    "t2m": "T 2 m (°C)",
    "ws10m": "Vento 10 m (m/s)",
}
