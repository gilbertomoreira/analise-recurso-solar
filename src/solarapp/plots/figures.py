"""Plotly figures (dark theme). Units are always shown in axis titles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from solarapp.analysis import MONTHS_PT

TEMPLATE = "plotly_dark"
COLORS = {"ghi": "#F5B041", "dni": "#E74C3C", "dhi": "#5DADE2", "kt": "#58D68D", "ref": "#AAB7B8"}
_LAYOUT = {"template": TEMPLATE, "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
           "legend": {"orientation": "h", "y": -0.2}}


def _fig(title: str, ytitle: str, xtitle: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, yaxis_title=ytitle, xaxis_title=xtitle, **_LAYOUT)
    return fig


def monthly_irradiation(enriched: pd.DataFrame) -> go.Figure:
    fig = _fig("Irradiação diária média mensal", "kWh/m²/dia")
    for key, name in (("ghi", "GHI"), ("dni", "DNI"), ("dhi", "DHI (difusa)")):
        if key in enriched and enriched[key].notna().any():
            fig.add_bar(x=MONTHS_PT, y=enriched[key], name=name, marker_color=COLORS[key])
    fig.update_layout(barmode="group")
    return fig


def monthly_kt(enriched: pd.DataFrame) -> go.Figure:
    fig = _fig("Índice de claridade Kt mensal", "Kt (–)")
    fig.add_scatter(x=MONTHS_PT, y=enriched["kt_calc"], name="Kt = H/H0 (calculado)",
                    mode="lines+markers", line={"color": COLORS["kt"]})
    if "kt" in enriched:
        fig.add_scatter(x=MONTHS_PT, y=enriched["kt"], name="Kt NASA", mode="lines+markers",
                        line={"dash": "dot", "color": COLORS["ref"]})
    fig.update_yaxes(range=[0, 1])
    return fig


def diffuse_fraction_comparison(table: pd.DataFrame, title: str, x=None) -> go.Figure:
    fig = _fig(title, "Hd/H (–)")
    x = MONTHS_PT if x is None else x
    for col in table.columns:
        fig.add_scatter(x=x, y=table[col], name=col, mode="lines+markers",
                        line={"width": 3 if col == "NASA" else 1.5})
    return fig


def tilt_comparison(ht: pd.DataFrame) -> go.Figure:
    fig = _fig("Irradiação no plano: comparativo de inclinações", "kWh/m²/dia")
    for col in ht.columns:
        fig.add_scatter(x=MONTHS_PT, y=ht[col], name=col, mode="lines+markers")
    return fig


def tilt_sweep_curve(sweep: pd.DataFrame, annual: pd.Series, beta_opt: float) -> go.Figure:
    fig = _fig("Média anual vs. inclinação", "HT média (kWh/m²/dia)", "Inclinação β (°)")
    fig.add_scatter(x=annual.index, y=annual.values, name="HT anual", line={"color": COLORS["ghi"]})
    fig.add_vline(x=beta_opt, line_dash="dash", annotation_text=f"ótima = {beta_opt:.0f}°")
    return fig


def optimal_tilt_by_month(monthly_best: pd.Series) -> go.Figure:
    fig = _fig("Inclinação ótima mensal", "β ótimo (°)")
    fig.add_bar(x=MONTHS_PT, y=monthly_best.values, marker_color=COLORS["kt"])
    return fig


def daily_series(daily: pd.DataFrame) -> go.Figure:
    fig = _fig("Série diária", "kWh/m²/dia")
    fig.add_scatter(x=daily.index, y=daily["ghi"], name="GHI", mode="lines",
                    line={"color": COLORS["ghi"], "width": 1})
    if "ghi_clear" in daily:
        fig.add_scatter(x=daily.index, y=daily["ghi_clear"], name="GHI céu claro", mode="lines",
                        line={"color": COLORS["ref"], "width": 1, "dash": "dot"})
    fig.add_scatter(x=daily.index, y=daily["h0_calc"], name="H0", mode="lines",
                    line={"color": "#7F8C8D", "width": 1})
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def kt_histogram(daily: pd.DataFrame) -> go.Figure:
    fig = _fig("Distribuição do Kt diário", "Nº de dias", "Kt (–)")
    fig.add_histogram(x=daily["kt_calc"].dropna(), nbinsx=40, marker_color=COLORS["kt"])
    return fig


def hourly_profile(df: pd.DataFrame, g0: np.ndarray) -> go.Figure:
    fig = _fig("Perfil horário (hora solar local)", "W/m²", "Hora solar")
    x = df.index.hour
    for key, name in (("ghi", "GHI"), ("dni", "DNI"), ("dhi", "DHI")):
        if key in df:
            fig.add_scatter(x=x, y=df[key], name=name, line={"color": COLORS[key]})
    fig.add_scatter(x=x, y=g0, name="G0 (extraterrestre)",
                    line={"color": COLORS["ref"], "dash": "dot"})
    return fig


def sun_path_cartesian(table: pd.DataFrame, lines: pd.DataFrame) -> go.Figure:
    """Cylindrical sun chart: altitude vs azimuth (0 = N, clockwise)."""
    fig = _fig("Carta solar cilíndrica", "Altura solar (°)", "Azimute (° a partir do Norte)")
    for _, grp in lines.groupby("solar_hour"):
        fig.add_scatter(x=grp["azimuth_north_deg"], y=grp["altitude_deg"], mode="lines",
                        line={"color": "#566573", "width": 1}, showlegend=False,
                        hovertext=f"{int(grp['solar_hour'].iloc[0])} h solar")
    for label, grp in table.groupby("day_label", sort=False):
        fig.add_scatter(x=grp["azimuth_north_deg"], y=grp["altitude_deg"], mode="lines", name=label)
    fig.update_xaxes(range=[0, 360], dtick=45)
    fig.update_yaxes(range=[0, 90])
    return fig


def sun_path_polar(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for label, grp in table.groupby("day_label", sort=False):
        fig.add_scatterpolar(theta=grp["azimuth_north_deg"], r=90 - grp["altitude_deg"],
                             mode="lines", name=label)
    fig.update_layout(
        title="Carta solar polar (raio = ângulo zenital)", template=TEMPLATE,
        polar={"angularaxis": {"direction": "clockwise", "rotation": 90},
               "radialaxis": {"range": [0, 90], "ticksuffix": "°"}},
        margin=_LAYOUT["margin"], legend=_LAYOUT["legend"],
    )
    return fig


def location_map(lat: float, lon: float, label: str, zoom: float = 5) -> go.Figure:
    fig = go.Figure(go.Scattermap(lat=[lat], lon=[lon], mode="markers+text", text=[label],
                                  textposition="top right",
                                  marker={"size": 14, "color": "#E74C3C"}))
    fig.update_layout(map={"style": "open-street-map", "center": {"lat": lat, "lon": lon},
                           "zoom": zoom}, template=TEMPLATE, height=360,
                      margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig