"""Streamlit app: solar resource analysis. Run: streamlit run src/solarapp/app.py"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import folium  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit_folium import st_folium  # noqa: E402

from solarapp import analysis as an  # noqa: E402
from solarapp.config import DEFAULT_ALBEDO, KWH_TO_MJ  # noqa: E402
from solarapp.data import importers  # noqa: E402
from solarapp.data.geocoding import GeocodingError, search_places  # noqa: E402
from solarapp.data.nasa_power import NasaPowerClient, NasaPowerError  # noqa: E402
from solarapp.notes import LIMITATIONS  # noqa: E402
from solarapp.physics import indicators, sunpath, tilted  # noqa: E402
from solarapp.physics.diffuse import MONTHLY_MODELS  # noqa: E402
from solarapp.physics.extraterrestrial import g0_horizontal  # noqa: E402
from solarapp.physics.geometry import (  # noqa: E402
    DAYS_IN_MONTH,
    MEAN_DAYS,
    declination,
    solar_altitude,
    solar_azimuth,
)
from solarapp.plots import figures as fg  # noqa: E402
from solarapp.reports import export  # noqa: E402

st.set_page_config(page_title="Recurso solar", page_icon="☀️", layout="wide")

ENERGY_COLS = {"ghi", "dhi", "dni", "ghi_clear", "h0", "h0_calc", "dhi_model"}
DEFAULT_PLACE = ("João Pessoa, Paraíba, Brasil", -7.115, -34.861)


# ------------------------------------------------------------------ cached data access
@st.cache_data(show_spinner="Consultando NASA POWER (mensal)…", ttl=3600)
def load_monthly(lat: float, lon: float, y0: int, y1: int):
    return NasaPowerClient().fetch_monthly(lat, lon, y0, y1)


@st.cache_data(show_spinner="Consultando NASA POWER (diário)…", ttl=3600)
def load_daily(lat: float, lon: float, y0: int, y1: int) -> pd.DataFrame:
    return NasaPowerClient().fetch_daily(lat, lon, date(y0, 1, 1), date(y1, 12, 31))


@st.cache_data(show_spinner="Consultando NASA POWER (horário)…", ttl=3600)
def load_hourly(lat: float, lon: float, day: date) -> pd.DataFrame:
    return NasaPowerClient().fetch_hourly(lat, lon, day, day)


def scale_units(df: pd.DataFrame, unit: str, extra: set[str] | None = None) -> pd.DataFrame:
    """Convert energy columns of a display table from kWh/m2/day to the chosen unit."""
    if unit == "kWh/m²/dia":
        return df
    cols = ENERGY_COLS | (extra or set())
    out = df.copy()
    for c in out.columns:
        if c in cols:
            out[c] = out[c] * KWH_TO_MJ
    return out


# ------------------------------------------------------------------ state & sidebar
st.session_state.setdefault("label", DEFAULT_PLACE[0])
st.session_state.setdefault("lat", DEFAULT_PLACE[1])
st.session_state.setdefault("lon", DEFAULT_PLACE[2])
st.session_state.setdefault("places", [])

sb = st.sidebar
sb.title("☀️ Recurso solar")
sb.header("Localidade")
with sb.form("busca_cidade", border=False):
    query = st.text_input("Cidade", placeholder="Ex.: Campina Grande")
    only_br = st.checkbox("Somente Brasil", True)
    submitted = st.form_submit_button("Buscar cidade")
if submitted and query:
    try:
        st.session_state.places = search_places(query, country_code="BR" if only_br else None)
        if not st.session_state.places:
            sb.warning("Nenhuma cidade encontrada.")
    except GeocodingError as exc:
        sb.error(str(exc))
if st.session_state.places:
    options = {p.label: p for p in st.session_state.places}
    choice = sb.selectbox("Resultados", list(options))
    if sb.button("Usar esta localidade"):
        p = options[choice]
        st.session_state.update(label=p.label, lat=p.latitude, lon=p.longitude, places=[])
        st.rerun()
lat = sb.number_input("Latitude (° N+)", -90.0, 90.0, float(st.session_state.lat), 0.01, "%.4f")
lon = sb.number_input("Longitude (° E+)", -180.0, 180.0, float(st.session_state.lon), 0.01, "%.4f")
if (round(lat, 4), round(lon, 4)) != (
    round(st.session_state.lat, 4),
    round(st.session_state.lon, 4),
):
    st.session_state.update(lat=lat, lon=lon, label=f"{lat:.3f}°, {lon:.3f}°")
label = st.session_state.label

sb.header("Dados")
source = sb.radio("Fonte", ["NASA POWER", "Arquivo CSV/Excel"], horizontal=True)
this_year = date.today().year
y0, y1 = sb.slider("Período (anos)", 2001, this_year - 1, (this_year - 5, this_year - 1))

monthly_raw: pd.DataFrame | None = None
annual_raw: pd.DataFrame | None = None
daily_raw: pd.DataFrame | None = None
file_clim: pd.DataFrame | None = None
if source == "Arquivo CSV/Excel":
    up = sb.file_uploader("Arquivo", type=["csv", "xlsx", "xls"])
    if up is None:
        st.info("Envie um arquivo na barra lateral (Atlas INPE/LABREN ou série própria).")
        st.stop()
    try:
        table = importers.read_table(up, up.name)
    except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
        st.error(f"Erro ao ler o arquivo: {exc}")
        st.stop()
    unit_in = sb.selectbox("Unidade dos valores", list(importers.UNIT_FACTORS), index=0)
    layout = sb.radio("Formato", ["12 colunas mensais", "Série diária"])
    cols = list(table.columns)
    try:
        if layout == "12 colunas mensais":
            months = sb.multiselect(
                "Colunas jan→dez (em ordem)", cols, default=importers.guess_month_columns(cols) or []
            )
            lat_c = sb.selectbox("Coluna de latitude (opcional)", ["(nenhuma)", *cols])
            lon_c = sb.selectbox("Coluna de longitude (opcional)", ["(nenhuma)", *cols])
            if lat_c != "(nenhuma)" and lon_c != "(nenhuma)":
                row = importers.nearest_row(table, lat_c, lon_c, lat, lon)
                sb.caption(f"Linha mais próxima: {row} ({table[lat_c].iloc[row]}, {table[lon_c].iloc[row]})")
            else:
                row = int(sb.number_input("Linha (0 = primeira)", 0, len(table) - 1, 0))
            ghi_m = importers.wide_to_monthly(table, months, row, unit_in)
            file_clim = pd.DataFrame({"ghi": ghi_m})
        else:
            dcol = sb.selectbox("Coluna de data", cols)
            vcol = sb.selectbox("Coluna de GHI", cols, index=min(1, len(cols) - 1))
            s = importers.daily_series(table, dcol, vcol, unit_in)
            daily_raw = s.to_frame("ghi")
            file_clim = (
                s.groupby(s.index.month).mean().reindex(range(1, 13)).rename_axis("month").to_frame("ghi")
            )
    except ValueError as exc:
        st.warning(f"Verifique o mapeamento de colunas: {exc}")
        st.stop()
    if file_clim["ghi"].isna().any():
        st.warning("Há meses sem dado no arquivo; as análises mensais ficarão incompletas.")
else:
    try:
        monthly_raw, annual_raw = load_monthly(lat, lon, y0, y1)
    except (NasaPowerError, ValueError) as exc:
        st.error(f"Não foi possível obter os dados mensais: {exc}")
        st.stop()
    try:
        daily_raw = load_daily(lat, lon, y0, y1)
    except (NasaPowerError, ValueError) as exc:
        st.warning(f"Série diária indisponível: {exc}")

sb.header("Superfície e modelos")
mode = sb.radio("Inclinação", ["Horizontal", "Definida", "Ótima anual"])
beta_user = sb.slider("Inclinação β (°)", 0, 90, int(min(abs(lat), 90)), disabled=mode != "Definida")
toward_eq = sb.checkbox("Voltada para o Equador", True)
az_user = sb.slider("Azimute γ (°; 0=Sul, −90=Leste, +90=Oeste)", -180, 180, 0, disabled=toward_eq)
azimuth = tilted.equator_facing_azimuth(lat) if toward_eq else float(az_user)
albedo = sb.slider("Albedo do solo ρg (–)", 0.0, 1.0, DEFAULT_ALBEDO, 0.05)
trans_model = sb.selectbox("Transposição", tilted.TRANSPOSITION_MODELS, index=1)
diff_model = sb.selectbox("Correlação difusa (se faltar dado)", list(MONTHLY_MODELS))
unit = sb.selectbox("Unidade das tabelas", ["kWh/m²/dia", "MJ/m²/dia"])

sb.header("Indicadores")
p_kwp = sb.number_input("Potência FV instalada (kWp)", 0.0, 1e5, 5.0, 0.5)
pr = sb.slider("Performance ratio PR (–)", 0.5, 1.0, 0.80, 0.01)
area = sb.number_input("Área de coletor (m²)", 0.0, 1e5, 4.0, 0.5)
eta = sb.slider("Eficiência do coletor η (–)", 0.0, 1.0, 0.50, 0.01)

# ------------------------------------------------------------------ computations
clim = file_clim if file_clim is not None else an.climatology(monthly_raw)
enriched = an.enrich_monthly(clim, lat, diff_model)
if enriched["ghi"].isna().any():
    st.warning("Há meses sem dados de GHI: modelos mensais indisponíveis.")
    st.stop()

beta_opt, monthly_best, sweep = an.optimal_tilt_info(lat, enriched, azimuth, albedo, trans_model)
annual_curve = sweep.apply(tilted.annual_mean_daily, axis=1)
if mode == "Horizontal":
    surface = an.Surface("Horizontal", 0.0, azimuth)
elif mode == "Definida":
    surface = an.Surface(f"β={beta_user}°, γ={azimuth:.0f}°", float(beta_user), azimuth)
else:
    surface = an.Surface(f"Ótima ({beta_opt:.0f}°)", beta_opt, azimuth)
surf_table = tilted.monthly_tilted(
    lat, enriched["ghi"], surface.beta, surface.azimuth, albedo, trans_model, enriched["dhi"]
)
ht_mean = tilted.annual_mean_daily(surf_table["HT"])
metrics = an.key_metrics(enriched, beta_opt, float(annual_curve.max()))
comparison = an.compare_surfaces(
    lat,
    enriched,
    an.surfaces_for_comparison(lat, float(beta_user), azimuth, beta_opt),
    albedo,
    trans_model,
)
pv_year = indicators.pv_energy_kwh(ht_mean * 365.0, p_kwp, pr)
col_year = indicators.collector_energy_kwh(ht_mean * 365.0, area, eta)
fdisp = 1.0 if unit == "kWh/m²/dia" else KWH_TO_MJ

# ------------------------------------------------------------------ main page
st.title(f"Análise do recurso solar — {label}")
st.caption(
    f"Lat {lat:.3f}°, Lon {lon:.3f}° · Fonte: {source}"
    + (f" · {y0}–{y1}" if source == "NASA POWER" else "")
    + f" · Superfície: {surface.label}"
)
with st.expander("🗺️ Escolher o ponto clicando no mapa"):
    m = folium.Map(location=[lat, lon], zoom_start=5, tiles="OpenStreetMap")
    folium.Marker([lat, lon]).add_to(m)
    out = st_folium(m, height=320, use_container_width=True, returned_objects=["last_clicked"])
    clicked = (out or {}).get("last_clicked")
    if clicked and (round(clicked["lat"], 4), round(clicked["lng"], 4)) != (
        round(lat, 4),
        round(lon, 4),
    ):
        st.session_state.update(
            lat=clicked["lat"], lon=clicked["lng"], label=f"{clicked['lat']:.3f}°, {clicked['lng']:.3f}°"
        )
        st.rerun()

tab_sum, tab_m, tab_d, tab_t, tab_s, tab_raw = st.tabs(
    ["Resumo", "Dados mensais", "Série diária", "Plano inclinado", "Trajetória solar", "Dados brutos"]
)

# Resumo ------------------------------------------------------------------
with tab_sum:
    c = st.columns(6)
    c[0].metric("GHI média anual", f"{metrics['ghi_mean_kwh']:.2f} kWh/m²/dia",
                f"{metrics['ghi_mean_mj']:.1f} MJ/m²/dia", delta_color="off")
    c[1].metric("Melhor mês", f"{metrics['best_month']}", f"{metrics['best_value']:.2f} kWh/m²/dia",
                delta_color="off")
    c[2].metric("Pior mês", f"{metrics['worst_month']}", f"{metrics['worst_value']:.2f} kWh/m²/dia",
                delta_color="off")
    c[3].metric("Kt médio", f"{metrics['kt_mean']:.2f}")
    c[4].metric("Inclinação ótima", f"{metrics['beta_opt']:.0f}°", f"γ = {azimuth:.0f}°",
                delta_color="off")
    c[5].metric("Horas de sol (astron.)", f"{metrics['sun_hours_mean']:.1f} h/dia")
    st.caption(
        f"GHI anual ≈ {metrics['ghi_year_kwh']:.0f} kWh/m²/ano · HT na superfície selecionada: "
        f"{ht_mean:.2f} kWh/m²/dia"
    )
    a, b = st.columns(2)
    a.plotly_chart(fg.monthly_irradiation(enriched), width="stretch")
    b.plotly_chart(fg.monthly_kt(enriched), width="stretch")
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(fg.location_map(lat, lon, label), width="stretch")
    with c2:
        st.subheader("Indicadores simples")
        st.metric("Geração FV anual", f"{pv_year:,.0f} kWh/ano".replace(",", "."),
                  f"{pv_year / p_kwp:.0f} kWh/kWp" if p_kwp else None, delta_color="off")
        st.metric("Energia útil no coletor", f"{col_year:,.0f} kWh/ano".replace(",", "."),
                  f"{col_year * KWH_TO_MJ:,.0f} MJ/ano".replace(",", "."), delta_color="off")
    figs_report = [fg.monthly_irradiation(enriched), fg.monthly_kt(enriched),
                   fg.tilt_comparison(comparison)]
    html = export.build_html_report(
        f"Recurso solar — {label}",
        {"Localidade": label, "Latitude": f"{lat:.4f}°", "Longitude": f"{lon:.4f}°",
         "Fonte": source, "Superfície": f"{surface.label} (albedo {albedo})",
         "Transposição": trans_model},
        {"GHI média": f"{metrics['ghi_mean_kwh']:.2f} kWh/m²/dia",
         "Melhor mês": f"{metrics['best_month']} ({metrics['best_value']:.2f})",
         "Pior mês": f"{metrics['worst_month']} ({metrics['worst_value']:.2f})",
         "Kt médio": f"{metrics['kt_mean']:.2f}",
         "Inclinação ótima": f"{metrics['beta_opt']:.0f}°",
         "Geração FV": f"{pv_year:,.0f} kWh/ano"},
        {"Dados mensais (kWh/m²/dia)": enriched.drop(columns="dhi_source"),
         "Irradiação no plano (kWh/m²/dia)": comparison},
        figs_report, LIMITATIONS,
    )
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Relatório HTML", html, "relatorio_solar.html", "text/html")
    d2.download_button(
        "⬇️ Tabelas (Excel)",
        export.to_excel_bytes({"mensal": enriched, "plano_inclinado": comparison}),
        "recurso_solar.xlsx",
    )
    with st.expander("Hipóteses e limitações do modelo"):
        for item in LIMITATIONS:
            st.markdown(f"- {item}")

# Dados mensais -----------------------------------------------------------
with tab_m:
    show = scale_units(enriched, unit).rename_axis("mês")
    st.dataframe(show.style.format(precision=3, na_rep="–"), width="stretch")
    st.download_button("⬇️ CSV", export.to_csv_bytes(show), "mensal.csv", "text/csv", key="csv_m")
    a, b = st.columns(2)
    a.plotly_chart(fg.diffuse_fraction_comparison(
        an.diffuse_comparison_monthly(enriched, lat), "Fração difusa mensal: NASA × correlações"),
        width="stretch")
    if monthly_raw is not None:
        inter = monthly_raw["ghi"].unstack(level="year")
        b.plotly_chart(
            fg.diffuse_fraction_comparison(inter, "GHI mensal por ano (kWh/m²/dia)"),
            width="stretch",
        )
        st.subheader("Média anual por ano")
        st.dataframe(scale_units(annual_raw, unit).style.format(precision=3, na_rep="–"),
                     width="stretch")

# Série diária ------------------------------------------------------------
with tab_d:
    if daily_raw is None or daily_raw["ghi"].dropna().empty:
        st.info("Série diária indisponível para esta fonte/seleção.")
    else:
        daily = an.enrich_daily(daily_raw, lat)
        st.plotly_chart(fg.daily_series(daily), width="stretch")
        a, b = st.columns(2)
        a.plotly_chart(fg.kt_histogram(daily), width="stretch")
        if "dhi" in daily:
            sample = daily.dropna(subset=["kt_calc", "fd_NASA"])
            bins = pd.cut(sample["kt_calc"], np.arange(0, 1.05, 0.05))
            cols_fd = [c for c in sample.columns if c.startswith("fd_")]
            curve = sample[cols_fd].groupby(bins, observed=True).mean()
            curve.index = [f"{i.mid:.2f}" for i in curve.index]
            curve.columns = [c.removeprefix("fd_") for c in curve.columns]
            b.plotly_chart(
                fg.diffuse_fraction_comparison(
                    curve, "Hd/H diário médio por faixa de Kt (NASA × correlações)", x=curve.index
                ),
                width="stretch",
            )
        st.dataframe(scale_units(daily, unit, {"ghi"}).style.format(precision=3, na_rep="–"),
                     width="stretch", height=300)
        st.download_button("⬇️ CSV diário", export.to_csv_bytes(scale_units(daily, unit)),
                           "diario.csv", "text/csv")
    if source == "NASA POWER":
        with st.expander("Perfil horário de um dia (NASA POWER, hora solar local)"):
            day = st.date_input("Dia", date(y1, 6, 21), min_value=date(2001, 1, 1),
                                max_value=date(y1, 12, 31))
            if st.button("Carregar perfil horário"):
                try:
                    hourly = load_hourly(lat, lon, day)
                    n = int(pd.Timestamp(day).day_of_year)
                    n = n - 1 if pd.Timestamp(day).is_leap_year and day.month > 2 else n
                    g0 = np.asarray(g0_horizontal(lat, min(n, 365), 15.0 * (hourly.index.hour + 0.5 - 12.0)))
                    st.plotly_chart(fg.hourly_profile(hourly, g0), width="stretch")
                    st.dataframe(hourly, width="stretch")
                except (NasaPowerError, ValueError) as exc:
                    st.error(str(exc))

# Plano inclinado -----------------------------------------------------------
with tab_t:
    st.plotly_chart(fg.tilt_comparison(comparison), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(fg.tilt_sweep_curve(sweep, annual_curve, beta_opt), width="stretch")
    b.plotly_chart(fg.optimal_tilt_by_month(monthly_best), width="stretch")
    ann = comparison.apply(tilted.annual_mean_daily)
    summary = pd.DataFrame({
        "HT média (kWh/m²/dia)": ann,
        "HT anual (kWh/m²/ano)": ann * 365.0,
        "Ganho vs horizontal (%)": (ann / ann.iloc[0] - 1) * 100,
    })
    st.dataframe(summary.style.format(precision=2), width="stretch")
    st.subheader(f"Detalhe mensal da superfície selecionada ({surface.label})")
    st.dataframe(scale_units(surf_table, unit, {"HT"}).style.format(precision=3), width="stretch")
    st.caption(
        f"Mês a mês, a inclinação ótima varia de {monthly_best.min():.0f}° a "
        f"{monthly_best.max():.0f}° (γ = {azimuth:.0f}°). Rb = razão de feixe direto plano/horizontal."
    )
    st.download_button(
        "⬇️ Excel (plano inclinado)",
        export.to_excel_bytes({"comparativo": comparison, "varredura_beta": sweep,
                               "superficie": surf_table}),
        "plano_inclinado.xlsx",
    )

# Trajetória solar ----------------------------------------------------------
with tab_s:
    path = sunpath.sun_path_table(lat)
    lines = sunpath.hourly_lines(lat)
    a, b = st.columns(2)
    a.plotly_chart(fg.sun_path_cartesian(path, lines), width="stretch")
    b.plotly_chart(fg.sun_path_polar(path), width="stretch")
    dd = st.date_input("Posição do Sol em uma data", date(y1, 12, 21) if lat < 0 else date(y1, 6, 21))
    hh = st.slider("Hora solar", 0.0, 24.0, 12.0, 0.25)
    nn = min(int(pd.Timestamp(dd).day_of_year), 365)
    dl = float(declination(nn))
    om = 15.0 * (hh - 12.0)
    st.write(
        f"Declinação **{dl:.2f}°** · altura solar **{float(solar_altitude(lat, dl, om)):.1f}°** · "
        f"azimute **{float(solar_azimuth(lat, dl, om)):.1f}°** (0=Sul, Leste −, Oeste +)"
    )
    dlt = pd.DataFrame({"Duração do dia (h)": enriched["day_length_h"],
                        "Declinação (°)": declination(MEAN_DAYS)}, index=enriched.index)
    st.dataframe(dlt.style.format(precision=2), width="stretch")
    st.caption(f"Dias por mês usados nas médias anuais: {', '.join(map(str, DAYS_IN_MONTH))}.")

# Dados brutos ---------------------------------------------------------------
with tab_raw:
    if monthly_raw is not None:
        st.subheader("Mensal (NASA POWER, kWh/m²/dia)")
        st.dataframe(monthly_raw, width="stretch")
        st.subheader("Média anual (NASA 'mês 13')")
        st.dataframe(annual_raw, width="stretch")
    if daily_raw is not None:
        st.subheader("Diário")
        st.dataframe(daily_raw, width="stretch", height=300)
    sheets = {"mensal_media": clim}
    if monthly_raw is not None:
        sheets.update(mensal_bruto=monthly_raw, anual=annual_raw)
    if daily_raw is not None:
        sheets["diario"] = daily_raw
    st.download_button("⬇️ Tudo em Excel", export.to_excel_bytes(sheets), "dados_brutos.xlsx")
    if monthly_raw is not None:
        st.download_button("⬇️ Mensal CSV", export.to_csv_bytes(monthly_raw), "mensal_bruto.csv",
                           "text/csv", key="csv_raw")
    st.caption("Valores ausentes (−999 na API) aparecem como vazios/NaN.")