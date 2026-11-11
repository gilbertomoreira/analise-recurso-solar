"""Integration tests for analysis, plots and exports with synthetic data (no network)."""

import numpy as np
import pandas as pd
import pytest

from solarapp import analysis as an
from solarapp.physics import tilted
from solarapp.physics.extraterrestrial import h0_monthly_mean_kwh
from solarapp.plots import figures as fg
from solarapp.reports import export

LAT = -7.12


@pytest.fixture
def monthly():
    h0 = h0_monthly_mean_kwh(LAT)
    idx = pd.MultiIndex.from_product([[2020, 2021], range(1, 13)], names=["year", "month"])
    ghi = np.tile(h0 * 0.6, 2)
    return pd.DataFrame(
        {"ghi": ghi, "dhi": ghi * 0.35, "dni": ghi * 0.6, "kt": 0.6, "t2m": 27.0}, index=idx
    )


@pytest.fixture
def enriched(monthly):
    return an.enrich_monthly(an.climatology(monthly), LAT)


def test_climatology_and_enrich(enriched):
    assert len(enriched) == 12
    assert enriched["kt_calc"].iloc[0] == pytest.approx(0.6)
    assert (enriched["dhi_source"] == "NASA").all()
    assert 11 < enriched["day_length_h"].min() and enriched["day_length_h"].max() < 13


def test_missing_diffuse_filled_by_model(monthly):
    m = monthly.drop(columns=["dhi"])
    e = an.enrich_monthly(an.climatology(m), LAT, "Collares-Pereira–Rabl")
    assert (e["dhi_source"] == "correlação").all()
    assert (e["dhi"] > 0).all() and (e["dhi"] < e["ghi"]).all()


def test_diffuse_comparison(enriched):
    tbl = an.diffuse_comparison_monthly(enriched, LAT)
    assert list(tbl.columns) == ["NASA", "Erbs", "Orgill-Hollands", "Collares-Pereira–Rabl"]
    assert tbl["NASA"].iloc[0] == pytest.approx(0.35)


def test_enrich_daily_leap_year():
    idx = pd.date_range("2020-02-27", "2020-03-02")
    d = pd.DataFrame({"ghi": 5.0, "dhi": 1.5}, index=idx)
    out = an.enrich_daily(d, LAT)
    assert list(out["n"]) == [58, 59, 59, 60, 61]  # 28 Feb, 29 Feb (-> 59), 1-2 Mar
    assert out.loc["2020-03-01", "n"] == 60 and out.loc["2020-02-29", "n"] == 59
    assert {"kt_calc", "fd_Erbs", "fd_Orgill-Hollands", "fd_NASA"} <= set(out.columns)


def test_surface_comparison_and_metrics(enriched):
    best, monthly_best, sweep = an.optimal_tilt_info(LAT, enriched, None, 0.2, "Hay-Davies")
    surfaces = an.surfaces_for_comparison(LAT, 15.0, 90.0, best)
    ht = an.compare_surfaces(LAT, enriched, surfaces, 0.2, "Hay-Davies")
    assert ht.shape == (12, 4)
    annual = sweep.apply(tilted.annual_mean_daily, axis=1)
    assert annual.idxmax() == best
    m = an.key_metrics(enriched, best, float(annual.max()))
    assert m["ghi_mean_mj"] == pytest.approx(3.6 * m["ghi_mean_kwh"])
    assert m["best_value"] >= m["worst_value"] and 0.55 < m["kt_mean"] < 0.65
    figs = [
        fg.monthly_irradiation(enriched), fg.monthly_kt(enriched),
        fg.tilt_comparison(ht), fg.tilt_sweep_curve(sweep, annual, best),
        fg.optimal_tilt_by_month(monthly_best),
        fg.diffuse_fraction_comparison(an.diffuse_comparison_monthly(enriched, LAT), "t"),
        fg.location_map(LAT, -34.88, "JP"),
    ]
    assert all(len(f.data) > 0 for f in figs)
    html = export.build_html_report("T", {"a": "b"}, {"k": "v"}, {"tab": ht}, figs[:2], ["nota"])
    assert "<h1>T</h1>" in html and "Hipóteses" in html


def test_exports(enriched):
    csv = export.to_csv_bytes(enriched[["ghi"]])
    assert csv.startswith(b"\xef\xbb\xbf") and b";" in csv
    xl = export.to_excel_bytes({"a": enriched[["ghi"]], "b": enriched[["kt_calc"]]})
    assert xl[:2] == b"PK"