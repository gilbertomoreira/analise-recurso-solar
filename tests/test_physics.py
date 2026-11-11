"""Physics tests: D&B Table 1.6.1, Example 1.5.1 values and analytic identities."""

from datetime import date

import numpy as np
import pytest

from solarapp.physics import clearness, diffuse, indicators, sunpath
from solarapp.physics import extraterrestrial as et
from solarapp.physics import geometry as g
from solarapp.physics import tilted as t

# D&B Table 1.6.1: recommended average day and declination of each month.
TABLE_161_DELTA = [-20.9, -13.0, -2.4, 9.4, 18.8, 23.1, 21.2, 13.5, 2.2, -9.6, -18.9, -23.0]


def test_day_of_year():
    assert g.day_of_year(1, 1) == 1 and g.day_of_year(12, 31) == 365
    assert g.day_of_year(2, 13) == 44 and g.day_of_year(3, 15) == 74
    assert g.day_of_year_from_date(date(2020, 2, 29)) == 59
    with pytest.raises(ValueError):
        g.day_of_year(2, 30)


def test_mean_days_match_table():
    expected = [17, 47, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]
    assert list(g.MEAN_DAYS) == expected
    # mean days are n = cumulative + offset consistent with the calendar
    days = [17, 16, 16, 15, 15, 11, 17, 16, 15, 15, 14, 10]  # calendar days of Table 1.6.1
    assert [g.day_of_year(m, d) for m, d in zip(range(1, 13), days, strict=True)] == expected


def test_declination_table_values():
    np.testing.assert_allclose(g.declination(g.MEAN_DAYS), TABLE_161_DELTA, atol=0.15)


def test_declination_example_feb13():
    # D&B Example 1.6.1 (13 Feb, n = 44): delta = -14 deg (rounded)
    assert g.declination(44) == pytest.approx(-14.0, abs=0.1)


def test_spencer_close_to_cooper():
    n = np.arange(1, 366)
    diff = np.abs(g.declination(n) - g.declination_spencer(n))
    assert diff.max() < 1.5  # Cooper vs Spencer differ by up to ~1.4 deg


def test_equation_of_time_example_151():
    # D&B Example 1.5.1: 3 Feb (n = 34): E = -13.5 min
    assert g.equation_of_time(34) == pytest.approx(-13.5, abs=0.1)
    # Madison, 89.4 W, std meridian 90 W, 10:30 CST -> ~10:19 solar time
    ts = g.solar_time(10.5, 34, longitude=-89.4, std_meridian=-90.0)
    assert ts == pytest.approx(10 + 18.9 / 60, abs=0.01)


def test_equation_of_time_range():
    e = g.equation_of_time(np.arange(1, 366))
    assert -15 < e.min() < -13 and 15.5 < e.max() < 17  # ~ -14.2 (Feb) .. +16.4 (Nov)


def test_longitude_correction():
    assert g.longitude_correction_min(-45.0, -45.0) == 0
    assert g.longitude_correction_min(-34.88, g.standard_meridian(-3)) == pytest.approx(4 * 10.12)


def test_hour_angle_and_sunset():
    assert g.hour_angle(12) == 0 and g.hour_angle(10) == -30 and g.hour_angle(15) == 45
    assert g.sunset_hour_angle(0, 23.0) == pytest.approx(90.0)
    assert g.sunset_hour_angle(80, 20) == 180.0  # polar day
    assert g.sunset_hour_angle(80, -20) == 0.0  # polar night
    # equinox: 12 h everywhere
    assert g.day_length(43, 80) == pytest.approx(12.0, abs=0.1)
    # 43N, 21 Jun ~ 15.2 h
    assert g.day_length(43, 172) == pytest.approx(15.2, abs=0.2)


def test_zenith_altitude_azimuth():
    # equinox noon at equator: sun at zenith
    assert g.zenith_angle(0, 0, 0) == pytest.approx(0.0, abs=1e-6)
    # noon at 40N, delta = 0 -> altitude 50, azimuth 0
    assert g.solar_altitude(40, 0, 0) == pytest.approx(50.0)
    assert g.solar_azimuth(40, 0, 0) == pytest.approx(0.0, abs=1e-3)
    # morning east (negative), afternoon west (positive)
    assert g.solar_azimuth(40, 0, -45) < 0 < g.solar_azimuth(40, 0, 45)
    # equinox, any lat: sunrise due east (-90)
    assert g.solar_azimuth(40, 0, -90) == pytest.approx(-90.0, abs=1e-6)
    # Southern hemisphere summer noon: sun is to the south of zenith? (az = +-180 -> north)
    assert abs(g.solar_azimuth(-23, 20, 0)) == pytest.approx(180.0, abs=1e-6)


def test_eccentricity():
    assert et.eccentricity_factor(3) == pytest.approx(1.033, abs=0.001)
    assert et.eccentricity_factor(185) == pytest.approx(0.967, abs=0.001)
    assert et.normal_extraterrestrial_irradiance(1) == pytest.approx(1361 * 1.033, rel=1e-3)


def test_h0_equator_equinox_analytic():
    n = 80
    expected = 24 * 3600 / np.pi * 1361 * et.eccentricity_factor(n) * np.cos(0) * 1.0 / 3.6e6
    assert et.h0_daily_kwh(0, n) == pytest.approx(expected * np.cos(np.radians(g.declination(n))),
                                                  rel=5e-3)


def test_h0_hand_calculation_43n_mar16():
    # n = 75, 43N, Gsc = 1367 (value used in D&B text): H0 = 26.04 MJ/m2 (hand calculation)
    assert et.h0_daily_mj(43, 75, gsc=1367) == pytest.approx(26.04, abs=0.05)


def test_hourly_sums_to_daily_and_g0():
    lat, n = 43.0, 75
    hours = sum(et.h0_hourly_kwh(lat, n, w, w + 15) for w in range(-180, 180, 15))
    assert hours == pytest.approx(float(et.h0_daily_kwh(lat, n)), rel=1e-9)
    # hourly energy ~ G0 at hour midpoint
    g0 = et.g0_horizontal(lat, n, -22.5)
    assert et.h0_hourly_kwh(lat, n, -30, -15) * 1000 == pytest.approx(g0, rel=0.02)
    assert et.g0_horizontal(lat, n, 180.0) == 0.0  # night
    assert et.h0_hourly_kwh(lat, n, 100, 120) == 0.0


def test_h0_polar_night_and_mj():
    assert et.h0_daily_kwh(85, 355) == pytest.approx(0.0, abs=1e-9)
    assert et.h0_daily_mj(0, 80) == pytest.approx(3.6 * et.h0_daily_kwh(0, 80))
    assert et.h0_monthly_mean_kwh(-7.12).shape == (12,)


def test_kt():
    assert clearness.clearness_index(5.0, 10.0) == 0.5
    assert np.isnan(clearness.clearness_index(1.0, 0.0))
    kt = clearness.monthly_kt([5.5] * 12, -7.12)
    assert np.all((kt > 0.4) & (kt < 0.9))
    assert clearness.daily_kt(5.5, -7.12, 100) == pytest.approx(5.5 / et.h0_daily_kwh(-7.12, 100))


def test_diffuse_limits_and_monotonic():
    assert diffuse.erbs_daily(0.8, 70) == 0.143 and diffuse.erbs_daily(0.8, 95) == 0.175
    assert diffuse.orgill_hollands(0.9) == 0.177
    assert diffuse.orgill_hollands(0.35) == pytest.approx(0.913, abs=0.001)
    assert diffuse.collares_pereira_rabl_daily(0.9) == 0.2
    assert diffuse.collares_pereira_rabl_daily(0.1) == 0.99
    kts = np.linspace(0.2, 0.7, 20)
    for f in (diffuse.erbs_daily(kts, 70), diffuse.orgill_hollands(kts),
              diffuse.collares_pereira_rabl_daily(kts)):
        assert np.all(np.diff(f) <= 1e-9) and np.all((f >= 0) & (f <= 1))
    # the three correlations agree within ~0.15 for mid Kt
    mid = [float(diffuse.erbs_daily(0.5, 70)), float(diffuse.orgill_hollands(0.5)),
           float(diffuse.collares_pereira_rabl_daily(0.5))]
    assert max(mid) - min(mid) < 0.15


def test_monthly_diffuse_models():
    erbs = diffuse.erbs_monthly(0.5, 80)
    cpr = diffuse.collares_pereira_rabl_monthly(0.5, 80)
    assert erbs == pytest.approx(1.391 - 3.560 * 0.5 + 4.189 * 0.25 - 2.137 * 0.125)
    assert abs(erbs - cpr) < 0.1 and 0 < erbs < 1
    assert diffuse.diffuse_fraction(0.5, 80, "Erbs", monthly=True) == erbs
    with pytest.raises(ValueError):
        diffuse.diffuse_fraction(0.5, 80, "xx", monthly=False)


@pytest.mark.parametrize("lat,beta", [(40, 60), (-7.12, 20), (-23.5, 30), (43, 45)])
def test_rb_numeric_matches_closed_form(lat, beta):
    for n in g.MEAN_DAYS:
        assert t.rb_daily(lat, int(n), beta) == pytest.approx(
            t.rb_monthly_closed_form(lat, int(n), beta), rel=2e-3
        )


def test_rb_horizontal_is_one_and_incidence():
    assert t.rb_daily(40, 100, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert t.rb_instantaneous(40, 10.0, 0.0, 0.0, 30.0) == pytest.approx(1.0)
    # surface normal to the sun: beta = zenith, gamma = solar azimuth -> cos(theta) = 1
    d, w, lat = 10.0, -30.0, 40.0
    beta = float(g.zenith_angle(lat, d, w))
    gam = float(g.solar_azimuth(lat, d, w))
    assert t.cos_incidence(lat, d, beta, gam, w) == pytest.approx(1.0, abs=1e-9)


def test_rb_winter_gain_and_azimuth_symmetry():
    assert t.rb_daily(40, 17, 60) > 2  # tilted equator-facing surface in winter
    # east/west symmetry of daily integral
    assert t.rb_daily(40, 100, 30, gamma=-60) == pytest.approx(t.rb_daily(40, 100, 30, gamma=60))
    # facing away from the equator gets less
    assert t.rb_daily(40, 100, 30, gamma=180) < t.rb_daily(40, 100, 30, gamma=0)
    arr = t.rb_daily(40, [17, 105], 30)
    assert np.shape(arr) == (2,)


def test_transposition_models():
    h, hd, h0, rb = 5.0, 1.5, 9.0, 1.3
    # horizontal plane: both return H
    assert t.tilted_isotropic(h, hd, 1.0, 0.0) == pytest.approx(h)
    assert t.tilted_hay_davies(h, hd, h0, 1.0, 0.0) == pytest.approx(h)
    # isotropic formula by hand, beta = 30
    b = np.cos(np.radians(30))
    expected = (h - hd) * rb + hd * (1 + b) / 2 + h * 0.2 * (1 - b) / 2
    assert t.tilted_isotropic(h, hd, rb, 30.0) == pytest.approx(expected)
    # Hay-Davies concentrates diffuse towards the sun: >= isotropic when Rb > 1
    assert t.tilted_hay_davies(h, hd, h0, rb, 30.0) > t.tilted_isotropic(h, hd, rb, 30.0)
    # albedo increases HT on a vertical plane
    assert t.tilted_isotropic(h, hd, rb, 90.0, 0.5) > t.tilted_isotropic(h, hd, rb, 90.0, 0.1)
    assert t.transpose_daily(h, hd, h0, rb, 30, 0.2, "Hay-Davies") > 0
    with pytest.raises(ValueError):
        t.transpose_daily(h, hd, h0, rb, 30, 0.2, "x")


def test_monthly_tilted_and_optimum_midlat():
    lat = 40.0
    h = et.h0_monthly_mean_kwh(lat) * 0.55  # constant Kt
    df = t.monthly_tilted(lat, h, 40)
    assert list(df.columns) == ["H", "Hd", "H0", "Kt", "Rb", "HT"] and len(df) == 12
    assert df["Kt"].iloc[0] == pytest.approx(0.55)
    sweep = t.tilt_sweep(lat, h)
    best, monthly_best = t.optimal_tilts(sweep)
    assert 25 <= best <= 45  # rule of thumb: close to latitude
    assert monthly_best[12] > monthly_best[6]  # steeper in winter than in summer
    assert t.annual_mean_daily([1.0] * 12) == pytest.approx(1.0)
    both = t.monthly_tilted(lat, h, 40, model="Hay-Davies", hd_monthly=h * 0.4)
    assert np.all(both["HT"] > 0)


def test_optimal_tilt_southern_hemisphere_faces_north():
    lat = -23.5
    h = et.h0_monthly_mean_kwh(lat) * 0.55
    best, _ = t.optimal_tilts(t.tilt_sweep(lat, h))
    assert 10 <= best <= 35
    assert t.equator_facing_azimuth(lat) == 180.0 and t.equator_facing_azimuth(10) == 0.0


def test_sunpath():
    p = sunpath.day_path(40.0, 80)  # equinox
    assert p["altitude_deg"].max() == pytest.approx(50.0, abs=0.7)
    assert (p["altitude_deg"] >= 0).all()
    assert p["azimuth_north_deg"].between(0, 360).all()
    tbl = sunpath.sun_path_table(-7.12)
    assert tbl["day_label"].nunique() == 3
    lines = sunpath.hourly_lines(-7.12)
    assert (lines["altitude_deg"] >= 0).all() and len(lines) > 50


def test_indicators():
    assert indicators.pv_energy_kwh(5.0, 3.0, 0.8) == pytest.approx(12.0)
    assert indicators.collector_energy_kwh(5.0, 4.0, 0.5) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        indicators.pv_energy_kwh(5.0, 3.0, 1.5)
    with pytest.raises(ValueError):
        indicators.collector_energy_kwh(5.0, -1.0, 0.5)