import io

import numpy as np
import pandas as pd
import pytest

from solarapp.data import importers as imp

ATLAS_CSV = (
    "LON;LAT;ANNUAL;JAN;FEB;MAR;APR;MAY;JUN;JUL;AUG;SEP;OCT;NOV;DEC\n"
    + "-34,88;-7,12;5500;"
    + ";".join(str(5000 + 100 * m) for m in range(12))
    + "\n-40,00;-10,00;5000;"
    + ";".join("4000" for _ in range(12))
    + "\n"
)


def test_read_csv_decimal_comma_and_months():
    df = imp.read_table(io.BytesIO(ATLAS_CSV.encode()), "atlas.csv")
    assert df["LAT"].iloc[0] == pytest.approx(-7.12)
    cols = imp.guess_month_columns(list(df.columns))
    assert cols == "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    row = imp.nearest_row(df, "LAT", "LON", -7.0, -35.0)
    assert row == 0
    s = imp.wide_to_monthly(df, cols, row, unit="Wh/m²/dia")
    assert s.loc[1] == pytest.approx(5.0) and s.loc[12] == pytest.approx(6.1)


def test_portuguese_month_names():
    cols = ["lat", "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho",
            "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    assert imp.guess_month_columns(cols) is not None
    assert imp.guess_month_columns(["a", "b"]) is None


def test_wrong_month_count():
    with pytest.raises(ValueError):
        imp.wide_to_monthly(pd.DataFrame({"a": [1]}), ["a"])


def test_daily_series_units_and_missing():
    df = pd.DataFrame({"data": ["01/02/2020", "02/02/2020", "xx"], "v": [18.0, -999, 3]})
    s = imp.daily_series(df, "data", "v", unit="MJ/m²/dia")
    assert s.iloc[0] == pytest.approx(5.0) and np.isnan(s.iloc[1]) and len(s) == 2
    assert s.index[0] == pd.Timestamp("2020-02-01")


def test_read_excel(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(p, index=False)
    with open(p, "rb") as f:
        assert len(imp.read_table(f, "x.xlsx")) == 2