import numpy as np
import pandas as pd

from lazyprices.signal import sector_neutral, to_monthly

MONTHS = pd.date_range("2020-01-31", periods=24, freq="ME")


def test_signal_uses_only_filings_before_month_end():
    f = pd.DataFrame({"ticker": ["A", "A"], "filing_date": ["2020-03-31", "2021-03-01"],
                      "x": [1.0, 2.0]})
    p = to_monthly(f, "x", MONTHS)["A"]
    assert np.isnan(p[pd.Timestamp("2020-02-29")])
    assert np.isnan(p[pd.Timestamp("2020-03-31")])      # filed that day: not yet usable
    assert p[pd.Timestamp("2020-04-30")] == 1.0
    assert p[pd.Timestamp("2021-02-28")] == 1.0
    assert p[pd.Timestamp("2021-03-31")] == 2.0


def test_stale_filings_expire_after_a_year():
    f = pd.DataFrame({"ticker": ["A"], "filing_date": ["2020-01-15"], "x": [1.0]})
    p = to_monthly(f, "x", MONTHS)["A"]
    assert p[pd.Timestamp("2020-12-31")] == 1.0
    assert np.isnan(p[pd.Timestamp("2021-06-30")])


def test_sector_neutral_demeans_within_groups():
    panel = pd.DataFrame([[1.0, 3.0, 10.0, 12.0, 14.0]], columns=list("ABCDE"))
    sectors = pd.Series({"A": "10", "B": "10", "C": "20", "D": "20", "E": "20"})
    out = sector_neutral(panel, sectors)
    assert np.allclose(out[["C", "D", "E"]].values, [[-2.0, 0.0, 2.0]])
    # A and B form a group below the minimum size, so they are demeaned against everyone
    assert np.isclose(out["A"].iloc[0], 1.0 - 8.0)
