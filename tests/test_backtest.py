import numpy as np
import pandas as pd

from lazyprices.backtest import forward_returns, ic_by_horizon, long_short, performance
from lazyprices.market import parse_french_csv

rng = np.random.default_rng(0)
MONTHS = pd.date_range("2015-01-31", periods=96, freq="ME")
NAMES = [f"S{i}" for i in range(100)]


def planted(strength: float):
    signal = pd.DataFrame(rng.normal(size=(96, 100)), index=MONTHS, columns=NAMES)
    noise = rng.normal(0, 0.05, size=(96, 100))
    rets = pd.DataFrame(noise, index=MONTHS, columns=NAMES)
    rets.iloc[1:] += strength * signal.shift(1).iloc[1:].values  # signal at t predicts t+1
    return signal, rets


def test_forward_returns_compound_the_next_months():
    r = pd.DataFrame({"A": [0.1, 0.2, -0.1, 0.05]}, index=MONTHS[:4])
    f = forward_returns(r, 2)
    assert np.isclose(f["A"].iloc[0], 1.2 * 0.9 - 1)
    assert np.isnan(f["A"].iloc[-1])


def test_planted_signal_is_found_and_random_one_is_not():
    sig, rets = planted(0.01)
    ic = ic_by_horizon(sig, rets, horizons=(1,))
    assert ic["mean_ic"].iloc[0] > 0.1 and ic["t_stat_nw"].iloc[0] > 5
    ls = long_short(sig, rets, cost_bps=0)
    assert performance(ls["net"])["sharpe"] > 2
    noise_sig = pd.DataFrame(rng.normal(size=(96, 100)), index=MONTHS, columns=NAMES)
    assert abs(ic_by_horizon(noise_sig, rets, horizons=(1,))["mean_ic"].iloc[0]) < 0.05


def test_costs_reduce_net_return():
    sig, rets = planted(0.01)
    gross = long_short(sig, rets, cost_bps=0)["net"].mean()
    net = long_short(sig, rets, cost_bps=50)["net"].mean()
    assert net < gross


def test_parse_french_csv_monthly_block():
    text = ("This file was created...\n\n,Mkt-RF,SMB,HML,RF\n"
            "202001,  -0.11, -3.09, -6.25, 0.13\n202002, -8.13, 0.95, -3.84, 0.12\n\n"
            " Annual Factors: January-December \n,Mkt-RF,SMB,HML,RF\n2020, 23.66, 12.0, -46.5, 0.45\n")
    df = parse_french_csv(text)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 2 and np.isclose(df["Mkt-RF"].iloc[1], -0.0813)
    assert df.index[0] == pd.Timestamp("2020-01-31")
