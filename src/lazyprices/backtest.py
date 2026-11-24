"""Signal evaluation: rank IC by horizon, quintile long-short, turnover and factor alpha."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MIN_NAMES = 20
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def forward_returns(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Compounded return over months t+1..t+h, aligned to month t."""
    log = np.log1p(returns)
    fwd = log[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)
    return np.expm1(fwd)


def ic_series(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    sig, ret = signal.align(fwd, join="inner")
    ics = {}
    for t in sig.index:
        pair = pd.concat([sig.loc[t], ret.loc[t]], axis=1).dropna()
        if len(pair) >= MIN_NAMES:
            ics[t] = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank())
    return pd.Series(ics, dtype=float)


def nw_tstat(x: pd.Series, lags: int) -> float:
    x = x.dropna()
    if len(x) < 12:
        return float("nan")
    fit = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def ic_by_horizon(signal: pd.DataFrame, returns: pd.DataFrame,
                  horizons=(1, 3, 6, 12)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ic = ic_series(signal, forward_returns(returns, h))
        rows.append({"horizon_months": h, "mean_ic": ic.mean(), "t_stat_nw": nw_tstat(ic, h),
                     "hit_rate": (ic > 0).mean(), "months": len(ic)})
    return pd.DataFrame(rows)


def quintile_weights(signal_row: pd.Series, q: int = 5) -> pd.Series:
    s = signal_row.dropna()
    if len(s) < MIN_NAMES:
        return pd.Series(dtype=float)
    bucket = pd.qcut(s.rank(method="first"), q, labels=False)
    long, short = s.index[bucket == q - 1], s.index[bucket == 0]
    w = pd.Series(0.0, index=s.index)
    w[long] = 1.0 / len(long)
    w[short] = -1.0 / len(short)
    return w


def long_short(signal: pd.DataFrame, returns: pd.DataFrame, q: int = 5,
               cost_bps: float = 10.0) -> pd.DataFrame:
    """Monthly long-short returns: top quintile minus bottom, equal weight, rebalanced monthly."""
    rows, prev = [], pd.Series(dtype=float)
    dates = signal.index.intersection(returns.index)
    for t, nxt in zip(dates[:-1], dates[1:]):
        w = quintile_weights(signal.loc[t], q)
        if w.empty:
            continue
        r = returns.loc[nxt].reindex(w.index).fillna(0.0)
        turnover = w.sub(prev, fill_value=0.0).abs().sum() / 2.0
        gross = float((w * r).sum())
        rows.append({"date": nxt, "gross": gross, "turnover": turnover,
                     "net": gross - 2.0 * turnover * cost_bps / 1e4,
                     "long": float(r[w > 0].mean()), "short": float(r[w < 0].mean())})
        prev = w
    return pd.DataFrame(rows).set_index("date")


def performance(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"ann_return": r.mean() * 12, "ann_vol": r.std() * np.sqrt(12),
            "sharpe": r.mean() / r.std() * np.sqrt(12) if r.std() else np.nan,
            "t_stat": r.mean() / r.std() * np.sqrt(len(r)) if r.std() else np.nan,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "months": len(r)}


def factor_alpha(ls: pd.Series, ff: pd.DataFrame) -> dict:
    """Alpha of a zero-cost long-short against FF5 + momentum, HAC errors (3 lags)."""
    df = pd.concat([ls.rename("ls"), ff[FACTORS]], axis=1, join="inner").dropna()
    if len(df) < 24:
        return {"alpha_ann": np.nan, "alpha_t": np.nan, "months": len(df)}
    fit = sm.OLS(df["ls"], sm.add_constant(df[FACTORS])).fit(cov_type="HAC",
                                                            cov_kwds={"maxlags": 3})
    out = {"alpha_ann": fit.params["const"] * 12, "alpha_t": fit.tvalues["const"],
           "r2": fit.rsquared, "months": len(df)}
    out.update({f"beta_{k}": fit.params[k] for k in FACTORS})
    return out
