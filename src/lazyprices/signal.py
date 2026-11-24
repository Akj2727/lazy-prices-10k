"""Turn filing-level features into a point-in-time monthly signal panel.

At month-end t a firm's value is its latest 10-K filed strictly before t, as long as
that filing is at most 12 months old. Positions formed at t earn the return of month
t+1, so nothing filed after t can leak into a trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_AGE_MONTHS = 12
MIN_SECTOR_SIZE = 3


def to_monthly(features: pd.DataFrame, col: str, months: pd.DatetimeIndex,
               max_age: int = MAX_AGE_MONTHS) -> pd.DataFrame:
    f = features.dropna(subset=[col]).copy()
    f["filing_date"] = pd.to_datetime(f["filing_date"])
    out = {}
    for ticker, g in f.groupby("ticker"):
        g = g.sort_values("filing_date")
        s = pd.Series(g[col].values, index=g["filing_date"].values)
        dates = pd.Series(g["filing_date"].values, index=g["filing_date"].values)
        # value and filing date known at each month-end, using filings strictly before it
        pos = np.searchsorted(s.index.values, months.values, side="left") - 1
        vals = np.where(pos >= 0, s.values[np.clip(pos, 0, None)], np.nan)
        filed = np.where(pos >= 0, dates.values[np.clip(pos, 0, None)],
                         np.datetime64("NaT", "ns"))
        age = (months.values - filed) / np.timedelta64(1, "D")
        vals = np.where(age <= max_age * 31, vals, np.nan)
        out[ticker] = vals
    return pd.DataFrame(out, index=months)


def zscore(panel: pd.DataFrame) -> pd.DataFrame:
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sd, axis=0)


def sector_neutral(panel: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Demean each month within 2-digit SIC groups; small groups fall back to the market."""
    groups = sectors.reindex(panel.columns).fillna("NA")
    out = panel.copy()
    for _, cols in groups.groupby(groups).groups.items():
        cols = list(cols)
        if len(cols) >= MIN_SECTOR_SIZE:
            out[cols] = panel[cols].sub(panel[cols].mean(axis=1), axis=0)
        else:
            out[cols] = panel[cols].sub(panel.mean(axis=1), axis=0)
    return out


def sic2(features: pd.DataFrame) -> pd.Series:
    s = features.drop_duplicates("ticker", keep="last").set_index("ticker")["sic"]
    return s.astype(str).str.extract(r"^(\d{2})")[0]


def combine(panels: list[pd.DataFrame]) -> pd.DataFrame:
    """Average of cross-sectional z-scores, skipping missing pieces per firm."""
    z = [zscore(p) for p in panels]
    stacked = pd.concat(z, keys=range(len(z)))
    return stacked.groupby(level=1).mean().reindex(panels[0].index)
