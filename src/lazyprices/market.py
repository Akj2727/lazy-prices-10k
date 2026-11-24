"""Market data: the stock universe, monthly total returns and Fama-French factors."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[2] / "cache"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5 = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM = "F-F_Momentum_Factor_CSV.zip"


def sp500_tickers() -> list[str]:
    """Current S&P 500 members. Using today's list adds survivorship bias (see README)."""
    html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return sorted(table["Symbol"].astype(str).str.upper().tolist())


def monthly_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Month-end total returns (dividend and split adjusted), dates x tickers."""
    path = CACHE / f"prices_{start}_{end}_{len(tickers)}.csv"
    if path.exists():
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        yahoo = [t.replace(".", "-") for t in tickers]
        raw = yf.download(yahoo, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)["Close"]
        raw.columns = [c.replace("-", ".") for c in raw.columns]
        prices = raw.resample("ME").last()
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path)
    return prices.pct_change(fill_method=None).iloc[1:]


def parse_french_csv(text: str) -> pd.DataFrame:
    """Read the monthly block of a Ken French CSV (values in percent) as decimals."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip()[:6].isdigit())
    header = lines[start - 1].split(",")
    rows = []
    for l in lines[start:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 6:
            break  # the annual block starts after a blank line
        rows.append(parts)
    cols = ["date"] + [h.strip() for h in header[1:]]
    df = pd.DataFrame(rows, columns=cols[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date").astype(float) / 100.0


def _french(name: str) -> pd.DataFrame:
    path = CACHE / "french" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(FF_BASE + name, timeout=60).content)
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    return parse_french_csv(text)


def fama_french() -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RMW, CMA, RF and Mom."""
    ff = _french(FF5)
    mom = _french(MOM)
    mom.columns = ["Mom"]
    return ff.join(mom, how="inner")
