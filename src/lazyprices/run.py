"""End-to-end experiment.

    export SEC_USER_AGENT="Your Name your.email@example.com"
    python -m lazyprices.run                      # S&P 500, 10-Ks filed 2010-2024
    python -m lazyprices.run --limit 50           # quick test on 50 firms
    python -m lazyprices.run --finbert            # add the FinBERT tone change signal

Steps:
1. Download each firm's 10-Ks from EDGAR, cut out Item 1A and Item 7, and compare each
   with the prior year's version (cached in cache/, so reruns are fast).
2. Build point-in-time monthly signals from those features.
3. Pull monthly total returns from Yahoo Finance and Fama-French factors.
4. Evaluate each signal: rank IC at 1, 3, 6 and 12 months, sector-neutral quintile
   long-short after costs, turnover, alpha after FF5 + momentum, and results by year.
5. Write results/summary.md, results/*.csv and results/long_short.png.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import factor_alpha, ic_by_horizon, long_short, performance
from .build import build_features
from .market import fama_french, monthly_returns, sp500_tickers
from .signal import combine, sector_neutral, sic2, to_monthly

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# higher value = expected to outperform
SIGNALS = {
    "risk_cosine": ["risk_cosine"],
    "mdna_cosine": ["mdna_cosine"],
    "risk_new_share (inverted)": ["-risk_new_share"],
    "risk_len_change (inverted)": ["-risk_len_change"],
    "combined_similarity": ["risk_cosine", "mdna_cosine"],
}


def _panel(features: pd.DataFrame, col: str, months: pd.DatetimeIndex) -> pd.DataFrame:
    sign = -1.0 if col.startswith("-") else 1.0
    return sign * to_monthly(features, col.lstrip("-"), months)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--limit", type=int, default=None, help="only the first N tickers")
    p.add_argument("--tickers", type=Path, default=None, help="text file, one ticker per line")
    p.add_argument("--finbert", action="store_true")
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--rebuild", action="store_true", help="recompute the feature table")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tickers = (args.tickers.read_text().split() if args.tickers else sp500_tickers())
    tickers = tickers[: args.limit] if args.limit else tickers

    feat_path = RESULTS / "filing_features.csv"
    if feat_path.exists() and not args.rebuild:
        features = pd.read_csv(feat_path)
    else:
        features = build_features(tickers, args.start_year, args.end_year, feat_path,
                                  finbert=args.finbert)
    logging.info("%d filing pairs from %d firms", len(features), features["ticker"].nunique())

    returns = monthly_returns(sorted(features["ticker"].unique()),
                              f"{args.start_year}-01-01", f"{args.end_year + 1}-12-31")
    months = returns.index
    sectors = sic2(features)

    signals = dict(SIGNALS)
    if "mdna_tone_change" in features:
        signals["mdna_tone_change"] = ["mdna_tone_change"]

    try:
        ff = fama_french()
    except Exception as exc:
        logging.warning("Fama-French download failed (%s); skipping alpha", exc)
        ff = None

    ic_rows, perf_rows, yearly, curves = [], [], {}, {}
    for name, cols in signals.items():
        raw = combine([_panel(features, c, months) for c in cols])
        sig = sector_neutral(raw, sectors)
        ic = ic_by_horizon(sig, returns)
        ic.insert(0, "signal", name)
        ic_rows.append(ic)
        ls = long_short(sig, returns, cost_bps=args.cost_bps)
        row = {"signal": name, **{f"net_{k}": v for k, v in performance(ls["net"]).items()},
               "gross_ann_return": ls["gross"].mean() * 12,
               "avg_monthly_turnover": ls["turnover"].mean()}
        if ff is not None:
            row.update(factor_alpha(ls["net"], ff))
        perf_rows.append(row)
        yearly[name] = ls["net"].groupby(ls.index.year).apply(lambda r: (1 + r).prod() - 1)
        curves[name] = (1 + ls["net"]).cumprod()

    ic_tab = pd.concat(ic_rows, ignore_index=True)
    perf = pd.DataFrame(perf_rows)
    yearly = pd.DataFrame(yearly)
    RESULTS.mkdir(exist_ok=True)
    ic_tab.to_csv(RESULTS / "ic_by_horizon.csv", index=False)
    perf.to_csv(RESULTS / "long_short.csv", index=False)
    yearly.to_csv(RESULTS / "long_short_by_year.csv")
    coverage = features[["risk_found", "mdna_found"]].mean()
    write_summary(features, ic_tab, perf, yearly, coverage, args)
    plot(curves)
    print(perf.to_string())


def write_summary(features, ic_tab, perf, yearly, coverage, args) -> None:
    pct = lambda v: f"{v * 100:.2f}%"
    cols = ["signal", "net_ann_return", "net_sharpe", "net_t_stat", "avg_monthly_turnover"]
    cols += [c for c in ("alpha_ann", "alpha_t") if c in perf]
    view = perf[cols].copy()
    for c in ("net_ann_return", "avg_monthly_turnover", "alpha_ann"):
        if c in view:
            view[c] = view[c].map(pct)
    lines = [
        "# Results", "",
        f"{len(features):,} year-over-year 10-K pairs from {features['ticker'].nunique()} firms, "
        f"filed {args.start_year} to {args.end_year}. Item 1A found in {coverage['risk_found']:.0%} "
        f"of pairs, Item 7 in {coverage['mdna_found']:.0%}.", "",
        f"Signals are sector neutral (2-digit SIC). Long-short is top minus bottom quintile, "
        f"equal weight, monthly, net of {args.cost_bps:g} bp per unit traded.", "",
        "## Rank IC by horizon", "",
        ic_tab.to_markdown(index=False, floatfmt=".4f"), "",
        "## Long-short performance", "",
        view.to_markdown(index=False, floatfmt=".2f"), "",
        "## Long-short net return by year", "",
        yearly.to_markdown(floatfmt=".3f"), "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(lines))


def plot(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, c in curves.items():
        ax.plot(c, label=name, linewidth=1.3)
    ax.set_ylabel("Growth of 1 (net)")
    ax.set_title("Sector-neutral quintile long-short")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "long_short.png", dpi=130)


if __name__ == "__main__":
    main()
