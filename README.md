# lazy-prices-10k

Do firms that quietly rewrite their annual reports underperform? This repo tests that idea on S&P 500 10-K filings from SEC EDGAR, following Cohen, Malloy and Nguyen, "Lazy Prices" (Journal of Finance, 2020). They found that firms whose 10-K text changes a lot from one year to the next earn lower future returns, because investors are slow to read the changes.

The pipeline goes from raw filings to a point-in-time, sector-neutral backtest.

## Pipeline

1. **Download** (`edgar.py`). Lists each firm's original 10-K filings through the EDGAR submissions API and downloads the main document. Everything is cached under `cache/`, and requests are throttled under SEC's 10-per-second limit.
2. **Sections** (`sections.py`). Converts the HTML to text, drops hidden inline XBRL facts, and cuts out **Item 1A (Risk Factors)** and **Item 7 (MD&A)**. Headings appear in the table of contents as well as at the real section, so for each candidate heading the code measures the span to the next item heading and keeps the longest.
3. **Pairs** (`build.py`). Compares each 10-K with the firm's previous one, only when the two are 9 to 15 months apart. That way a skipped year or a change of fiscal year is never compared against the wrong document.
4. **Text change features** (`features.py`), per section:
   - `cosine`: cosine similarity of word-count vectors, the paper's main measure
   - `jaccard`: vocabulary overlap
   - `len_change`: log change in word count
   - `new_share`: share of this year's sentences that did not appear last year
   - optional `mdna_tone_change`: change in FinBERT tone (P(positive) − P(negative)) on MD&A (`sentiment.py`)
5. **Signals** (`signal.py`). At each month-end, a firm's value is its latest 10-K filed **strictly before** that date and at most 12 months old. Positions formed at month-end t earn the return of month t+1. Signals are z-scored each month and demeaned within 2-digit SIC industries.
6. **Evaluation** (`backtest.py`):
   - rank IC at 1, 3, 6 and 12 month horizons, with Newey-West t-stats
   - top-minus-bottom quintile long-short, equal weight, rebalanced monthly, net of costs
   - turnover
   - alpha after Fama-French five factors plus momentum
   - net long-short return by year

## Signals tested

Higher always means "expected to outperform":

| Signal | Built from |
| --- | --- |
| `risk_cosine` | Item 1A similarity to last year |
| `mdna_cosine` | Item 7 similarity to last year |
| `risk_new_share (inverted)` | minus the share of new risk-factor sentences |
| `risk_len_change (inverted)` | minus the growth in risk-factor length |
| `combined_similarity` | average z-score of the two cosine signals |
| `mdna_tone_change` | FinBERT tone change, only with `--finbert` |

## Run

```
pip install -e ".[test]"          # add ,finbert for the FinBERT signal
pytest

export SEC_USER_AGENT="Your Name your.email@example.com"   # required by SEC
python -m lazyprices.run --limit 50    # quick check on 50 firms
python -m lazyprices.run               # full S&P 500, 10-Ks filed 2010 to 2024
python -m lazyprices.run --finbert     # adds the tone signal (slow on CPU)
```

The first full run downloads several thousand filings, so expect a few hours. Everything is cached, so reruns take minutes. Use `--rebuild` to recompute the feature table after changing the text code.

Outputs in `results/`:
- `summary.md`: coverage, IC table, long-short table and returns by year
- `filing_features.csv`: one row per filing pair
- `ic_by_horizon.csv`, `long_short.csv`, `long_short_by_year.csv`
- `long_short.png`: net equity curves

## Tests

`pytest` covers:
- section extraction, including the table-of-contents trap and hidden XBRL text
- the text features
- point-in-time alignment: a filing is never used before the month after it is filed, and expires after 12 months
- sector neutralisation
- forward-return alignment
- the backtest: a planted signal is found, a random one is not, and costs lower net returns
- the Fama-French file parser

## Limitations

- **Survivorship bias.** The universe is today's S&P 500 list from Wikipedia, so firms that were delisted or dropped out of the index are missing. This usually flatters long-short results. A clean test needs point-in-time index membership.
- **Yahoo Finance prices** can have gaps for tickers that changed or were delisted.
- **Heading regexes** miss some unusual layouts. `summary.md` reports how often each section was found, so check coverage before trusting a signal.
- **Large caps only.** The paper's effect is strongest in smaller, less-covered stocks, so a weaker result here would be consistent with it.
