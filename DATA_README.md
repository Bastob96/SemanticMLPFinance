# Data Documentation

## Overview

This dataset contains daily market data for AAPL, MSFT, NVDA, and QQQ. It is intended for subsequent feature construction, target creation, and modelling experiments.

## Data Source

- Source: Yahoo Finance
- Download tool: Python `yfinance`
- Download interval: Daily (`1d`)
- Requested period: 2021-08-31 to 2026-09-01
- Price adjustment: `auto_adjust=True`
- No API key is required.

Because `auto_adjust=True` was used, the Open, High, Low, and Close columns contain adjusted prices.

## Columns

The raw and processed datasets use the following columns:

- `Ticker`: Instrument symbol
- `Date`: Trading date in `YYYY-MM-DD` format
- `Open`: Adjusted opening price
- `High`: Adjusted highest price
- `Low`: Adjusted lowest price
- `Close`: Adjusted closing price
- `Volume`: Daily trading volume

## Raw Data

The four raw files are:

- `AAPL_daily_5y.csv`
- `MSFT_daily_5y.csv`
- `NVDA_daily_5y.csv`
- `QQQ_daily_5y.csv`

The raw files have been preserved without modification. Each raw file contains 1,254 rows covering 2021-08-31 to 2026-08-28.

## Data Quality Checks

The following checks were performed:

- Missing-value check
- Duplicate ticker-date check
- OHLC consistency check
- Chronological sorting
- Common trading-date alignment across all four instruments
- Column consistency check

No duplicate ticker-date records or invalid OHLC relationships were found.

## Removed Records

Four incomplete records dated 2026-08-28 were removed from the processed dataset:

- AAPL — 2026-08-28
- MSFT — 2026-08-28
- NVDA — 2026-08-28
- QQQ — 2026-08-28

These records contained Volume values, but Open, High, Low, and Close were all missing. The values were not imputed because doing so would create artificial price data.

The original records remain unchanged in the raw files.

## Final Processed Dataset

File:

`data/processed/aligned_daily_ohlcv_5y.csv`

Final dataset details:

- Date range: 2021-08-31 to 2026-08-27
- Rows per instrument: 1,253
- Total rows: 5,012
- Missing values: 0
- Duplicate ticker-date rows: 0
- All four instruments use the same 1,253 common trading dates
- Data is sorted chronologically by Date and Ticker

This processed dataset should be used as the shared input for the next feature and modelling stage.

## Reproducibility

The accompanying Jupyter Notebook contains the download, validation, cleaning, alignment, and export process. No API key or credential is stored in the notebook.

## 2025 Quarterly Split Dataset

A separate one-year dataset was created from the cleaned five-year dataset.
The original five-year dataset and the four raw files remain unchanged.

### File

`data/processed/aligned_daily_ohlcv_2025_split.csv`

### Dataset details

- Instruments: AAPL, MSFT, NVDA and QQQ
- Date range: 2025-01-02 to 2025-12-31
- Common trading dates per instrument: 250
- Rows per instrument: 250
- Total rows: 1,000
- Missing values: 0
- Duplicate ticker-date rows: 0
- Invalid OHLC rows: 0
- Dates without all four instruments: 0

### Columns

- `Ticker`
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Quarter`
- `Split`

### Chronological split

- Q1 and Q2: training set
- Q3: validation set
- Q4: test set

The data was not randomly shuffled. The chronological split preserves time
order and helps reduce the risk of future information leakage.

This file contains cleaned OHLCV data and split labels. It does not yet contain
engineered features or the AAPL prediction target.
