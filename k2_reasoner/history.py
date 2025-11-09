from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

HISTORY_SCHEMA = {
    "date": "ISO date (YYYY-MM-DD)",
    "pnl": "Daily P&L in base currency",
    "rates_change_bps": "Rate move in basis points",
    "credit_change_bps": "Credit spread change in basis points",
    "fx_change_pct": "FX percentage move",
    "equity_return_pct": "Equity index return in percent",
}


@dataclass
class HistoricalSeries:
    raw: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self.raw.copy()

    @property
    def latest(self) -> pd.Series:
        return self.raw.iloc[-1]


def load_history(uploaded_file: Optional[object] = None, sample_path: Optional[Path] = None) -> HistoricalSeries:
    if uploaded_file is None and sample_path is None:
        raise ValueError("Either uploaded_file or sample_path must be provided.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(sample_path)  # type: ignore[arg-type]

    missing = [col for col in HISTORY_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(f"Historical file missing required columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = [col for col in HISTORY_SCHEMA if col != "date"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Historical data file contained no rows.")
    return HistoricalSeries(raw=df)


def summarize_history(history: HistoricalSeries) -> dict[str, pd.Series]:
    df = history.df
    if df.empty:
        raise ValueError("Historical dataset is empty; cannot summarize.")
    worst = df.nsmallest(1, "pnl").iloc[0]
    best = df.nlargest(1, "pnl").iloc[0]
    latest = df.iloc[-1]
    return {"latest": latest, "worst": worst, "best": best}
