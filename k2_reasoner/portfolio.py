from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SAMPLE_COLUMNS: dict[str, str] = {
    "instrument": "Display name for the risk position",
    "asset_class": "Asset bucket (Rates, Credit, Equity, FX, Commodity)",
    "notional": "Exposure in millions",
    "duration": "Interest-rate duration in years",
    "dv01": "DV01 in $ per 1bp move",
    "convexity": "Second-order rate sensitivity",
    "credit_spread_dv01": "PnL change for 1bp move in credit spreads",
    "fx_delta": "FX delta in base currency",
    "beta": "Equity beta vs. MSCI World",
    "commentary": "Optional analyst note",
}


@dataclass
class Portfolio:
    raw: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self.raw.copy()


def _coerce_numeric(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    """Force numeric columns and default missing values to zero."""
    coerced = df.copy()
    for col in numeric_columns:
        coerced[col] = pd.to_numeric(coerced.get(col, 0), errors="coerce").fillna(0.0)
    return coerced


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in SAMPLE_COLUMNS if c not in df.columns]
    if missing:
        columns = ", ".join(missing)
        raise ValueError(f"Portfolio file is missing required columns: {columns}")


def load_portfolio_dataframe(
    uploaded_file: Optional[object] = None, sample_path: Optional[Path] = None
) -> Portfolio:
    """
    Read a CSV file into a Portfolio object.

    Parameters
    ----------
    uploaded_file:
        The file-like object returned by Streamlit's uploader.
    sample_path:
        Optional path to a bundled sample CSV when no file is uploaded.
    """

    if uploaded_file is None and sample_path is None:
        raise ValueError("Either an uploaded file or a sample path must be provided.")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(sample_path)  # type: ignore[arg-type]

    _validate_columns(df)
    numeric_cols = [c for c in SAMPLE_COLUMNS if c not in {"instrument", "asset_class", "commentary"}]
    cleaned = _coerce_numeric(df, numeric_cols)
    cleaned["asset_class"] = cleaned["asset_class"].str.title()

    return Portfolio(raw=cleaned)
