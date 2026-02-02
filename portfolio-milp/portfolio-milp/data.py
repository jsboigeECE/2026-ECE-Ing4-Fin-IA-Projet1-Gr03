import pandas as pd
import yfinance as yf


def _ensure_df(x) -> pd.DataFrame:
    """Series -> DataFrame, DataFrame -> DataFrame"""
    if isinstance(x, pd.Series):
        return x.to_frame()
    return x.copy()


def download_prices(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    """
    Download prices from yfinance and return a clean DataFrame:
    - index: DatetimeIndex
    - columns: tickers
    - values: close-like prices (auto_adjust=True so it's adjusted)
    Robust to yfinance returning:
      - single-level columns (OHLCV)
      - MultiIndex columns with levels either (Field, Ticker) or (Ticker, Field)
    """
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned empty data. Check internet/tickers/start date.")

    # Normalize tickers list
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        if "Close" in lvl0:
            close = df["Close"]
        elif "Adj Close" in lvl0:
            close = df["Adj Close"]
        elif "Close" in lvl1:
            close = df.xs("Close", level=1, axis=1)
        elif "Adj Close" in lvl1:
            close = df.xs("Adj Close", level=1, axis=1)
        else:
            raise KeyError("Could not find Close/Adj Close in yfinance MultiIndex columns.")

        close = _ensure_df(close)

        cols = [t for t in tickers_list if t in close.columns]
        if len(cols) == 0:
            return close.dropna(how="all").sort_index()

        close = close.loc[:, cols]
        return close.dropna(how="all").sort_index()

    # Single-level columns
    if isinstance(df.columns, pd.Index):
        if "Close" in df.columns:
            close = _ensure_df(df["Close"])
        elif "Adj Close" in df.columns:
            close = _ensure_df(df["Adj Close"])
        else:
            raise KeyError("Could not find Close/Adj Close in yfinance columns.")

        if close.shape[1] == 1 and len(tickers_list) == 1:
            close.columns = [tickers_list[0]]

        return close.dropna(how="all").sort_index()

    raise RuntimeError("Unexpected yfinance dataframe columns format.")
