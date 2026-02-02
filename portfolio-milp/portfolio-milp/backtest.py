import numpy as np
import pandas as pd


def wealth_curve(returns: pd.Series) -> pd.Series:
    """Cumulative wealth index starting at 1.0."""
    returns = returns.fillna(0.0)
    return (1.0 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown computed from cumulative wealth."""
    w = wealth_curve(returns)
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return float(dd.min())


def perf_metrics(returns: pd.Series) -> dict:
    """Annualised mean/vol/sharpe + max drawdown (approx)."""
    r = returns.dropna()
    if len(r) == 0:
        return {"mean": 0.0, "vol": 0.0, "sharpe": 0.0, "maxDD": 0.0}

    mu_d = float(r.mean())
    vol_d = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    mu_a = mu_d * 252.0
    vol_a = vol_d * np.sqrt(252.0)
    sharpe = (mu_a / vol_a) if vol_a > 1e-12 else 0.0
    mdd = max_drawdown(r)

    return {"mean": mu_a, "vol": vol_a, "sharpe": sharpe, "maxDD": mdd}
