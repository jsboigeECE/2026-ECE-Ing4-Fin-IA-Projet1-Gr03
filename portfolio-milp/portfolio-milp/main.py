import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import download_prices
from backtest import perf_metrics, wealth_curve
from scenarios import bootstrap_scenarios
from optimize_cpsat import cpsat_cvar_portfolio
from optimize_cvxpy import markowitz_with_turnover


# ----------------------------
# Logging / Noise control
# ----------------------------
def silence_cvxpy_logs():
    """
    CVXPY prints noisy messages about OR-Tools versions (GLOP/PDLP).
    This is harmless but pollutes the terminal.
    We keep errors visible, but hide INFO/WARN logs from CVXPY.
    """
    try:
        logging.getLogger("cvxpy").setLevel(logging.ERROR)
        logging.getLogger("CVXPY").setLevel(logging.ERROR)
    except Exception:
        pass


# ----------------------------
# Terminal diagnostics
# ----------------------------
def print_data_report(prices: pd.DataFrame, rets: pd.DataFrame, tickers: list[str], title: str = "DATA REPORT"):
    print("\n" + "=" * 98)
    print(title)
    print("=" * 98)

    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"Prices shape: {prices.shape}  (rows=dates, cols=assets)")

    if len(prices.index) > 0:
        print(f"Date range : {prices.index.min().date()} -> {prices.index.max().date()}")
        print(f"Nb trading days (prices): {len(prices):,}")

    # Missing values
    na_counts = prices.isna().sum().sort_values(ascending=False)
    total_cells = prices.shape[0] * prices.shape[1]
    total_na = int(na_counts.sum())
    if total_na > 0:
        pct_na = 100.0 * total_na / max(1, total_cells)
        print(f"\nMissing values (prices): {total_na} cells ({pct_na:.3f}%) ❌")
        print("Missing per asset (top):")
        for k, v in na_counts.head(10).items():
            if v > 0:
                print(f"- {k}: {int(v)}")
    else:
        print("\nMissing values (prices): none ✅")

    # Quick sanity checks
    print("\nSample prices (first 2 rows):")
    print(prices.head(2).round(2).to_string())
    print("\nSample prices (last 2 rows):")
    print(prices.tail(2).round(2).to_string())

    # Returns
    print(f"\nReturns shape: {rets.shape}")
    if len(rets.index) > 0:
        print(f"Returns range: {rets.index.min().date()} -> {rets.index.max().date()}")
        print(f"Nb trading days (returns): {len(rets):,}")

    # Annualised stats
    mu_d = rets.mean()
    sig_d = rets.std()
    mu_a = mu_d * 252.0
    sig_a = sig_d * np.sqrt(252.0)
    stats = pd.DataFrame({"mu_ann": mu_a, "vol_ann": sig_a}).sort_values("vol_ann", ascending=False)

    print("\nAnnualised stats (approx):")
    print(stats.round(4).to_string())

    # Correlations
    if rets.shape[1] >= 2:
        corr = rets.corr()
        c = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(ascending=False)
        print("\nTop correlations:")
        top = c.head(5)
        for (a, b), val in top.items():
            print(f"- Corr({a},{b}) = {val:.3f}")

    print("=" * 98 + "\n")


def plot_oos_performance(dates, ew_curve, cps_curve, outpath="performance_oos.png", show=True):
    """
    Plot only Equal-weight vs CP-SAT (remove CVXPY curve).
    Curves are cumulative wealth indices (start at 1.0).
    """
    plt.figure(figsize=(11, 5.5))
    plt.plot(dates, ew_curve, label="Equal-weight")
    plt.plot(dates, cps_curve, label="CP-SAT CVaR (lots + K + sector + TC)")
    plt.title("OOS Walk-forward performance (cumulative)")
    plt.xlabel("Date")
    plt.ylabel("Wealth (start=1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(outpath, dpi=170)
    print(f"Saved figure: {os.path.abspath(outpath)}")

    if show:
        # block=True => the window stays open until you close it
        plt.show(block=True)
    else:
        plt.close()


# ----------------------------
# MAIN
# ----------------------------
def main():
    silence_cvxpy_logs()
    print("=== MAIN.PY (Full #40 project: scenarios -> CP-SAT CVaR + constraints, OOS walk-forward) ===")

    # ---- Universe
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]

    # Sector labels (demo-level but sufficient)
    sector_map = {
        "AAPL": "TECH",
        "MSFT": "TECH",
        "AMZN": "CONS",
        "GOOGL": "TECH",
        "META": "TECH",
        "NVDA": "TECH",
        "JPM": "FIN",
        "XOM": "ENERGY",
        "JNJ": "HEALTH",
        "PG": "CONS",
    }
    sectors = [sector_map.get(t, "OTHER") for t in tickers]

    # Sector bounds (weights)
    sector_bounds = {
        "TECH": (0.00, 0.70),
        "FIN": (0.00, 0.30),
        "ENERGY": (0.00, 0.30),
        "HEALTH": (0.00, 0.30),
        "CONS": (0.00, 0.50),
        "OTHER": (0.00, 0.30),
    }

    # ---- Data range
    start = "2018-01-01"

    # ---- Download prices and compute returns
    prices = download_prices(tickers, start=start)

    # Important: keep only columns in tickers order (defensive)
    prices = prices.loc[:, [t for t in tickers if t in prices.columns]].copy()

    # Returns: drop rows with any missing values to keep alignment
    rets = prices.pct_change().dropna(how="any")

    # Terminal report on data
    print_data_report(prices, rets, tickers, title="DATA REPORT (yfinance download)")

    # ---- Walk-forward setup (OOS)
    train_len = 252 * 2
    test_len = 21
    step = test_len

    # ---- CP-SAT / Real constraints
    Q = 1000
    K = 5
    q_min = 1
    w_max = 0.60
    tc_per_lot = 5
    beta = 0.95
    lambda_cvar = 1.0
    cps_time_limit = 10.0

    # ---- CVXPY baseline (kept for logs/metrics; not plotted)
    cvx_lam = 0.3
    cvx_gamma_tc = 0.05
    cvx_w_max = w_max

    print("Walk-forward config:")
    print(f"- train_len={train_len} days, test_len={test_len} days, step={step}")
    print("Constraints:")
    print(f"- lots Q={Q}, cardinality K={K}, q_min={q_min}, w_max={w_max}")
    print(f"- turnover cost tc_per_lot={tc_per_lot}, CVaR beta={beta}, lambda={lambda_cvar}\n")

    if len(rets) < train_len + test_len + 5:
        raise ValueError("Not enough data for the chosen train/test lengths. Increase history or reduce train_len.")

    # ---- Storage OOS returns
    oos_dates = []
    oos_ret_ew = []
    oos_ret_cvx = []
    oos_ret_cps = []

    # Diagnostics
    turnovers_cvx = []
    turnovers_cps_lots = []
    selected_cps = []
    statuses = []

    # Initial portfolios
    n = len(tickers)
    w_ew = np.ones(n) / n
    w_cvx_prev = w_ew.copy()
    q_old = np.zeros(n, dtype=int)
    w_cps_prev = w_ew.copy()

    # ---- Walk-forward loop
    for start_i in range(0, len(rets) - train_len - test_len + 1, step):
        train = rets.iloc[start_i: start_i + train_len]
        test = rets.iloc[start_i + train_len: start_i + train_len + test_len]

        mu = train.mean().values
        Sigma = train.cov().values

        scenarios = bootstrap_scenarios(train.values, n_scenarios=min(250, len(train)), seed=42)

        # ---- CVXPY baseline (kept for metrics)
        w_cvx, _ = markowitz_with_turnover(
            mu=mu,
            Sigma=Sigma,
            w_old=w_cvx_prev,
            lam=cvx_lam,
            gamma_tc=cvx_gamma_tc,
            w_max=cvx_w_max,
        )
        turnover_cvx = float(np.sum(np.abs(w_cvx - w_cvx_prev)))
        turnovers_cvx.append(turnover_cvx)
        w_cvx_prev = w_cvx.copy()

        # ---- CP-SAT CVaR (real constraints)
        w_cps_new, q_sol, status = cpsat_cvar_portfolio(
            scenarios_returns=scenarios,
            mu=mu,
            sectors=sectors,
            sector_bounds=sector_bounds,
            q_old=q_old,
            Q=Q,
            K=K,
            q_min=q_min,
            w_max=w_max,
            tc_per_lot=tc_per_lot,
            beta=beta,
            lambda_cvar=lambda_cvar,
            time_limit_s=cps_time_limit,
            seed=42,
        )
        statuses.append(str(status))

        if w_cps_new is None or q_sol is None:
            w_cps_new = w_cps_prev.copy()
            q_sol = q_old.copy()

        turnover_lots = float(np.sum(np.abs(q_sol - q_old)))
        turnovers_cps_lots.append(turnover_lots)
        q_old = q_sol.copy()
        w_cps_prev = w_cps_new.copy()

        selected_cps.append(int(np.sum(q_sol > 0)))

        # ---- Apply to test period
        for dt, row in test.iterrows():
            rvec = row.values.astype(float)
            oos_dates.append(dt)
            oos_ret_ew.append(float(np.dot(w_ew, rvec)))
            oos_ret_cvx.append(float(np.dot(w_cvx, rvec)))
            oos_ret_cps.append(float(np.dot(w_cps_new, rvec)))

    # ---- Build series
    oos_dates = pd.to_datetime(oos_dates)
    s_ew = pd.Series(oos_ret_ew, index=oos_dates, name="EW")
    s_cvx = pd.Series(oos_ret_cvx, index=oos_dates, name="CVXPY")
    s_cps = pd.Series(oos_ret_cps, index=oos_dates, name="CP-SAT")

    # ---- Performance
    ew_m = perf_metrics(s_ew)
    cvx_m = perf_metrics(s_cvx)
    cps_m = perf_metrics(s_cps)

    print("\n=== OOS Summary (annualised approx) ===")
    print(f"Equal-weight : mean={ew_m['mean']:.3f} vol={ew_m['vol']:.3f} sharpe={ew_m['sharpe']:.3f} maxDD={ew_m['maxDD']:.3f}")
    print(f"CVXPY        : mean={cvx_m['mean']:.3f} vol={cvx_m['vol']:.3f} sharpe={cvx_m['sharpe']:.3f} maxDD={cvx_m['maxDD']:.3f}")
    print(f"CP-SAT CVaR  : mean={cps_m['mean']:.3f} vol={cps_m['vol']:.3f} sharpe={cps_m['sharpe']:.3f} maxDD={cps_m['maxDD']:.3f}")

    print("\n=== Trading / Constraints diagnostics ===")
    print(f"Avg turnover CVXPY (L1 weights): {np.mean(turnovers_cvx):.4f}")
    print(f"Avg turnover CP-SAT (lots)     : {np.mean(turnovers_cps_lots):.4f}")
    print(f"Avg selected assets CP-SAT     : {np.mean(selected_cps):.2f} (max K={K})")
    print(f"CP-SAT statuses (last 10)      : {statuses[-10:]}")

    # ---- Wealth curves
    w_ew_curve = wealth_curve(s_ew)
    w_cps_curve = wealth_curve(s_cps)

    # ---- Plot (NO CVXPY curve)
    plot_oos_performance(
        dates=w_ew_curve.index,
        ew_curve=w_ew_curve.values,
        cps_curve=w_cps_curve.values,
        outpath="performance_oos.png",
        show=True,
    )

    # ---- Save run outputs
    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_json = {
        "timestamp": ts,
        "tickers": tickers,
        "start": start,
        "train_len": train_len,
        "test_len": test_len,
        "Q": Q,
        "K": K,
        "q_min": q_min,
        "w_max": w_max,
        "tc_per_lot": tc_per_lot,
        "beta": beta,
        "lambda_cvar": lambda_cvar,
        "cvx_lam": cvx_lam,
        "cvx_gamma_tc": cvx_gamma_tc,
        "metrics": {
            "equal_weight": ew_m,
            "cvxpy": cvx_m,
            "cpsat": cps_m,
        },
        "diagnostics": {
            "avg_turnover_cvx_L1": float(np.mean(turnovers_cvx)),
            "avg_turnover_cps_lots": float(np.mean(turnovers_cps_lots)),
            "avg_selected_assets_cps": float(np.mean(selected_cps)),
            "statuses_tail": statuses[-10:],
        },
    }

    run_path = os.path.join("runs", f"run_{ts}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2)

    metrics_df = pd.DataFrame(
        [
            {"strategy": "equal_weight", **ew_m},
            {"strategy": "cvxpy", **cvx_m},
            {"strategy": "cpsat_cvar", **cps_m},
        ]
    )
    metrics_path = os.path.join("runs", f"metrics_{ts}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\nSaved run json: {os.path.abspath(run_path)}")
    print(f"Saved metrics : {os.path.abspath(metrics_path)}")


if __name__ == "__main__":
    main()
