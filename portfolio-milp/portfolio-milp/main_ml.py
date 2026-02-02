import numpy as np
import matplotlib.pyplot as plt

from data import download_prices, returns_from_prices
from ml_module import train_ridge_mu, predict_mu
from optimize_cvxpy import markowitz_with_turnover


def main():
    print("=== MAIN_ML.PY (ML mu -> CVXPY optimisation) ===")

    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]
    prices = download_prices(tickers, start="2020-01-01")
    rets = returns_from_prices(prices)
    R = rets.values

    train = R[-252:, :]
    models = train_ridge_mu(train)
    mu_ml = predict_mu(models, train)

    Sigma = np.cov(train.T)
    w, obj = markowitz_with_turnover(mu_ml, Sigma, w_old=None, lam=10.0, gamma_tc=0.2, w_max=0.60)

    w = np.clip(w, 0, 0.60)
    w = w / w.sum()

    print("\nOptimised weights (sorted):")
    for t, wi in sorted(zip(tickers, w), key=lambda x: -x[1]):
        print(f"{t}: {wi:.4f}")

    # quick plot
    pr = (1 + (rets @ w)).cumprod()
    plt.figure()
    pr.plot()
    plt.title("ML->CVXPY portfolio curve")
    plt.tight_layout()
    plt.savefig("performance_ml.png", dpi=150)
    plt.close()
    print("\nSaved: performance_ml.png")


if __name__ == "__main__":
    main()
