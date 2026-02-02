import numpy as np


def bootstrap_scenarios(train_returns: np.ndarray, n_scenarios: int = 250, seed: int = 42) -> np.ndarray:
    """
    Build scenario matrix by bootstrapping historical daily returns from the training window.

    train_returns: shape (T, n_assets)
    output       : shape (S, n_assets)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(train_returns, dtype=float)
    T, n = X.shape
    if T < 5:
        raise ValueError("Not enough training returns to bootstrap scenarios.")
    idx = rng.integers(low=0, high=T, size=n_scenarios)
    return X[idx, :].copy()
