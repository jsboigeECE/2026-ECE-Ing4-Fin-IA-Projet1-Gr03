import numpy as np

from .parametres import Parametres
from .simulation import SimulateurMarche
from .strategies import StrategieSpreadFixe, StrategieSkewInventaire

def backtest(strategy, p: Parametres):
    env = SimulateurMarche(p)
    env.reset()

    pnls = []
    inventories = []
    execs = 0

    for _ in range(p.T):
        out = strategy.quote(env.etat.S, env.etat.q)
        etat, pnl, executed = env.step(out.bid, out.ask)

        pnls.append(pnl)
        inventories.append(etat.q)
        execs += int(executed)

    pnls = np.array(pnls, dtype=float)
    inventories = np.array(inventories, dtype=int)
    inv_abs_mean = float(np.mean(np.abs(inventories)))
    time_at_bound = float(np.mean(np.abs(inventories) == p.q_max))

    metrics = {
        "PnL_final": float(pnls[-1]),
        "PnL_moyen": float(pnls.mean()),
        "PnL_volatilite": float(pnls.std()),
        "inventaire_max_abs": int(np.max(np.abs(inventories))),
        "nb_executions": int(execs),
        "inventaire_abs_moyen": inv_abs_mean,
        "ratio_temps_a_la_borne": time_at_bound,

    }
    return metrics

def main():
    p = Parametres()

    # Baseline : spread constant
    strat_base = StrategieSpreadFixe(spread=1.0)

    # Stratégie skew : spread de base + gamma (force du retour vers q=0)
    strat_skew = StrategieSkewInventaire(base_spread=1.0, gamma=0.05)

    res_base = backtest(strat_base, p)
    res_skew = backtest(strat_skew, p)

    print("\n=== Résultats ===")
    print("\n--- Baseline (Spread fixe) ---")
    for k, v in res_base.items():
        print(f"{k}: {v}")

    print("\n--- Stratégie inventaire (Skew) ---")
    for k, v in res_skew.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

