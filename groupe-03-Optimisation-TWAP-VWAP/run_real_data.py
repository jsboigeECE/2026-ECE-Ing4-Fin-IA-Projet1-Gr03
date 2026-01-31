import matplotlib.pyplot as plt

from src.data.market_data import load_recent_intraday
from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


# ---------- Utils VWAP target (doit matcher la logique de ton projet) ----------
def vwap_target(Q, volumes):
    total = sum(volumes)
    if total <= 0:
        # fallback : uniform
        base = [Q // len(volumes)] * len(volumes)
        for i in range(Q - sum(base)):
            base[i] += 1
        return base

    raw = [Q * v / total for v in volumes]
    base = [int(x) for x in raw]
    diff = Q - sum(base)

    fracs = sorted([(raw[i] - base[i], i) for i in range(len(volumes))], reverse=True)
    for k in range(abs(diff)):
        idx = fracs[k % len(volumes)][1]
        base[idx] += 1 if diff > 0 else -1

    return base


# ---------- Cumulative costs ----------
def cumulative_impact(x):
    out = []
    s = 0
    for v in x:
        s += v * v
        out.append(s)
    return out


def cumulative_tracking(x, target):
    out = []
    s = 0
    for i in range(len(x)):
        s += (x[i] - target[i]) ** 2
        out.append(s)
    return out


# ---------- Plot ----------
def plot_results(twap, vwap, opt, target, symbol):
    t = list(range(1, len(twap) + 1))

    plt.figure(figsize=(12, 5))

    # --- Impact ---
    plt.subplot(1, 2, 1)
    plt.plot(t, cumulative_impact(twap), label="TWAP")
    plt.plot(t, cumulative_impact(vwap), label="VWAP")
    plt.plot(t, cumulative_impact(opt), label="OPT (CSP)")
    plt.title(f"Cumulative Market Impact ({symbol})")
    plt.xlabel("Time slice")
    plt.ylabel("Impact cost (sum x_t^2)")
    plt.legend()
    plt.grid(True)

    # --- Tracking ---
    plt.subplot(1, 2, 2)
    plt.plot(t, cumulative_tracking(twap, target), label="TWAP")
    plt.plot(t, cumulative_tracking(vwap, target), label="VWAP")
    plt.plot(t, cumulative_tracking(opt, target), label="OPT (CSP)")
    plt.title(f"Cumulative VWAP Tracking Error ({symbol})")
    plt.xlabel("Time slice")
    plt.ylabel("Tracking error (sum (x_t-target_t)^2)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Option: save as png (useful for slides/report)
    plt.savefig("execution_comparison.png", dpi=150)

    plt.show()


def main():
    symbol = "TSLA"      # change ici: "TSLA", "MSFT", "AIR.PA"
    Q = 3000            # volume total à exécuter
    last_n = 20          # nombre de minutes/barres utilisées = N

    print("\n--------------------------------------------------------------------------")
    print(f"Downloading {symbol} data from yfinance...")

    prices, volumes = load_recent_intraday(symbol, interval="1m", period="1d", last_n=last_n)

    N = len(volumes)
    print("Symbol:", symbol)
    print("N (bars):", N)
    print("\nLast close:", prices[-1])
    print("Volumes:", volumes)

    # Baselines
    twap = twap_schedule(Q=Q, N=N).slices
    vwap = vwap_schedule(Q=Q, volumes=volumes, participation_rate=1.0).slices

    # Optimisation CSP (CP-SAT)
    opt = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=1.0,
        w_impact=1,
        w_track=20,
    ).slices

    print("\nTWAP:", twap, "sum=", sum(twap))
    print("VWAP:", vwap, "sum=", sum(vwap))
    print("OPT :", opt,  "sum=", sum(opt))
    print("--------------------------------------------------------------------------\n")

    # Target VWAP (pour tracking)
    target = vwap_target(Q, volumes)

    # Plot
    plot_results(twap, vwap, opt, target, symbol)


if __name__ == "__main__":
    main()
