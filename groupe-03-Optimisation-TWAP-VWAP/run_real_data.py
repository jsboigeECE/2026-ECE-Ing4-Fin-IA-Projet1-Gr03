from src.data.market_data import load_recent_intraday
from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def main():
    symbol = "AAPL"      # <- change ici (ex: "TSLA", "MSFT", "AIR.PA" pour Air Liquide)
    Q = 20000              # <- volume total à exécuter
    last_n = 10         # <- nombre de minutes/barres utilisées (donc N)
    print("\n--------------------------------------------------------------------------")
    print(f"Downloading {symbol} data from yfinance...")


    prices, volumes = load_recent_intraday(symbol, interval="1m", period="1d", last_n=last_n)

    N = len(volumes)
    print("Symbol:", symbol)
    print("N (bars):", N)
    print("\n")

    print("Last close:", prices[-1])
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
        w_track=20,   # <- augmente/diminue pour coller + ou - au VWAP
    ).slices
    
    print("\nTWAP:", twap, "sum=", sum(twap))
    print("VWAP:", vwap, "sum=", sum(vwap))
    print("OPT :", opt,  "sum=", sum(opt))
    print("--------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
