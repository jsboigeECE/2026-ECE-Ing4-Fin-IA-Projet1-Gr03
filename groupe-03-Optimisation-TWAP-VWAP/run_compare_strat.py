from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def impact_cost(x):
    # proxy simple d'impact : somme des carrés
    return sum(v * v for v in x)

def tracking_cost(x, target):
    # proxy de "coller au VWAP": somme des erreurs au carré
    return sum((x[i] - target[i]) ** 2 for i in range(len(x)))

def vwap_target(Q, volumes):
    tot = sum(volumes)
    raw = [Q * v / tot for v in volumes]
    base = [int(r) for r in raw]
    diff = Q - sum(base)
    fracs = sorted([(raw[i] - int(raw[i]), i) for i in range(len(volumes))], reverse=True)
    for k in range(abs(diff)):
        idx = fracs[k][1]
        base[idx] += 1 if diff > 0 else -1
    return base


def main():
#--------------------------------------------------------------
    ##PARAMETRES
    Q = 300
    volumes = [10, 40, 200, 100, 100]
    N = len(volumes)

    # Contraintes communes (pour comparaison "fair")
    participation_rate = 1.0  # mets 0.5 si tu veux des caps plus stricts
    max_per_slice = None
#------------------------------------------------------------------------
    # Baselines
    twap = twap_schedule(Q=Q, N=N, max_per_slice=max_per_slice).slices
    vwap = vwap_schedule(Q=Q, volumes=volumes, participation_rate=participation_rate, max_per_slice=max_per_slice).slices

    # Optimisation CP-SAT
    opt_impact_only = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=participation_rate,
        w_impact=1,
        w_track=0
    ).slices

    opt_track = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=participation_rate,
        w_impact=1,
        w_track=20
    ).slices

    target = vwap_target(Q, volumes)
    print("\n--------------------------------------------------------------------------")
    print("Running Comparaison des 3 strategies:")
    print("Q =", Q)
    print("volumes =", volumes)
    print("VWAP target (rounded) =", target)
    print()
    
    rows = [
        ("TWAP", twap),
        ("VWAP", vwap),
        ("OPT impact-only", opt_impact_only),
        ("OPT impact+track", opt_track),
       
    ]
    
    for name, x in rows:
        print(f"{name:16} -> {x}  | sum={sum(x)}  impact={impact_cost(x)}  track={tracking_cost(x, target)}")


if __name__ == "__main__":
    main()
