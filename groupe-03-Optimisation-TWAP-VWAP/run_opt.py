from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule

Q = 200
volumes = [10, 40, 200, 100, 100]

print("\n--------------------------------------------------------------------------")

print("=== CP-SAT OPT (impact only) ===")
print(constrained_opt_cp_schedule(
    Q=Q,
    volumes=volumes,
    participation_rate=0.5,
    w_impact=5,
    w_track=0
))

print("\n=== CP-SAT OPT (impact + track VWAP) ===")
print(constrained_opt_cp_schedule(
    Q=Q,
    volumes=volumes,
    participation_rate=0.5,
    w_impact=1,
    w_track=1
))

