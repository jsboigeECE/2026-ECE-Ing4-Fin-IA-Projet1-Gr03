from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class ScheduleResult:
    slices: List[int]
    total: int
    feasible: bool
    message: str


def constrained_opt_cp_schedule(
    Q: int,
    volumes: List[int],
    min_per_slice: int = 0,
    max_per_slice: Optional[int] = None,
    participation_rate: Optional[float] = None,
    caps: Optional[List[int]] = None,
    w_impact: int = 1,
    w_track: int = 0,
    time_limit_s: float = 5.0,
) -> ScheduleResult:
    if Q < 0:
        raise ValueError("Q must be >= 0")
    if min_per_slice < 0:
        raise ValueError("min_per_slice must be >= 0")
    if len(volumes) == 0:
        raise ValueError("volumes must be non-empty")
    if w_impact < 0 or w_track < 0:
        raise ValueError("weights must be >= 0")

    N = len(volumes)

    # caps
    caps_vec = [10**9] * N
    if participation_rate is not None:
        if not (0 < participation_rate <= 1):
            raise ValueError("participation_rate must be in (0, 1]")
        for i in range(N):
            caps_vec[i] = min(caps_vec[i], int(participation_rate * volumes[i]))

    if max_per_slice is not None:
        if max_per_slice < 0:
            raise ValueError("max_per_slice must be >= 0")
        for i in range(N):
            caps_vec[i] = min(caps_vec[i], max_per_slice)

    if caps is not None:
        if len(caps) != N:
            raise ValueError("caps must have length == len(volumes)")
        caps_vec = caps[:]

    min_total = N * min_per_slice
    max_total = sum(caps_vec)

    if Q < min_total:
        return ScheduleResult([min_per_slice] * N, min_total, False,
                              f"Infeasible: Q={Q} < N*min_per_slice={min_total}. Returning minimum schedule.")
    if Q > max_total:
        return ScheduleResult(caps_vec, max_total, False,
                              f"Infeasible: Q={Q} > sum(caps)={max_total}. Returning capped schedule.")

    model = cp_model.CpModel()

    x = [model.NewIntVar(min_per_slice, caps_vec[i], f"x_{i}") for i in range(N)]
    model.Add(sum(x) == Q)

    obj_terms = []

    # impact term: sum(x^2)
    if w_impact > 0:
        for i in range(N):
            sq = model.NewIntVar(0, caps_vec[i] * caps_vec[i], f"x2_{i}")
            model.AddMultiplicationEquality(sq, [x[i], x[i]])
            obj_terms.append(w_impact * sq)

    # tracking term to VWAP target: sum((x - target)^2)
    if w_track > 0:
        total_vol = sum(volumes)
        if total_vol <= 0:
            targets = [Q // N] * N
        else:
            raw = [Q * volumes[i] / total_vol for i in range(N)]
            targets = [int(r) for r in raw]
            diff = Q - sum(targets)
            fracs = [(raw[i] - int(raw[i]), i) for i in range(N)]
            fracs.sort(reverse=True)
            for k in range(abs(diff)):
                idx = fracs[k][1]
                targets[idx] += 1 if diff > 0 else -1

        for i in range(N):
            d = model.NewIntVar(-caps_vec[i], caps_vec[i], f"d_{i}")
            model.Add(d == x[i] - targets[i])
            dsq = model.NewIntVar(0, caps_vec[i] * caps_vec[i], f"dsq_{i}")
            model.AddMultiplicationEquality(dsq, [d, d])
            obj_terms.append(w_track * dsq)

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ScheduleResult([], 0, False, "Solver failed to find a feasible solution")

    slices = [int(solver.Value(x[i])) for i in range(N)]
    return ScheduleResult(slices, sum(slices), True, "OK")
