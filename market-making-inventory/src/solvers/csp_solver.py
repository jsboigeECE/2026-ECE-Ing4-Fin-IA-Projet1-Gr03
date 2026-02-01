"""
CSP Solver for Optimal Market Making

This module implements a Constraint Satisfaction Problem (CSP) formulation
for optimal market making using OR-Tools.

The CSP formulation treats the market making problem as an optimization
problem with explicit constraints on inventory, risk, and spreads.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from ortools.sat.python import cp_model


@dataclass
class CSPSolverConfig:
    """Configuration for CSP solver."""
    
    # Time parameters
    T: float = 1.0              # Time horizon
    dt: float = 0.01            # Time step
    
    # Inventory constraints
    Q_max: int = 10             # Maximum inventory (absolute)
    
    # Spread bounds
    delta_min: int = 1           # Minimum half-spread (in ticks)
    delta_max: int = 100         # Maximum half-spread (in ticks)
    
    # Risk constraints
    VaR_max: float = 5.0         # Maximum Value at Risk
    alpha: float = 0.95          # VaR confidence level
    
    # Drawdown constraint
    max_drawdown: float = 0.1     # Maximum drawdown (as fraction)
    
    # Market parameters
    sigma: float = 0.2           # Volatility
    S0: float = 100.0           # Initial price
    
    # Order flow parameters
    A: float = 1.0              # Base intensity
    k: float = 0.5              # Market depth
    
    # Solver parameters
    max_time: int = 60           # Maximum solve time (seconds)
    num_solutions: int = 1       # Number of solutions to find


class CSPMarketMakingSolver:
    """
    Constraint Satisfaction Problem solver for market making.
    
    This solver formulates the market making problem as a CSP
    and uses OR-Tools to find optimal quoting strategies.
    """
    
    def __init__(self, config: CSPSolverConfig):
        """
        Initialize CSP solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        
        # Discretization
        self.N = int(config.T / config.dt)
        self.time_grid = np.linspace(0, config.T, self.N + 1)
        
        # Create CP-SAT model
        self.model = cp_model.CpModel()
        
        # Decision variables
        self.delta_b: List[cp_model.IntVar] = []
        self.delta_a: List[cp_model.IntVar] = []
        self.inventory: List[cp_model.IntVar] = []
        self.buy: List[cp_model.IntVar] = []
        self.sell: List[cp_model.IntVar] = []
        
        # Build model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the CSP model."""
        # Create decision variables
        for t in range(self.N):
            # Spreads (in ticks)
            delta_b_t = self.model.NewIntVar(
                self.config.delta_min,
                self.config.delta_max,
                f'delta_b_{t}'
            )
            delta_a_t = self.model.NewIntVar(
                self.config.delta_min,
                self.config.delta_max,
                f'delta_a_{t}'
            )
            
            self.delta_b.append(delta_b_t)
            self.delta_a.append(delta_a_t)
            
            # Buy/sell indicators (0 or 1)
            buy_t = self.model.NewIntVar(0, 1, f'buy_{t}')
            sell_t = self.model.NewIntVar(0, 1, f'sell_{t}')
            
            self.buy.append(buy_t)
            self.sell.append(sell_t)
        
        # Inventory variables (including initial and terminal)
        for t in range(self.N + 1):
            q_t = self.model.NewIntVar(
                -self.config.Q_max,
                self.config.Q_max,
                f'q_{t}'
            )
            self.inventory.append(q_t)
        
        # Add constraints
        self._add_inventory_constraints()
        self._add_spread_constraints()
        self._add_risk_constraints()
        self._add_objective()
    
    def _add_inventory_constraints(self) -> None:
        """Add inventory-related constraints."""
        # Initial inventory
        self.model.Add(self.inventory[0] == 0)
        
        # Inventory dynamics
        for t in range(self.N):
            # q_{t+1} = q_t + buy_t - sell_t
            self.model.Add(
                self.inventory[t + 1] == self.inventory[t] + self.buy[t] - self.sell[t]
            )
            
            # Cannot buy and sell at the same time
            self.model.Add(self.buy[t] + self.sell[t] <= 1)
            
            # Inventory bounds
            self.model.Add(self.inventory[t] >= -self.config.Q_max)
            self.model.Add(self.inventory[t] <= self.config.Q_max)
        
        # Terminal inventory (prefer zero)
        self.model.Add(self.inventory[self.N] == 0)
    
    def _add_spread_constraints(self) -> None:
        """Add spread-related constraints."""
        for t in range(self.N):
            # Ask must be >= bid
            self.model.Add(self.delta_a[t] >= self.delta_b[t])
            
            # Minimum spread
            min_spread = self.config.delta_min * 2
            self.model.Add(self.delta_a[t] - self.delta_b[t] >= min_spread)
    
    def _add_risk_constraints(self) -> None:
        """Add risk-related constraints."""
        # VaR constraint (simplified)
        # For each time step, ensure inventory doesn't exceed VaR limit
        for t in range(self.N + 1):
            # VaR = |q| * sigma * sqrt(T-t) * z_alpha
            tau = self.config.T - self.time_grid[t]
            if tau > 0:
                from scipy.stats import norm
                z_alpha = norm.ppf(self.config.alpha)
                max_q_for_var = int(self.config.VaR_max / (self.config.sigma * np.sqrt(tau) * z_alpha))
                
                # Constrain inventory based on VaR
                self.model.Add(self.inventory[t] <= max_q_for_var)
                self.model.Add(self.inventory[t] >= -max_q_for_var)
    
    def _add_objective(self) -> None:
        """Add objective function."""
        # Objective: maximize expected profit
        # Expected profit = sum of spreads * probability of fill
        
        # Simplified objective: maximize sum of spreads
        total_spread = sum(self.delta_a[t] - self.delta_b[t] for t in range(self.N))
        
        # Penalize large inventory
        inventory_penalty = sum(
            self.inventory[t] * self.inventory[t] for t in range(self.N + 1)
        )
        
        # Objective: maximize spread - penalize inventory
        self.model.Maximize(total_spread - 0.1 * inventory_penalty)
    
    def solve(self) -> Optional[Dict]:
        """
        Solve the CSP problem.
        
        Returns:
            Dictionary with solution or None if no solution found
        """
        # Create solver
        solver = cp_model.CpSolver()
        
        # Set solver parameters
        solver.parameters.max_time_in_seconds = self.config.max_time
        solver.parameters.num_search_workers = 8
        
        # Solve
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract solution
            solution = {
                'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
                'objective': solver.ObjectiveValue(),
                'delta_b': [solver.Value(self.delta_b[t]) for t in range(self.N)],
                'delta_a': [solver.Value(self.delta_a[t]) for t in range(self.N)],
                'inventory': [solver.Value(self.inventory[t]) for t in range(self.N + 1)],
                'buy': [solver.Value(self.buy[t]) for t in range(self.N)],
                'sell': [solver.Value(self.sell[t]) for t in range(self.N)]
            }
            return solution
        else:
            return None
    
    def get_policy(self, solution: Dict) -> callable:
        """
        Get policy function from solution.
        
        Args:
            solution: Solution dictionary from solve()
            
        Returns:
            Function that takes (t, q) and returns (delta_b, delta_a)
        """
        delta_b_sol = solution['delta_b']
        delta_a_sol = solution['delta_a']
        
        def policy(t: float, q: int) -> Tuple[float, float]:
            t_idx = int(t / self.config.dt)
            t_idx = min(t_idx, self.N - 1)
            
            # Convert ticks to price units
            tick_size = 0.01  # Assume 1 tick = 0.01 price units
            delta_b = delta_b_sol[t_idx] * tick_size
            delta_a = delta_a_sol[t_idx] * tick_size
            
            return delta_b, delta_a
        
        return policy


class SimplifiedCSPSolver:
    """
    Simplified CSP solver for faster computation.
    
    This solver uses a simplified formulation that focuses on
    inventory constraints and spread optimization.
    """
    
    def __init__(self, config: CSPSolverConfig):
        """
        Initialize simplified CSP solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        self.N = int(config.T / config.dt)
        self.model = cp_model.CpModel()
        
        # Variables
        self.delta_b: List[cp_model.IntVar] = []
        self.delta_a: List[cp_model.IntVar] = []
        self.inventory: List[cp_model.IntVar] = []
        
        # Build model
        self._build_model()
    
    def _build_model(self) -> None:
        """Build simplified CSP model."""
        # Create variables
        for t in range(self.N):
            delta_b_t = self.model.NewIntVar(
                self.config.delta_min,
                self.config.delta_max,
                f'delta_b_{t}'
            )
            delta_a_t = self.model.NewIntVar(
                self.config.delta_min,
                self.config.delta_max,
                f'delta_a_{t}'
            )
            
            self.delta_b.append(delta_b_t)
            self.delta_a.append(delta_a_t)
        
        for t in range(self.N + 1):
            q_t = self.model.NewIntVar(
                -self.config.Q_max,
                self.config.Q_max,
                f'q_{t}'
            )
            self.inventory.append(q_t)
        
        # Add constraints
        self._add_constraints()
        self._add_objective()
    
    def _add_constraints(self) -> None:
        """Add constraints."""
        # Initial inventory
        self.model.Add(self.inventory[0] == 0)
        
        # Terminal inventory
        self.model.Add(self.inventory[self.N] == 0)
        
        # Inventory bounds
        for t in range(self.N + 1):
            self.model.Add(self.inventory[t] >= -self.config.Q_max)
            self.model.Add(self.inventory[t] <= self.config.Q_max)
        
        # Spread constraints
        for t in range(self.N):
            self.model.Add(self.delta_a[t] >= self.delta_b[t])
            self.model.Add(self.delta_a[t] - self.delta_b[t] >= self.config.delta_min * 2)
    
    def _add_objective(self) -> None:
        """Add objective."""
        # Maximize total spread
        total_spread = sum(self.delta_a[t] - self.delta_b[t] for t in range(self.N))
        self.model.Maximize(total_spread)
    
    def solve(self) -> Optional[Dict]:
        """
        Solve the simplified CSP problem.
        
        Returns:
            Dictionary with solution or None
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.max_time
        
        status = solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return {
                'status': 'optimal' if status == cp_model.OPTIMAL else 'feasible',
                'objective': solver.ObjectiveValue(),
                'delta_b': [solver.Value(self.delta_b[t]) for t in range(self.N)],
                'delta_a': [solver.Value(self.delta_a[t]) for t in range(self.N)],
                'inventory': [solver.Value(self.inventory[t]) for t in range(self.N + 1)]
            }
        return None


def create_default_csp_solver(
    simplified: bool = True
) -> Tuple[object, Optional[callable]]:
    """
    Create a CSP solver with default configuration.
    
    Args:
        simplified: Whether to use simplified solver
        
    Returns:
        Tuple of (solver, policy_function)
    """
    config = CSPSolverConfig(
        T=1.0,
        dt=0.01,
        Q_max=10,
        delta_min=1,
        delta_max=100,
        VaR_max=5.0,
        alpha=0.95,
        max_drawdown=0.1,
        sigma=0.2,
        S0=100.0,
        A=1.0,
        k=0.5,
        max_time=60,
        num_solutions=1
    )
    
    if simplified:
        solver = SimplifiedCSPSolver(config)
    else:
        solver = CSPMarketMakingSolver(config)
    
    # Solve
    solution = solver.solve()
    
    if solution is None:
        return solver, None
    
    # Get policy
    policy = solver.get_policy(solution)
    
    return solver, policy


if __name__ == "__main__":
    # Example usage
    print("Creating CSP solver...")
    solver, policy = create_default_csp_solver(simplified=True)
    
    if policy is not None:
        print(f"\nSolver configuration:")
        print(f"  Time horizon: {solver.config.T}")
        print(f"  Time steps: {solver.N}")
        print(f"  Max inventory: {solver.config.Q_max}")
        
        # Example: Get optimal spreads
        print(f"\nOptimal spreads at different times:")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            delta_b, delta_a = policy(t, 0)
            print(f"  t={t:.2f}: delta_b={delta_b:.4f}, delta_a={delta_a:.4f}")
    else:
        print("No solution found!")
