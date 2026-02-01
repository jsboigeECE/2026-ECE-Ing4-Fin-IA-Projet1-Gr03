"""
HJB Solver for Optimal Market Making

This module implements numerical solvers for the Hamilton-Jacobi-Bellman
equation in optimal market making with inventory constraints.

Methods implemented:
- Finite difference method
- Policy iteration
- Value iteration
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


@dataclass
class HJBSolverConfig:
    """Configuration for HJB solver."""
    
    # Time discretization
    T: float = 1.0              # Time horizon
    dt: float = 0.01            # Time step
    
    # Inventory discretization
    Q_max: int = 10             # Maximum inventory (absolute)
    
    # Market parameters
    sigma: float = 0.2           # Volatility
    A: float = 1.0              # Base intensity
    k: float = 0.5              # Market depth
    
    # Risk parameters
    gamma: float = 0.1           # Risk aversion
    phi: float = 0.01            # Terminal liquidation cost
    
    # Spread bounds
    delta_min: float = 0.001    # Minimum half-spread
    delta_max: float = 0.1      # Maximum half-spread
    
    # Solver parameters
    tol: float = 1e-6           # Convergence tolerance
    max_iter: int = 1000        # Maximum iterations


class FiniteDifferenceSolver:
    """
    Finite difference solver for the HJB equation.
    
    This solver uses backward induction with finite difference
    discretization of the HJB equation.
    """
    
    def __init__(self, config: HJBSolverConfig):
        """
        Initialize the finite difference solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        
        # Discretization
        self.N = int(config.T / config.dt)
        self.time_grid = np.linspace(0, config.T, self.N + 1)
        
        # Inventory grid
        self.inventory_grid = np.arange(-config.Q_max, config.Q_max + 1)
        self.n_inventory = len(self.inventory_grid)
        
        # Value function
        self.v = np.zeros((self.N + 1, self.n_inventory))
        
        # Optimal spreads
        self.delta_b = np.zeros((self.N + 1, self.n_inventory))
        self.delta_a = np.zeros((self.N + 1, self.n_inventory))
    
    def _intensity(self, delta: float) -> float:
        """Compute order arrival intensity."""
        return self.config.A * np.exp(-self.config.k * delta)
    
    def _terminal_condition(self) -> np.ndarray:
        """Compute terminal condition."""
        q = self.inventory_grid
        return -self.config.phi * q**2
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the HJB equation.
        
        Returns:
            Tuple of (value_function, delta_b, delta_a)
        """
        # Terminal condition
        self.v[self.N, :] = self._terminal_condition()
        
        # Backward induction
        for n in range(self.N - 1, -1, -1):
            for q_idx, q in enumerate(self.inventory_grid):
                # Solve for optimal spreads
                delta_b, delta_a = self._solve_optimal_spreads(n, q_idx, q)
                
                # Store optimal spreads
                self.delta_b[n, q_idx] = delta_b
                self.delta_a[n, q_idx] = delta_a
                
                # Update value function
                self.v[n, q_idx] = self._update_value(n, q_idx, q, delta_b, delta_a)
        
        return self.v, self.delta_b, self.delta_a
    
    def _solve_optimal_spreads(
        self,
        n: int,
        q_idx: int,
        q: int
    ) -> Tuple[float, float]:
        """
        Solve for optimal spreads at given state.
        
        Args:
            n: Time index
            q_idx: Inventory index
            q: Inventory position
            
        Returns:
            Tuple of (delta_b, delta_a)
        """
        # Get value differences
        if q < self.config.Q_max:
            v_diff_b = self.v[n + 1, q_idx + 1] - self.v[n + 1, q_idx]
        else:
            v_diff_b = float('inf')
        
        if q > -self.config.Q_max:
            v_diff_a = self.v[n + 1, q_idx] - self.v[n + 1, q_idx - 1]
        else:
            v_diff_a = float('inf')
        
        # Compute optimal spreads
        delta_b = 1 / (2 * self.config.k) + v_diff_b
        delta_a = 1 / (2 * self.config.k) + v_diff_a
        
        # Apply bounds
        delta_b = np.clip(delta_b, self.config.delta_min, self.config.delta_max)
        delta_a = np.clip(delta_a, self.config.delta_min, self.config.delta_max)
        
        # Handle boundary conditions
        if q >= self.config.Q_max:
            delta_b = self.config.delta_max
        if q <= -self.config.Q_max:
            delta_a = self.config.delta_max
        
        return delta_b, delta_a
    
    def _update_value(
        self,
        n: int,
        q_idx: int,
        q: int,
        delta_b: float,
        delta_a: float
    ) -> float:
        """
        Update value function.
        
        Args:
            n: Time index
            q_idx: Inventory index
            q: Inventory position
            delta_b: Bid half-spread
            delta_a: Ask half-spread
            
        Returns:
            Updated value
        """
        dt = self.config.dt
        
        # Compute intensities
        lambda_b = self._intensity(delta_b)
        lambda_a = self._intensity(delta_a)
        
        # Get next values
        if q < self.config.Q_max:
            v_next_b = self.v[n + 1, q_idx + 1]
        else:
            v_next_b = self.v[n + 1, q_idx]
        
        if q > -self.config.Q_max:
            v_next_a = self.v[n + 1, q_idx - 1]
        else:
            v_next_a = self.v[n + 1, q_idx]
        
        v_current = self.v[n + 1, q_idx]
        
        # Update value
        dv = -dt * (
            lambda_b * (v_next_b - v_current - delta_b) +
            lambda_a * (v_next_a - v_current + delta_a)
        )
        
        return v_current + dv
    
    def get_policy(self) -> Callable[[float, int], Tuple[float, float]]:
        """
        Get the optimal policy function.
        
        Returns:
            Function that takes (t, q) and returns (delta_b, delta_a)
        """
        def policy(t: float, q: int) -> Tuple[float, float]:
            t_idx = int(t / self.config.dt)
            t_idx = min(t_idx, self.N)
            
            q_idx = q + self.config.Q_max
            q_idx = max(0, min(q_idx, self.n_inventory - 1))
            
            return self.delta_b[t_idx, q_idx], self.delta_a[t_idx, q_idx]
        
        return policy


class PolicyIterationSolver:
    """
    Policy iteration solver for the HJB equation.
    
    This solver alternates between policy evaluation and policy
    improvement until convergence.
    """
    
    def __init__(self, config: HJBSolverConfig):
        """
        Initialize the policy iteration solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        
        # Discretization
        self.N = int(config.T / config.dt)
        self.inventory_grid = np.arange(-config.Q_max, config.Q_max + 1)
        self.n_inventory = len(self.inventory_grid)
        
        # Value function and policy
        self.v = np.zeros((self.N + 1, self.n_inventory))
        self.delta_b = np.zeros((self.N + 1, self.n_inventory))
        self.delta_a = np.zeros((self.N + 1, self.n_inventory))
    
    def _intensity(self, delta: float) -> float:
        """Compute order arrival intensity."""
        return self.config.A * np.exp(-self.config.k * delta)
    
    def _terminal_condition(self) -> np.ndarray:
        """Compute terminal condition."""
        q = self.inventory_grid
        return -self.config.phi * q**2
    
    def _policy_evaluation(self, max_iter: int = 100) -> None:
        """
        Evaluate the current policy.
        
        Args:
            max_iter: Maximum iterations for evaluation
        """
        # Terminal condition
        self.v[self.N, :] = self._terminal_condition()
        
        # Backward induction with fixed policy
        for _ in range(max_iter):
            v_old = self.v.copy()
            
            for n in range(self.N - 1, -1, -1):
                for q_idx, q in enumerate(self.inventory_grid):
                    delta_b = self.delta_b[n, q_idx]
                    delta_a = self.delta_a[n, q_idx]
                    
                    # Update value
                    self.v[n, q_idx] = self._update_value(n, q_idx, q, delta_b, delta_a)
            
            # Check convergence
            if np.max(np.abs(self.v - v_old)) < self.config.tol:
                break
    
    def _policy_improvement(self) -> float:
        """
        Improve the policy.
        
        Returns:
            Maximum change in policy
        """
        max_change = 0.0
        
        for n in range(self.N):
            for q_idx, q in enumerate(self.inventory_grid):
                # Compute optimal spreads
                delta_b_new, delta_a_new = self._solve_optimal_spreads(n, q_idx, q)
                
                # Track maximum change
                max_change = max(
                    max_change,
                    abs(delta_b_new - self.delta_b[n, q_idx]),
                    abs(delta_a_new - self.delta_a[n, q_idx])
                )
                
                # Update policy
                self.delta_b[n, q_idx] = delta_b_new
                self.delta_a[n, q_idx] = delta_a_new
        
        return max_change
    
    def _solve_optimal_spreads(
        self,
        n: int,
        q_idx: int,
        q: int
    ) -> Tuple[float, float]:
        """Solve for optimal spreads."""
        if q < self.config.Q_max:
            v_diff_b = self.v[n + 1, q_idx + 1] - self.v[n + 1, q_idx]
        else:
            v_diff_b = float('inf')
        
        if q > -self.config.Q_max:
            v_diff_a = self.v[n + 1, q_idx] - self.v[n + 1, q_idx - 1]
        else:
            v_diff_a = float('inf')
        
        delta_b = 1 / (2 * self.config.k) + v_diff_b
        delta_a = 1 / (2 * self.config.k) + v_diff_a
        
        delta_b = np.clip(delta_b, self.config.delta_min, self.config.delta_max)
        delta_a = np.clip(delta_a, self.config.delta_min, self.config.delta_max)
        
        if q >= self.config.Q_max:
            delta_b = self.config.delta_max
        if q <= -self.config.Q_max:
            delta_a = self.config.delta_max
        
        return delta_b, delta_a
    
    def _update_value(
        self,
        n: int,
        q_idx: int,
        q: int,
        delta_b: float,
        delta_a: float
    ) -> float:
        """Update value function."""
        dt = self.config.dt
        
        lambda_b = self._intensity(delta_b)
        lambda_a = self._intensity(delta_a)
        
        if q < self.config.Q_max:
            v_next_b = self.v[n + 1, q_idx + 1]
        else:
            v_next_b = self.v[n + 1, q_idx]
        
        if q > -self.config.Q_max:
            v_next_a = self.v[n + 1, q_idx - 1]
        else:
            v_next_a = self.v[n + 1, q_idx]
        
        v_current = self.v[n + 1, q_idx]
        
        dv = -dt * (
            lambda_b * (v_next_b - v_current - delta_b) +
            lambda_a * (v_next_a - v_current + delta_a)
        )
        
        return v_current + dv
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve using policy iteration.
        
        Returns:
            Tuple of (value_function, delta_b, delta_a)
        """
        # Initialize policy with constant spreads
        base_spread = 1 / (2 * self.config.k)
        self.delta_b[:, :] = base_spread
        self.delta_a[:, :] = base_spread
        
        # Policy iteration
        for iteration in range(self.config.max_iter):
            # Policy evaluation
            self._policy_evaluation()
            
            # Policy improvement
            policy_change = self._policy_improvement()
            
            if policy_change < self.config.tol:
                break
        
        return self.v, self.delta_b, self.delta_a
    
    def get_policy(self) -> Callable[[float, int], Tuple[float, float]]:
        """Get the optimal policy function."""
        def policy(t: float, q: int) -> Tuple[float, float]:
            t_idx = int(t / self.config.dt)
            t_idx = min(t_idx, self.N)
            
            q_idx = q + self.config.Q_max
            q_idx = max(0, min(q_idx, self.n_inventory - 1))
            
            return self.delta_b[t_idx, q_idx], self.delta_a[t_idx, q_idx]
        
        return policy


def create_default_solver(
    solver_type: str = "finite_difference"
) -> Tuple[object, Callable[[float, int], Tuple[float, float]]]:
    """
    Create a solver with default configuration.
    
    Args:
        solver_type: Type of solver ("finite_difference" or "policy_iteration")
        
    Returns:
        Tuple of (solver, policy_function)
    """
    config = HJBSolverConfig(
        T=1.0,
        dt=0.01,
        Q_max=10,
        sigma=0.2,
        A=1.0,
        k=0.5,
        gamma=0.1,
        phi=0.01,
        delta_min=0.001,
        delta_max=0.1,
        tol=1e-6,
        max_iter=1000
    )
    
    if solver_type == "finite_difference":
        solver = FiniteDifferenceSolver(config)
    elif solver_type == "policy_iteration":
        solver = PolicyIterationSolver(config)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    # Solve
    v, delta_b, delta_a = solver.solve()
    
    # Get policy
    policy = solver.get_policy()
    
    return solver, policy


if __name__ == "__main__":
    # Example usage
    print("Creating finite difference solver...")
    solver, policy = create_default_solver("finite_difference")
    
    print(f"\nSolver configuration:")
    print(f"  Time horizon: {solver.config.T}")
    print(f"  Time steps: {solver.N}")
    print(f"  Max inventory: {solver.config.Q_max}")
    
    # Example: Get optimal spreads
    print(f"\nOptimal spreads at t=0.5:")
    for q in [-5, -2, 0, 2, 5]:
        delta_b, delta_a = policy(0.5, q)
        print(f"  q={q:2d}: delta_b={delta_b:.4f}, delta_a={delta_a:.4f}")
