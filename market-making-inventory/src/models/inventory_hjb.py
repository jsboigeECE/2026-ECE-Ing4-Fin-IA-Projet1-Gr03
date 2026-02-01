"""
Inventory HJB Model for Optimal Market Making

This module implements the Hamilton-Jacobi-Bellman (HJB) equation
for optimal market making with inventory constraints, based on the
Guéant-Lehalle-Fernandez-Tapia model.

References:
    Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013).
    Dealing with the inventory risk: A solution to the market making problem.
    Mathematical Finance, 23(3), 517-554.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class HJBParameters:
    """Parameters for the HJB market making model."""
    
    # Market parameters
    sigma: float = 0.2          # Volatility
    A: float = 1.0              # Base intensity parameter
    k: float = 0.5              # Market depth parameter
    
    # Risk parameters
    gamma: float = 0.1          # Risk aversion coefficient
    phi: float = 0.01           # Terminal liquidation cost parameter
    
    # Time parameters
    T: float = 1.0              # Time horizon
    dt: float = 0.01            # Time step
    
    # Inventory constraints
    Q_max: int = 10             # Maximum inventory (absolute value)
    
    # Spread bounds
    delta_min: float = 0.001    # Minimum half-spread
    delta_max: float = 0.1      # Maximum half-spread


class InventoryHJB:
    """
    HJB-based optimal market making model with inventory constraints.
    
    This class implements the solution to the stochastic optimal control
    problem for market making, using the Hamilton-Jacobi-Bellman equation.
    """
    
    def __init__(self, params: HJBParameters):
        """
        Initialize the HJB model.
        
        Args:
            params: Model parameters
        """
        self.params = params
        
        # Discretization
        self.N = int(params.T / params.dt)  # Number of time steps
        self.time_grid = np.linspace(0, params.T, self.N + 1)
        
        # Inventory grid
        self.inventory_grid = np.arange(-params.Q_max, params.Q_max + 1)
        self.n_inventory = len(self.inventory_grid)
        
        # Value function: v[t, q_idx]
        self.v = np.zeros((self.N + 1, self.n_inventory))
        
        # Optimal spreads: delta_b[t, q_idx], delta_a[t, q_idx]
        self.delta_b = np.zeros((self.N + 1, self.n_inventory))
        self.delta_a = np.zeros((self.N + 1, self.n_inventory))
        
        # Solve the HJB equation
        self._solve()
    
    def _intensity(self, delta: float) -> float:
        """
        Compute the order arrival intensity for a given spread.
        
        Args:
            delta: Half-spread
            
        Returns:
            Intensity (arrival rate)
        """
        return self.params.A * np.exp(-self.params.k * delta)
    
    def _terminal_condition(self) -> np.ndarray:
        """
        Compute the terminal condition for the value function.
        
        Returns:
            Terminal value function v(T, q)
        """
        q = self.inventory_grid
        return -self.params.phi * q**2
    
    def _solve(self) -> None:
        """
        Solve the HJB equation using backward induction.
        
        The algorithm:
        1. Set terminal condition
        2. Iterate backward in time
        3. At each step, compute optimal spreads and update value function
        """
        # Terminal condition
        self.v[self.N, :] = self._terminal_condition()
        
        # Backward induction
        for n in range(self.N - 1, -1, -1):
            for q_idx, q in enumerate(self.inventory_grid):
                # Compute optimal spreads
                delta_b, delta_a = self._compute_optimal_spreads(n, q_idx, q)
                
                # Store optimal spreads
                self.delta_b[n, q_idx] = delta_b
                self.delta_a[n, q_idx] = delta_a
                
                # Update value function
                self.v[n, q_idx] = self._update_value_function(
                    n, q_idx, q, delta_b, delta_a
                )
    
    def _compute_optimal_spreads(
        self, 
        n: int, 
        q_idx: int, 
        q: int
    ) -> Tuple[float, float]:
        """
        Compute optimal bid and ask spreads for given state.
        
        Args:
            n: Time index
            q_idx: Inventory index
            q: Inventory position
            
        Returns:
            Tuple of (delta_b, delta_a) optimal half-spreads
        """
        # Get value differences
        if q < self.params.Q_max:
            v_diff_b = self.v[n + 1, q_idx + 1] - self.v[n + 1, q_idx]
        else:
            # At upper bound, cannot buy more
            v_diff_b = float('inf')
        
        if q > -self.params.Q_max:
            v_diff_a = self.v[n + 1, q_idx] - self.v[n + 1, q_idx - 1]
        else:
            # At lower bound, cannot sell more
            v_diff_a = float('inf')
        
        # Compute optimal spreads
        delta_b = 1 / (2 * self.params.k) + v_diff_b
        delta_a = 1 / (2 * self.params.k) + v_diff_a
        
        # Apply bounds
        delta_b = np.clip(delta_b, self.params.delta_min, self.params.delta_max)
        delta_a = np.clip(delta_a, self.params.delta_min, self.params.delta_max)
        
        # Handle boundary conditions
        if q >= self.params.Q_max:
            delta_b = self.params.delta_max  # Cannot buy more
        if q <= -self.params.Q_max:
            delta_a = self.params.delta_max  # Cannot sell more
        
        return delta_b, delta_a
    
    def _update_value_function(
        self,
        n: int,
        q_idx: int,
        q: int,
        delta_b: float,
        delta_a: float
    ) -> float:
        """
        Update the value function for given state and spreads.
        
        Args:
            n: Time index
            q_idx: Inventory index
            q: Inventory position
            delta_b: Bid half-spread
            delta_a: Ask half-spread
            
        Returns:
            Updated value function value
        """
        dt = self.params.dt
        
        # Compute intensities
        lambda_b = self._intensity(delta_b)
        lambda_a = self._intensity(delta_a)
        
        # Compute value differences
        if q < self.params.Q_max:
            v_next_b = self.v[n + 1, q_idx + 1]
        else:
            v_next_b = self.v[n + 1, q_idx]
        
        if q > -self.params.Q_max:
            v_next_a = self.v[n + 1, q_idx - 1]
        else:
            v_next_a = self.v[n + 1, q_idx]
        
        v_current = self.v[n + 1, q_idx]
        
        # Update value function using HJB equation
        dv = -dt * (
            lambda_b * (v_next_b - v_current - delta_b) +
            lambda_a * (v_next_a - v_current + delta_a)
        )
        
        return v_current + dv
    
    def get_optimal_spreads(
        self,
        t: float,
        q: int
    ) -> Tuple[float, float]:
        """
        Get optimal bid and ask spreads for given time and inventory.
        
        Args:
            t: Current time
            q: Current inventory position
            
        Returns:
            Tuple of (delta_b, delta_a) optimal half-spreads
        """
        # Find time index
        t_idx = int(t / self.params.dt)
        t_idx = min(t_idx, self.N)
        
        # Find inventory index
        q_idx = q + self.params.Q_max
        q_idx = max(0, min(q_idx, self.n_inventory - 1))
        
        return self.delta_b[t_idx, q_idx], self.delta_a[t_idx, q_idx]
    
    def get_value_function(
        self,
        t: float,
        q: int
    ) -> float:
        """
        Get the value function for given time and inventory.
        
        Args:
            t: Current time
            q: Current inventory position
            
        Returns:
            Value function value
        """
        # Find time index
        t_idx = int(t / self.params.dt)
        t_idx = min(t_idx, self.N)
        
        # Find inventory index
        q_idx = q + self.params.Q_max
        q_idx = max(0, min(q_idx, self.n_inventory - 1))
        
        return self.v[t_idx, q_idx]
    
    def get_quotes(
        self,
        t: float,
        q: int,
        S: float
    ) -> Tuple[float, float]:
        """
        Get optimal bid and ask quotes for given state.
        
        Args:
            t: Current time
            q: Current inventory position
            S: Current midprice
            
        Returns:
            Tuple of (bid_price, ask_price)
        """
        delta_b, delta_a = self.get_optimal_spreads(t, q)
        
        bid_price = S - delta_b
        ask_price = S + delta_a
        
        return bid_price, ask_price
    
    def compute_asymptotic_spreads(
        self,
        t: float,
        q: int
    ) -> Tuple[float, float]:
        """
        Compute asymptotic optimal spreads (GLFT approximation).
        
        This uses the closed-form approximation for large time horizons.
        
        Args:
            t: Current time
            q: Current inventory position
            
        Returns:
            Tuple of (delta_b, delta_a) asymptotic half-spreads
        """
        tau = self.params.T - t  # Remaining time
        
        base_spread = 1 / (2 * self.params.k)
        inventory_adjustment = self.params.gamma * self.params.sigma**2 * tau
        
        delta_b = base_spread + inventory_adjustment * (q + 0.5)
        delta_a = base_spread + inventory_adjustment * (q - 0.5)
        
        # Apply bounds
        delta_b = np.clip(delta_b, self.params.delta_min, self.params.delta_max)
        delta_a = np.clip(delta_a, self.params.delta_min, self.params.delta_max)
        
        return delta_b, delta_a
    
    def compute_var(
        self,
        q: int,
        t: float,
        alpha: float = 0.95
    ) -> float:
        """
        Compute Value at Risk (VaR) for given inventory and time.
        
        Args:
            q: Current inventory position
            t: Current time
            alpha: Confidence level (default 0.95)
            
        Returns:
            VaR value
        """
        tau = self.params.T - t
        if tau <= 0:
            return 0.0
        
        from scipy.stats import norm
        
        var = abs(q) * self.params.sigma * np.sqrt(tau) * norm.ppf(alpha)
        return var
    
    def compute_cvar(
        self,
        q: int,
        t: float,
        alpha: float = 0.95
    ) -> float:
        """
        Compute Conditional VaR (CVaR) for given inventory and time.
        
        Args:
            q: Current inventory position
            t: Current time
            alpha: Confidence level (default 0.95)
            
        Returns:
            CVaR value
        """
        tau = self.params.T - t
        if tau <= 0:
            return 0.0
        
        from scipy.stats import norm
        
        z_alpha = norm.ppf(alpha)
        cvar = abs(q) * self.params.sigma * np.sqrt(tau) * norm.pdf(z_alpha) / (1 - alpha)
        return cvar


def create_default_model() -> InventoryHJB:
    """
    Create an HJB model with default parameters.
    
    Returns:
        InventoryHJB instance with default parameters
    """
    params = HJBParameters(
        sigma=0.2,
        A=1.0,
        k=0.5,
        gamma=0.1,
        phi=0.01,
        T=1.0,
        dt=0.01,
        Q_max=10,
        delta_min=0.001,
        delta_max=0.1
    )
    return InventoryHJB(params)


if __name__ == "__main__":
    # Example usage
    print("Creating HJB model with default parameters...")
    model = create_default_model()
    
    print(f"\nModel parameters:")
    print(f"  Volatility (sigma): {model.params.sigma}")
    print(f"  Market depth (k): {model.params.k}")
    print(f"  Risk aversion (gamma): {model.params.gamma}")
    print(f"  Time horizon (T): {model.params.T}")
    print(f"  Max inventory (Q_max): {model.params.Q_max}")
    
    # Example: Get optimal spreads for different inventory levels
    print(f"\nOptimal spreads at t=0.5:")
    for q in [-5, -2, 0, 2, 5]:
        delta_b, delta_a = model.get_optimal_spreads(0.5, q)
        print(f"  q={q:2d}: delta_b={delta_b:.4f}, delta_a={delta_a:.4f}")
    
    # Example: Compute VaR
    print(f"\nVaR (95%) at t=0.5:")
    for q in [-5, -2, 0, 2, 5]:
        var = model.compute_var(q, 0.5, alpha=0.95)
        print(f"  q={q:2d}: VaR={var:.4f}")
    
    # Example: Get quotes
    S = 100.0
    q = 3
    t = 0.3
    bid, ask = model.get_quotes(t, q, S)
    print(f"\nQuotes at t={t}, q={q}, S={S}:")
    print(f"  Bid: {bid:.4f}")
    print(f"  Ask: {ask:.4f}")
    print(f"  Spread: {ask - bid:.4f}")
