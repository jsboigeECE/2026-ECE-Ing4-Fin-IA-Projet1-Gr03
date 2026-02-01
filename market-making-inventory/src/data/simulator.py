"""
Market Making Simulator

This module implements a simulator for limit order book dynamics,
market order arrivals, and price trajectories for backtesting
market making strategies.

The simulator implements:
- Brownian motion price dynamics
- Poisson process for market order arrivals
- Order book state tracking
- Trade execution simulation
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = 1
    SELL = -1


@dataclass
class Trade:
    """Represents a executed trade."""
    timestamp: float
    side: OrderSide
    price: float
    quantity: int
    midprice: float


@dataclass
class SimulatorParameters:
    """Parameters for the market making simulator."""
    
    # Price dynamics
    sigma: float = 0.2              # Volatility
    mu: float = 0.0                 # Drift
    S0: float = 100.0               # Initial midprice
    
    # Order flow parameters
    A: float = 1.0                  # Base intensity
    k: float = 0.5                  # Market depth parameter
    
    # Time parameters
    T: float = 1.0                  # Time horizon
    dt: float = 0.001               # Time step for price updates
    
    # Inventory constraints
    Q_max: int = 10                 # Maximum inventory (absolute)
    
    # Order size
    order_size: int = 1             # Size of each order
    
    # Random seed
    seed: Optional[int] = None      # Random seed for reproducibility


class LimitOrderBook:
    """
    Simple limit order book simulator.
    
    This class simulates a limit order book with bid and ask quotes
    posted by the market maker.
    """
    
    def __init__(self, S0: float):
        """
        Initialize the limit order book.
        
        Args:
            S0: Initial midprice
        """
        self.S = S0  # Current midprice
        self.bid_price: Optional[float] = None
        self.ask_price: Optional[float] = None
        self.bid_qty: int = 0
        self.ask_qty: int = 0
    
    def update_midprice(self, S: float) -> None:
        """
        Update the midprice.
        
        Args:
            S: New midprice
        """
        self.S = S
    
    def post_quotes(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: int,
        ask_qty: int
    ) -> None:
        """
        Post bid and ask quotes.
        
        Args:
            bid_price: Bid price
            ask_price: Ask price
            bid_qty: Bid quantity
            ask_qty: Ask quantity
        """
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty
    
    def get_spread(self) -> float:
        """
        Get the current bid-ask spread.
        
        Returns:
            Spread (ask - bid)
        """
        if self.bid_price is None or self.ask_price is None:
            return 0.0
        return self.ask_price - self.bid_price
    
    def get_midprice(self) -> float:
        """
        Get the current midprice.
        
        Returns:
            Midprice
        """
        return self.S


class MarketMakingSimulator:
    """
    Simulator for market making strategies.
    
    This class simulates the market environment for a market maker,
    including price dynamics, order arrivals, and trade execution.
    """
    
    def __init__(self, params: SimulatorParameters):
        """
        Initialize the simulator.
        
        Args:
            params: Simulator parameters
        """
        self.params = params
        
        # Set random seed
        if params.seed is not None:
            np.random.seed(params.seed)
        
        # Initialize state
        self.t = 0.0
        self.S = params.S0
        self.q = 0  # Inventory
        self.X = 0.0  # Cash position
        
        # Initialize order book
        self.lob = LimitOrderBook(params.S0)
        
        # Trade history
        self.trades: List[Trade] = []
        
        # Price history
        self.price_history: List[float] = [params.S0]
        self.time_history: List[float] = [0.0]
        
        # Inventory history
        self.inventory_history: List[int] = [0]
        
        # PnL history
        self.pnl_history: List[float] = [0.0]
    
    def _generate_price_step(self) -> float:
        """
        Generate a price step using Brownian motion.
        
        Returns:
            Price increment
        """
        dt = self.params.dt
        sigma = self.params.sigma
        mu = self.params.mu
        
        # Brownian increment
        dW = np.sqrt(dt) * np.random.randn()
        
        # Price increment
        dS = mu * self.S * dt + sigma * self.S * dW
        
        return dS
    
    def _compute_intensity(self, delta: float) -> float:
        """
        Compute order arrival intensity for a given spread.
        
        Args:
            delta: Half-spread
            
        Returns:
            Intensity (arrival rate per unit time)
        """
        return self.params.A * np.exp(-self.params.k * delta)
    
    def _check_market_order(self, dt: float) -> Optional[OrderSide]:
        """
        Check if a market order arrives in the given time step.
        
        Args:
            dt: Time step
            
        Returns:
            Order side if order arrives, None otherwise
        """
        if self.lob.bid_price is None or self.lob.ask_price is None:
            return None
        
        # Compute half-spreads
        delta_b = self.S - self.lob.bid_price
        delta_a = self.lob.ask_price - self.S
        
        # Compute intensities
        lambda_b = self._compute_intensity(delta_b)
        lambda_a = self._compute_intensity(delta_a)
        
        # Check for buy order (fills ask)
        if np.random.rand() < lambda_a * dt:
            return OrderSide.BUY
        
        # Check for sell order (fills bid)
        if np.random.rand() < lambda_b * dt:
            return OrderSide.SELL
        
        return None
    
    def _execute_trade(self, side: OrderSide) -> Trade:
        """
        Execute a trade.
        
        Args:
            side: Order side
            
        Returns:
            Trade object
        """
        if side == OrderSide.BUY:
            # Buy order fills our ask
            price = self.lob.ask_price
            quantity = min(self.params.order_size, self.lob.ask_qty)
            
            # Update inventory and cash
            self.q -= quantity
            self.X += quantity * price
            
        else:  # OrderSide.SELL
            # Sell order fills our bid
            price = self.lob.bid_price
            quantity = min(self.params.order_size, self.lob.bid_qty)
            
            # Update inventory and cash
            self.q += quantity
            self.X -= quantity * price
        
        # Create trade record
        trade = Trade(
            timestamp=self.t,
            side=side,
            price=price,
            quantity=quantity,
            midprice=self.S
        )
        
        self.trades.append(trade)
        
        return trade
    
    def _check_inventory_constraints(self) -> bool:
        """
        Check if inventory constraints are violated.
        
        Returns:
            True if constraints are satisfied, False otherwise
        """
        return abs(self.q) <= self.params.Q_max
    
    def get_pnl(self) -> float:
        """
        Get current PnL (unrealized + realized).
        
        Returns:
            Current PnL
        """
        # Realized PnL from cash
        realized_pnl = self.X
        
        # Unrealized PnL from inventory
        unrealized_pnl = self.q * self.S
        
        return realized_pnl + unrealized_pnl
    
    def step(
        self,
        bid_price: float,
        ask_price: float,
        bid_qty: int = 1,
        ask_qty: int = 1
    ) -> Tuple[float, int, float, Optional[Trade]]:
        """
        Execute one simulation step.
        
        Args:
            bid_price: Bid price to post
            ask_price: Ask price to post
            bid_qty: Bid quantity
            ask_qty: Ask quantity
            
        Returns:
            Tuple of (new_midprice, new_inventory, pnl, trade)
        """
        # Post quotes
        self.lob.post_quotes(bid_price, ask_price, bid_qty, ask_qty)
        
        # Update price
        dS = self._generate_price_step()
        self.S += dS
        self.lob.update_midprice(self.S)
        
        # Check for market orders
        trade = None
        order_side = self._check_market_order(self.params.dt)
        
        if order_side is not None:
            # Check inventory constraints before executing
            if order_side == OrderSide.BUY and self.q > -self.params.Q_max:
                trade = self._execute_trade(order_side)
            elif order_side == OrderSide.SELL and self.q < self.params.Q_max:
                trade = self._execute_trade(order_side)
        
        # Update time
        self.t += self.params.dt
        
        # Record history
        self.price_history.append(self.S)
        self.time_history.append(self.t)
        self.inventory_history.append(self.q)
        self.pnl_history.append(self.get_pnl())
        
        return self.S, self.q, self.get_pnl(), trade
    
    def run_simulation(
        self,
        policy: Callable[[float, int, float], Tuple[float, float]],
        record_interval: int = 100
    ) -> dict:
        """
        Run a full simulation with a given policy.
        
        Args:
            policy: Function that takes (t, q, S) and returns (bid_price, ask_price)
            record_interval: Interval for recording intermediate states
            
        Returns:
            Dictionary with simulation results
        """
        # Reset state
        self.t = 0.0
        self.S = self.params.S0
        self.q = 0
        self.X = 0.0
        self.trades = []
        self.price_history = [self.params.S0]
        self.time_history = [0.0]
        self.inventory_history = [0]
        self.pnl_history = [0.0]
        
        # Number of steps
        n_steps = int(self.params.T / self.params.dt)
        
        # Run simulation
        for step in range(n_steps):
            # Get quotes from policy
            bid_price, ask_price = policy(self.t, self.q, self.S)
            
            # Execute step
            self.step(bid_price, ask_price)
        
        # Terminal liquidation
        terminal_pnl = self._liquidate_inventory()
        
        return {
            'final_pnl': terminal_pnl,
            'final_inventory': self.q,
            'final_midprice': self.S,
            'n_trades': len(self.trades),
            'price_history': np.array(self.price_history),
            'time_history': np.array(self.time_history),
            'inventory_history': np.array(self.inventory_history),
            'pnl_history': np.array(self.pnl_history),
            'trades': self.trades
        }
    
    def _liquidate_inventory(self) -> float:
        """
        Liquidate remaining inventory at terminal time.
        
        Returns:
            Final PnL after liquidation
        """
        if self.q != 0:
            # Liquidate at midprice
            self.X += self.q * self.S
            self.q = 0
        
        return self.get_pnl()
    
    def get_statistics(self) -> dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with statistics
        """
        if len(self.pnl_history) < 2:
            return {}
        
        pnl_array = np.array(self.pnl_history)
        returns = np.diff(pnl_array)
        
        return {
            'final_pnl': self.get_pnl(),
            'final_inventory': self.q,
            'n_trades': len(self.trades),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_drawdown': self._compute_max_drawdown(),
            'avg_inventory': np.mean(np.abs(self.inventory_history)),
            'max_inventory': np.max(np.abs(self.inventory_history))
        }
    
    def _compute_max_drawdown(self) -> float:
        """
        Compute maximum drawdown.
        
        Returns:
            Maximum drawdown
        """
        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdown = (pnl_array - running_max) / running_max
        return np.min(drawdown)


def create_default_simulator(seed: Optional[int] = None) -> MarketMakingSimulator:
    """
    Create a simulator with default parameters.
    
    Args:
        seed: Random seed
        
    Returns:
        MarketMakingSimulator instance
    """
    params = SimulatorParameters(
        sigma=0.2,
        mu=0.0,
        S0=100.0,
        A=1.0,
        k=0.5,
        T=1.0,
        dt=0.001,
        Q_max=10,
        order_size=1,
        seed=seed
    )
    return MarketMakingSimulator(params)


if __name__ == "__main__":
    # Example usage
    print("Creating market making simulator...")
    simulator = create_default_simulator(seed=42)
    
    # Define a simple policy (constant spread)
    def constant_spread_policy(t: float, q: int, S: float) -> Tuple[float, float]:
        spread = 0.02
        return S - spread/2, S + spread/2
    
    # Run simulation
    print("Running simulation...")
    results = simulator.run_simulation(constant_spread_policy)
    
    print(f"\nSimulation Results:")
    print(f"  Final PnL: {results['final_pnl']:.4f}")
    print(f"  Final Inventory: {results['final_inventory']}")
    print(f"  Final Midprice: {results['final_midprice']:.4f}")
    print(f"  Number of Trades: {results['n_trades']}")
    
    # Get statistics
    stats = simulator.get_statistics()
    print(f"\nStatistics:")
    print(f"  Mean Return: {stats['mean_return']:.6f}")
    print(f"  Std Return: {stats['std_return']:.6f}")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.4f}")
    print(f"  Avg |Inventory|: {stats['avg_inventory']:.2f}")
