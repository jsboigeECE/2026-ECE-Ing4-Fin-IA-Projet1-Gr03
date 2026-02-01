"""
Market Making Environment for Reinforcement Learning

This module implements a Gymnasium environment for market making
with inventory constraints, suitable for training RL agents.

The environment follows the Gymnasium API and can be used with
algorithms like PPO, DQN, SAC, etc.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces


@dataclass
class MarketMakingEnvConfig:
    """Configuration for market making environment."""
    
    # Market parameters
    sigma: float = 0.2              # Volatility
    mu: float = 0.0                 # Drift
    S0: float = 100.0               # Initial midprice
    
    # Order flow parameters
    A: float = 1.0                  # Base intensity
    k: float = 0.5                  # Market depth parameter
    
    # Time parameters
    T: float = 1.0                  # Episode length
    dt: float = 0.01                # Time step
    
    # Inventory constraints
    Q_max: int = 10                 # Maximum inventory (absolute)
    
    # Action space (spreads)
    delta_min: float = 0.001        # Minimum half-spread
    delta_max: float = 0.1          # Maximum half-spread
    n_spread_levels: int = 10        # Number of discrete spread levels
    
    # Reward parameters
    reward_type: str = "pnl"        # "pnl", "sharpe", "inventory_penalty"
    inventory_penalty: float = 0.01   # Penalty for holding inventory
    
    # Random seed
    seed: Optional[int] = None      # Random seed


class MarketMakingEnv(gym.Env):
    """
    Gymnasium environment for market making with inventory constraints.
    
    State space: [normalized_time, normalized_inventory, normalized_price]
    Action space: Discrete spread levels for bid and ask
    
    The environment simulates a market maker who posts bid and ask quotes
    and receives rewards based on PnL, Sharpe ratio, or other metrics.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: MarketMakingEnvConfig):
        """
        Initialize the market making environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Discretization
        self.N = int(config.T / config.dt)
        self.time_grid = np.linspace(0, config.T, self.N + 1)
        
        # Action space: discrete spread levels for bid and ask
        # Action = bid_level * n_spread_levels + ask_level
        self.spread_levels = np.linspace(
            config.delta_min,
            config.delta_max,
            config.n_spread_levels
        )
        self.action_space = spaces.Discrete(config.n_spread_levels ** 2)
        
        # Observation space: [time, inventory, price, price_change]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # State variables
        self.t = 0.0
        self.S = config.S0
        self.q = 0
        self.X = 0.0
        self.step_count = 0
        
        # History for rendering
        self.price_history = [config.S0]
        self.inventory_history = [0]
        self.pnl_history = [0.0]
        self.trade_history = []
        
        # Posted quotes
        self.bid_price = None
        self.ask_price = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        self.t = 0.0
        self.S = self.config.S0
        self.q = 0
        self.X = 0.0
        self.step_count = 0
        
        # Reset history
        self.price_history = [self.config.S0]
        self.inventory_history = [0]
        self.pnl_history = [0.0]
        self.trade_history = []
        
        # Reset quotes
        self.bid_price = None
        self.ask_price = None
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Discrete action (bid_level, ask_level)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action
        bid_level = action // self.config.n_spread_levels
        ask_level = action % self.config.n_spread_levels
        
        # Get spreads
        delta_b = self.spread_levels[bid_level]
        delta_a = self.spread_levels[ask_level]
        
        # Post quotes
        self.bid_price = self.S - delta_b
        self.ask_price = self.S + delta_a
        
        # Update price
        dS = self._generate_price_step()
        self.S += dS
        
        # Check for market orders
        trade_occurred = self._check_market_orders()
        
        # Update time
        self.t += self.config.dt
        self.step_count += 1
        
        # Record history
        self.price_history.append(self.S)
        self.inventory_history.append(self.q)
        self.pnl_history.append(self._get_pnl())
        
        # Compute reward
        reward = self._compute_reward(trade_occurred)
        
        # Check termination
        terminated = self.t >= self.config.T
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_price_step(self) -> float:
        """Generate a price step using Brownian motion."""
        dt = self.config.dt
        sigma = self.config.sigma
        mu = self.config.mu
        
        dW = np.sqrt(dt) * np.random.randn()
        dS = mu * self.S * dt + sigma * self.S * dW
        
        return dS
    
    def _check_market_orders(self) -> bool:
        """
        Check if market orders arrive and execute trades.
        
        Returns:
            True if a trade occurred, False otherwise
        """
        if self.bid_price is None or self.ask_price is None:
            return False
        
        # Compute half-spreads
        delta_b = self.S - self.bid_price
        delta_a = self.ask_price - self.S
        
        # Compute intensities
        lambda_b = self.config.A * np.exp(-self.config.k * delta_b)
        lambda_a = self.config.A * np.exp(-self.config.k * delta_a)
        
        dt = self.config.dt
        
        # Check for buy order (fills ask)
        if np.random.rand() < lambda_a * dt:
            if self.q > -self.config.Q_max:
                self.q -= 1
                self.X += self.ask_price
                self.trade_history.append({
                    'time': self.t,
                    'side': 'buy',
                    'price': self.ask_price,
                    'inventory': self.q
                })
                return True
        
        # Check for sell order (fills bid)
        if np.random.rand() < lambda_b * dt:
            if self.q < self.config.Q_max:
                self.q += 1
                self.X -= self.bid_price
                self.trade_history.append({
                    'time': self.t,
                    'side': 'sell',
                    'price': self.bid_price,
                    'inventory': self.q
                })
                return True
        
        return False
    
    def _get_pnl(self) -> float:
        """Get current PnL (realized + unrealized)."""
        realized_pnl = self.X
        unrealized_pnl = self.q * self.S
        return realized_pnl + unrealized_pnl
    
    def _compute_reward(self, trade_occurred: bool) -> float:
        """
        Compute reward based on configuration.
        
        Args:
            trade_occurred: Whether a trade occurred
            
        Returns:
            Reward value
        """
        if self.config.reward_type == "pnl":
            # Reward is change in PnL
            if len(self.pnl_history) >= 2:
                return self.pnl_history[-1] - self.pnl_history[-2]
            return 0.0
        
        elif self.config.reward_type == "sharpe":
            # Reward is Sharpe ratio of recent returns
            if len(self.pnl_history) >= 10:
                recent_returns = np.diff(self.pnl_history[-10:])
                if np.std(recent_returns) > 0:
                    return np.mean(recent_returns) / np.std(recent_returns)
            return 0.0
        
        elif self.config.reward_type == "inventory_penalty":
            # Reward is PnL minus inventory penalty
            pnl_change = 0.0
            if len(self.pnl_history) >= 2:
                pnl_change = self.pnl_history[-1] - self.pnl_history[-2]
            
            penalty = self.config.inventory_penalty * self.q ** 2
            return pnl_change - penalty
        
        else:
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Get normalized observation.
        
        Returns:
            Normalized observation array
        """
        # Normalize time
        norm_time = self.t / self.config.T
        
        # Normalize inventory
        norm_inventory = self.q / self.config.Q_max
        
        # Normalize price (relative to initial)
        norm_price = self.S / self.config.S0
        
        # Price change (normalized)
        if len(self.price_history) >= 2:
            price_change = (self.S - self.price_history[-2]) / self.price_history[-2]
        else:
            price_change = 0.0
        
        return np.array([norm_time, norm_inventory, norm_price, price_change], dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'time': self.t,
            'midprice': self.S,
            'inventory': self.q,
            'pnl': self._get_pnl(),
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.ask_price - self.bid_price if self.bid_price and self.ask_price else 0.0,
            'n_trades': len(self.trade_history)
        }
    
    def render(self, mode: str = "human") -> None:
        """
        Render the environment.
        
        Args:
            mode: Render mode
        """
        if mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"Time: {self.t:.3f} / {self.config.T}")
            print(f"Midprice: {self.S:.4f}")
            print(f"Inventory: {self.q}")
            print(f"PnL: {self._get_pnl():.4f}")
            print(f"Bid: {self.bid_price:.4f}" if self.bid_price else "Bid: None")
            print(f"Ask: {self.ask_price:.4f}" if self.ask_price else "Ask: None")
            print(f"Spread: {self.ask_price - self.bid_price:.4f}" if self.bid_price and self.ask_price else "Spread: N/A")
            print(f"Trades: {len(self.trade_history)}")
    
    def close(self) -> None:
        """Clean up the environment."""
        pass


def create_default_env(
    reward_type: str = "pnl",
    seed: Optional[int] = None
) -> MarketMakingEnv:
    """
    Create a market making environment with default configuration.
    
    Args:
        reward_type: Type of reward function
        seed: Random seed
        
    Returns:
        MarketMakingEnv instance
    """
    config = MarketMakingEnvConfig(
        sigma=0.2,
        mu=0.0,
        S0=100.0,
        A=1.0,
        k=0.5,
        T=1.0,
        dt=0.01,
        Q_max=10,
        delta_min=0.001,
        delta_max=0.1,
        n_spread_levels=10,
        reward_type=reward_type,
        inventory_penalty=0.01,
        seed=seed
    )
    return MarketMakingEnv(config)


if __name__ == "__main__":
    # Example usage
    print("Creating market making environment...")
    env = create_default_env(reward_type="pnl", seed=42)
    
    print(f"\nEnvironment info:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Max episode steps: {env.N}")
    
    # Run a random episode
    print("\nRunning random episode...")
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    for step in range(env.N):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: Reward={reward:.4f}, PnL={info['pnl']:.4f}, Inventory={info['inventory']}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished!")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final PnL: {info['pnl']:.4f}")
    print(f"  Final inventory: {info['inventory']}")
    print(f"  Number of trades: {info['n_trades']}")
    
    env.close()
