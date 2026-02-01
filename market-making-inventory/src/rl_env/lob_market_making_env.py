"""
LOB Market Making Environment for Reinforcement Learning

This module implements a Gymnasium environment for market making
using real LOB (Limit Order Book) data from LOBSTER or Binance.

The environment follows the Gymnasium API and can be used with
algorithms like PPO, DQN, SAC, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Any, List
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from ..data.lob_loader import LOBDataLoader, LOBDataConfig


@dataclass
class LOBMarketMakingEnvConfig:
    """Configuration for LOB market making environment."""
    
    # Data parameters
    data_source: str = "lobster"  # "lobster" or "binance"
    data_path: Optional[str] = None
    message_file: Optional[str] = None
    orderbook_file: Optional[str] = None
    symbol: str = "BTCUSDT"
    
    # LOB parameters
    n_levels: int = 10  # Number of price levels to use
    tick_size: float = 0.01
    
    # Episode parameters
    episode_length: int = 1000  # Number of timesteps per episode
    start_idx: int = 0  # Starting index in the data
    
    # Inventory constraints
    Q_max: int = 10  # Maximum inventory (absolute)
    
    # Action space (spreads)
    delta_min: float = 0.001  # Minimum half-spread
    delta_max: float = 0.1  # Maximum half-spread
    n_spread_levels: int = 10  # Number of discrete spread levels
    
    # Reward parameters
    reward_type: str = "pnl"  # "pnl", "sharpe", "inventory_penalty", "spread_profit"
    inventory_penalty: float = 0.01  # Penalty for holding inventory
    transaction_cost: float = 0.0001  # Transaction cost per trade
    
    # Feature parameters
    use_lob_features: bool = True  # Use LOB depth features
    use_order_flow: bool = True  # Use order flow imbalance
    use_volatility: bool = True  # Use volatility features
    volatility_window: int = 100
    
    # Random seed
    seed: Optional[int] = None


class LOBMarketMakingEnv(gym.Env):
    """
    Gymnasium environment for market making with real LOB data.
    
    State space includes:
    - Normalized time
    - Normalized inventory
    - Normalized midprice
    - Price change
    - Spread
    - Order flow imbalance (optional)
    - Market depth features (optional)
    - Volatility (optional)
    
    Action space: Discrete spread levels for bid and ask
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: LOBMarketMakingEnvConfig):
        """
        Initialize the LOB market making environment.
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Load LOB data
        self._load_data()
        
        # Action space: discrete spread levels for bid and ask
        # Action = bid_level * n_spread_levels + ask_level
        self.spread_levels = np.linspace(
            config.delta_min,
            config.delta_max,
            config.n_spread_levels
        )
        self.action_space = spaces.Discrete(config.n_spread_levels ** 2)
        
        # Observation space
        obs_size = self._get_observation_size()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # State variables
        self.current_idx = config.start_idx
        self.q = 0
        self.X = 0.0
        self.step_count = 0
        
        # History for rendering
        self.price_history = []
        self.inventory_history = []
        self.pnl_history = []
        self.trade_history = []
        
        # Posted quotes
        self.bid_price = None
        self.ask_price = None
        
        # Cached features
        self._compute_features()
    
    def _load_data(self):
        """Load LOB data from the configured source."""
        lob_config = LOBDataConfig(
            source=self.config.data_source,
            data_path=self.config.data_path,
            message_file=self.config.message_file,
            orderbook_file=self.config.orderbook_file,
            symbol=self.config.symbol,
            n_levels=self.config.n_levels,
            normalize=True
        )
        
        self.loader = LOBDataLoader(lob_config)
        self.data = self.loader.load()
        
        # Precompute features
        self._precompute_features()
    
    def _precompute_features(self):
        """Precompute features for the entire dataset."""
        # Price changes
        self.data['price_change'] = self.data['midprice'].pct_change().fillna(0)
        
        # Volatility
        if self.config.use_volatility:
            self.data['volatility'] = self.data['price_change'].rolling(
                window=self.config.volatility_window
            ).std().fillna(0)
        
        # Order flow imbalance
        if self.config.use_order_flow:
            bid_vol = self.data['bid_size_1']
            ask_vol = self.data['ask_size_1']
            self.data['ofi'] = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        
        # Market depth
        if self.config.use_lob_features:
            bid_depth = sum(self.data[f'bid_size_{i}'] for i in range(1, self.config.n_levels + 1))
            ask_depth = sum(self.data[f'ask_size_{i}'] for i in range(1, self.config.n_levels + 1))
            self.data['market_depth'] = bid_depth + ask_depth
            self.data['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
    
    def _get_observation_size(self) -> int:
        """Calculate the size of the observation space."""
        size = 4  # time, inventory, price, price_change
        
        if self.config.use_lob_features:
            size += 2  # spread, market_depth
        
        if self.config.use_order_flow:
            size += 1  # ofi
        
        if self.config.use_volatility:
            size += 1  # volatility
        
        return size
    
    def _compute_features(self):
        """Compute features for the current timestep."""
        if self.current_idx >= len(self.data):
            return
        
        row = self.data.iloc[self.current_idx]
        
        # Basic features
        self.current_midprice = row['midprice']
        self.current_spread = row['spread']
        self.current_price_change = row['price_change']
        
        # Optional features
        if self.config.use_lob_features:
            self.current_market_depth = row.get('market_depth', 0)
            self.current_depth_imbalance = row.get('depth_imbalance', 0)
        
        if self.config.use_order_flow:
            self.current_ofi = row.get('ofi', 0)
        
        if self.config.use_volatility:
            self.current_volatility = row.get('volatility', 0)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (can include 'start_idx')
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state
        if options and 'start_idx' in options:
            self.current_idx = options['start_idx']
        else:
            self.current_idx = self.config.start_idx
        
        self.q = 0
        self.X = 0.0
        self.step_count = 0
        
        # Reset history
        self.price_history = []
        self.inventory_history = []
        self.pnl_history = []
        self.trade_history = []
        
        # Reset quotes
        self.bid_price = None
        self.ask_price = None
        
        # Compute initial features
        self._compute_features()
        
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
        self.bid_price = self.current_midprice - delta_b
        self.ask_price = self.current_midprice + delta_a
        
        # Check for market orders (simulate based on LOB data)
        trade_occurred = self._check_market_orders()
        
        # Move to next timestep
        self.current_idx += 1
        self.step_count += 1
        
        # Check if we've reached the end of data
        if self.current_idx >= len(self.data):
            terminated = True
            truncated = False
        else:
            # Compute features for new timestep
            self._compute_features()
            
            # Check termination
            terminated = self.step_count >= self.config.episode_length
            truncated = False
        
        # Record history
        self.price_history.append(self.current_midprice)
        self.inventory_history.append(self.q)
        self.pnl_history.append(self._get_pnl())
        
        # Compute reward
        reward = self._compute_reward(trade_occurred)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _check_market_orders(self) -> bool:
        """
        Check if market orders arrive and execute trades.
        
        This simulates market orders based on the LOB data:
        - If the market price moves through our bid/ask, we get filled
        - Probability of fill depends on our spread relative to market spread
        
        Returns:
            True if a trade occurred, False otherwise
        """
        if self.bid_price is None or self.ask_price is None:
            return False
        
        # Get next price (if available)
        if self.current_idx + 1 >= len(self.data):
            return False
        
        next_midprice = self.data.iloc[self.current_idx + 1]['midprice']
        
        # Check if our bid is hit (price goes down through our bid)
        if next_midprice <= self.bid_price:
            if self.q < self.config.Q_max:
                self.q += 1
                self.X -= self.bid_price * (1 + self.config.transaction_cost)
                self.trade_history.append({
                    'time': self.step_count,
                    'side': 'sell',
                    'price': self.bid_price,
                    'inventory': self.q
                })
                return True
        
        # Check if our ask is lifted (price goes up through our ask)
        if next_midprice >= self.ask_price:
            if self.q > -self.config.Q_max:
                self.q -= 1
                self.X += self.ask_price * (1 - self.config.transaction_cost)
                self.trade_history.append({
                    'time': self.step_count,
                    'side': 'buy',
                    'price': self.ask_price,
                    'inventory': self.q
                })
                return True
        
        # Alternative: probabilistic fill based on spread
        # If our spread is tighter than market spread, higher probability of fill
        market_spread = self.current_spread
        our_spread = self.ask_price - self.bid_price
        
        if our_spread < market_spread:
            fill_prob = 0.1 * (market_spread / our_spread)
        else:
            fill_prob = 0.01
        
        # Random fill
        if np.random.rand() < fill_prob:
            # Random side
            if np.random.rand() < 0.5 and self.q < self.config.Q_max:
                self.q += 1
                self.X -= self.bid_price * (1 + self.config.transaction_cost)
                self.trade_history.append({
                    'time': self.step_count,
                    'side': 'sell',
                    'price': self.bid_price,
                    'inventory': self.q
                })
                return True
            elif self.q > -self.config.Q_max:
                self.q -= 1
                self.X += self.ask_price * (1 - self.config.transaction_cost)
                self.trade_history.append({
                    'time': self.step_count,
                    'side': 'buy',
                    'price': self.ask_price,
                    'inventory': self.q
                })
                return True
        
        return False
    
    def _get_pnl(self) -> float:
        """Get current PnL (realized + unrealized)."""
        realized_pnl = self.X
        unrealized_pnl = self.q * self.current_midprice
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
        
        elif self.config.reward_type == "spread_profit":
            # Reward is spread profit when trade occurs
            if trade_occurred:
                return (self.ask_price - self.bid_price) / 2
            return 0.0
        
        else:
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Get normalized observation.
        
        Returns:
            Normalized observation array
        """
        # Normalize time
        norm_time = self.step_count / self.config.episode_length
        
        # Normalize inventory
        norm_inventory = self.q / self.config.Q_max
        
        # Normalize price (relative to initial price)
        initial_price = self.data.iloc[self.config.start_idx]['midprice']
        norm_price = self.current_midprice / initial_price
        
        # Price change
        price_change = self.current_price_change
        
        obs = [norm_time, norm_inventory, norm_price, price_change]
        
        # Add optional features
        if self.config.use_lob_features:
            obs.append(self.current_spread / self.current_midprice)
            obs.append(self.current_market_depth)
        
        if self.config.use_order_flow:
            obs.append(self.current_ofi)
        
        if self.config.use_volatility:
            obs.append(self.current_volatility)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            'time': self.step_count,
            'midprice': self.current_midprice,
            'inventory': self.q,
            'pnl': self._get_pnl(),
            'bid_price': self.bid_price,
            'ask_price': self.ask_price,
            'spread': self.ask_price - self.bid_price if self.bid_price and self.ask_price else 0.0,
            'n_trades': len(self.trade_history),
            'data_idx': self.current_idx
        }
    
    def render(self, mode: str = "human") -> None:
        """
        Render the environment.
        
        Args:
            mode: Render mode
        """
        if mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"Data Index: {self.current_idx} / {len(self.data)}")
            print(f"Midprice: {self.current_midprice:.4f}")
            print(f"Inventory: {self.q}")
            print(f"PnL: {self._get_pnl():.4f}")
            print(f"Bid: {self.bid_price:.4f}" if self.bid_price else "Bid: None")
            print(f"Ask: {self.ask_price:.4f}" if self.ask_price else "Ask: None")
            print(f"Spread: {self.ask_price - self.bid_price:.4f}" if self.bid_price and self.ask_price else "Spread: N/A")
            print(f"Trades: {len(self.trade_history)}")
    
    def close(self) -> None:
        """Clean up the environment."""
        pass


def create_lob_env(
    data_source: str = "lobster",
    data_path: Optional[str] = None,
    message_file: Optional[str] = None,
    orderbook_file: Optional[str] = None,
    symbol: str = "BTCUSDT",
    episode_length: int = 1000,
    reward_type: str = "pnl",
    seed: Optional[int] = None
) -> LOBMarketMakingEnv:
    """
    Create a LOB market making environment with default configuration.
    
    Args:
        data_source: Data source ("lobster" or "binance")
        data_path: Path to data directory (for Binance)
        message_file: Path to LOBSTER message file
        orderbook_file: Path to LOBSTER orderbook file
        symbol: Trading symbol (for Binance)
        episode_length: Number of timesteps per episode
        reward_type: Type of reward function
        seed: Random seed
        
    Returns:
        LOBMarketMakingEnv instance
    """
    config = LOBMarketMakingEnvConfig(
        data_source=data_source,
        data_path=data_path,
        message_file=message_file,
        orderbook_file=orderbook_file,
        symbol=symbol,
        episode_length=episode_length,
        reward_type=reward_type,
        seed=seed
    )
    return LOBMarketMakingEnv(config)


if __name__ == "__main__":
    # Example usage
    print("Creating LOB market making environment...")
    print("\nExample for LOBSTER data:")
    print("env = create_lob_env(")
    print("    data_source='lobster',")
    print("    message_file='data/AAPL_message.csv',")
    print("    orderbook_file='data/AAPL_orderbook.csv',")
    print("    episode_length=1000,")
    print("    reward_type='pnl',")
    print("    seed=42")
    print(")")
    
    print("\nExample for Binance data:")
    print("env = create_lob_env(")
    print("    data_source='binance',")
    print("    data_path='data/binance/',")
    print("    symbol='BTCUSDT',")
    print("    episode_length=1000,")
    print("    reward_type='pnl',")
    print("    seed=42")
    print(")")
