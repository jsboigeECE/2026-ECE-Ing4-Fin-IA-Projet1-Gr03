"""
LOB (Limit Order Book) Data Loader

This module provides utilities to load and process LOB data from various sources:
- LOBSTER: Historical limit order book data
- Binance: Real-time and historical order book data

The loader provides a unified interface for accessing LOB data in a format
compatible with the market making RL environment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import warnings


@dataclass
class LOBDataConfig:
    """Configuration for LOB data loading."""
    
    # Data source
    source: str = "lobster"  # "lobster" or "binance"
    
    # File paths
    data_path: Optional[str] = None
    message_file: Optional[str] = None
    orderbook_file: Optional[str] = None
    
    # LOB parameters
    n_levels: int = 10  # Number of price levels to load
    tick_size: float = 0.01  # Minimum price increment
    
    # Time parameters
    start_time: Optional[str] = None  # Format: "HH:MM:SS"
    end_time: Optional[str] = None
    
    # Asset parameters
    symbol: str = "BTCUSDT"  # For Binance
    asset_name: str = "AAPL"  # For LOBSTER
    
    # Data preprocessing
    normalize: bool = True  # Normalize prices and volumes
    fill_missing: bool = True  # Fill missing order book levels
    max_spread: Optional[float] = None  # Filter out large spreads


class LOBDataLoader:
    """
    Unified LOB data loader for multiple data sources.
    
    Supports:
    - LOBSTER: Historical limit order book data with message and orderbook files
    - Binance: Historical order book depth data (snapshots)
    """
    
    def __init__(self, config: LOBDataConfig):
        """
        Initialize the LOB data loader.
        
        Args:
            config: Configuration for data loading
        """
        self.config = config
        self.data = None
        self.messages = None
        self.orderbook = None
        
    def load(self) -> pd.DataFrame:
        """
        Load LOB data from the configured source.
        
        Returns:
            DataFrame with LOB data including:
            - timestamp: Time of the snapshot
            - midprice: Mid price (bid + ask) / 2
            - bid_prices: Array of bid prices
            - ask_prices: Array of ask prices
            - bid_volumes: Array of bid volumes
            - ask_volumes: Array of ask volumes
            - spread: Ask - bid spread
        """
        if self.config.source == "lobster":
            return self._load_lobster()
        elif self.config.source == "binance":
            return self._load_binance()
        else:
            raise ValueError(f"Unknown data source: {self.config.source}")
    
    def _load_lobster(self) -> pd.DataFrame:
        """
        Load LOBSTER data format.
        
        LOBSTER provides two files:
        - message file: Contains order events (submissions, cancellations, executions)
        - orderbook file: Contains order book snapshots at each event
        
        Returns:
            DataFrame with LOB data
        """
        if self.config.message_file is None or self.config.orderbook_file is None:
            raise ValueError(
                "For LOBSTER data, both message_file and orderbook_file must be specified"
            )
        
        # Load message file
        # Format: time, type, order_id, size, price, direction
        message_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
        self.messages = pd.read_csv(
            self.config.message_file,
            header=None,
            names=message_cols,
            sep=' '
        )
        
        # Load orderbook file
        # Format: ask_price_1, ask_size_1, ..., bid_price_1, bid_size_1, ...
        n_cols = 4 * self.config.n_levels
        orderbook_cols = []
        for i in range(self.config.n_levels):
            orderbook_cols.extend([f'ask_price_{i+1}', f'ask_size_{i+1}'])
        for i in range(self.config.n_levels):
            orderbook_cols.extend([f'bid_price_{i+1}', f'bid_size_{i+1}'])
        
        self.orderbook = pd.read_csv(
            self.config.orderbook_file,
            header=None,
            names=orderbook_cols,
            sep=' '
        )
        
        # Combine data
        self.orderbook['timestamp'] = self.messages['time']
        
        # Filter by time if specified
        if self.config.start_time is not None:
            start_seconds = self._time_to_seconds(self.config.start_time)
            mask = self.orderbook['timestamp'] >= start_seconds
            self.orderbook = self.orderbook[mask]
            self.messages = self.messages[mask]
        
        if self.config.end_time is not None:
            end_seconds = self._time_to_seconds(self.config.end_time)
            mask = self.orderbook['timestamp'] <= end_seconds
            self.orderbook = self.orderbook[mask]
            self.messages = self.messages[mask]
        
        # Process orderbook
        return self._process_orderbook()
    
    def _load_binance(self) -> pd.DataFrame:
        """
        Load Binance order book depth data.
        
        Binance provides order book snapshots in JSON format with:
        - lastUpdateId: Last update ID
        - bids: Array of [price, quantity] for bids
        - asks: Array of [price, quantity] for asks
        
        Returns:
            DataFrame with LOB data
        """
        if self.config.data_path is None:
            raise ValueError("For Binance data, data_path must be specified")
        
        data_path = Path(self.config.data_path)
        
        # Load all JSON files in the directory
        json_files = list(data_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {data_path}")
        
        all_data = []
        
        for json_file in json_files:
            import json
            with open(json_file, 'r') as f:
                snapshot = json.load(f)
            
            # Extract bids and asks
            bids = snapshot.get('bids', [])[:self.config.n_levels]
            asks = snapshot.get('asks', [])[:self.config.n_levels]
            
            # Create row
            row = {
                'timestamp': snapshot.get('lastUpdateId', 0),
                'midprice': (float(bids[0][0]) + float(asks[0][0])) / 2 if bids and asks else np.nan
            }
            
            # Add bid levels
            for i, (price, size) in enumerate(bids):
                row[f'bid_price_{i+1}'] = float(price)
                row[f'bid_size_{i+1}'] = float(size)
            
            # Add ask levels
            for i, (price, size) in enumerate(asks):
                row[f'ask_price_{i+1}'] = float(price)
                row[f'ask_size_{i+1}'] = float(size)
            
            all_data.append(row)
        
        self.orderbook = pd.DataFrame(all_data)
        
        # Process orderbook
        return self._process_orderbook()
    
    def _process_orderbook(self) -> pd.DataFrame:
        """
        Process the loaded orderbook data.
        
        Returns:
            Processed DataFrame with derived features
        """
        # Calculate midprice
        bid_price_1 = self.orderbook['bid_price_1']
        ask_price_1 = self.orderbook['ask_price_1']
        self.orderbook['midprice'] = (bid_price_1 + ask_price_1) / 2
        
        # Calculate spread
        self.orderbook['spread'] = ask_price_1 - bid_price_1
        self.orderbook['spread_pct'] = (ask_price_1 - bid_price_1) / self.orderbook['midprice']
        
        # Filter out large spreads if specified
        if self.config.max_spread is not None:
            mask = self.orderbook['spread'] <= self.config.max_spread
            self.orderbook = self.orderbook[mask]
        
        # Fill missing values if specified
        if self.config.fill_missing:
            self.orderbook = self.orderbook.fillna(method='ffill').fillna(method='bfill')
        
        # Normalize if specified
        if self.config.normalize:
            self.orderbook = self._normalize_data()
        
        return self.orderbook
    
    def _normalize_data(self) -> pd.DataFrame:
        """
        Normalize the orderbook data.
        
        Returns:
            Normalized DataFrame
        """
        df = self.orderbook.copy()
        
        # Normalize prices relative to midprice
        for i in range(1, self.config.n_levels + 1):
            df[f'bid_price_{i}'] = df[f'bid_price_{i}'] / df['midprice'] - 1
            df[f'ask_price_{i}'] = df[f'ask_price_{i}'] / df['midprice'] - 1
        
        # Normalize volumes
        for i in range(1, self.config.n_levels + 1):
            total_volume = df[f'bid_size_{i}'].sum() + df[f'ask_size_{i}'].sum()
            if total_volume > 0:
                df[f'bid_size_{i}'] = df[f'bid_size_{i}'] / total_volume
                df[f'ask_size_{i}'] = df[f'ask_size_{i}'] / total_volume
        
        return df
    
    def _time_to_seconds(self, time_str: str) -> float:
        """
        Convert time string to seconds since midnight.
        
        Args:
            time_str: Time string in format "HH:MM:SS"
            
        Returns:
            Seconds since midnight
        """
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    def get_midprice_series(self) -> pd.Series:
        """Get the midprice time series."""
        if self.orderbook is None:
            self.load()
        return self.orderbook['midprice']
    
    def get_spread_series(self) -> pd.Series:
        """Get the spread time series."""
        if self.orderbook is None:
            self.load()
        return self.orderbook['spread']
    
    def get_orderbook_snapshot(self, index: int) -> Dict:
        """
        Get a single orderbook snapshot.
        
        Args:
            index: Index of the snapshot
            
        Returns:
            Dictionary with orderbook data
        """
        if self.orderbook is None:
            self.load()
        
        row = self.orderbook.iloc[index]
        
        snapshot = {
            'timestamp': row['timestamp'],
            'midprice': row['midprice'],
            'spread': row['spread'],
            'bids': [],
            'asks': []
        }
        
        for i in range(1, self.config.n_levels + 1):
            snapshot['bids'].append({
                'price': row[f'bid_price_{i}'],
                'size': row[f'bid_size_{i}']
            })
            snapshot['asks'].append({
                'price': row[f'ask_price_{i}'],
                'size': row[f'ask_size_{i}']
            })
        
        return snapshot
    
    def get_price_changes(self, window: int = 1) -> pd.Series:
        """
        Calculate price changes over a window.
        
        Args:
            window: Window size for price change calculation
            
        Returns:
            Series of price changes
        """
        midprice = self.get_midprice_series()
        return midprice.pct_change(window)
    
    def get_volatility(self, window: int = 100) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            window: Window size for volatility calculation
            
        Returns:
            Series of rolling volatility
        """
        returns = self.get_price_changes()
        return returns.rolling(window=window).std()
    
    def get_order_flow_imbalance(self) -> pd.Series:
        """
        Calculate order flow imbalance.
        
        OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        Returns:
            Series of order flow imbalance
        """
        if self.orderbook is None:
            self.load()
        
        bid_vol = self.orderbook['bid_size_1']
        ask_vol = self.orderbook['ask_size_1']
        
        ofi = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        return ofi
    
    def get_market_depth(self, n_levels: int = 5) -> pd.Series:
        """
        Calculate market depth (total volume at top n levels).
        
        Args:
            n_levels: Number of levels to consider
            
        Returns:
            Series of market depth
        """
        if self.orderbook is None:
            self.load()
        
        bid_depth = sum(self.orderbook[f'bid_size_{i}'] for i in range(1, n_levels + 1))
        ask_depth = sum(self.orderbook[f'ask_size_{i}'] for i in range(1, n_levels + 1))
        
        return bid_depth + ask_depth


def create_lobster_loader(
    message_file: str,
    orderbook_file: str,
    n_levels: int = 10,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    normalize: bool = True
) -> LOBDataLoader:
    """
    Create a LOBSTER data loader.
    
    Args:
        message_file: Path to LOBSTER message file
        orderbook_file: Path to LOBSTER orderbook file
        n_levels: Number of price levels to load
        start_time: Start time filter (format: "HH:MM:SS")
        end_time: End time filter (format: "HH:MM:SS")
        normalize: Whether to normalize the data
        
    Returns:
        LOBDataLoader instance
    """
    config = LOBDataConfig(
        source="lobster",
        message_file=message_file,
        orderbook_file=orderbook_file,
        n_levels=n_levels,
        start_time=start_time,
        end_time=end_time,
        normalize=normalize
    )
    return LOBDataLoader(config)


def create_binance_loader(
    data_path: str,
    symbol: str = "BTCUSDT",
    n_levels: int = 10,
    normalize: bool = True
) -> LOBDataLoader:
    """
    Create a Binance data loader.
    
    Args:
        data_path: Path to directory containing Binance JSON files
        symbol: Trading symbol (e.g., "BTCUSDT")
        n_levels: Number of price levels to load
        normalize: Whether to normalize the data
        
    Returns:
        LOBDataLoader instance
    """
    config = LOBDataConfig(
        source="binance",
        data_path=data_path,
        symbol=symbol,
        n_levels=n_levels,
        normalize=normalize
    )
    return LOBDataLoader(config)


if __name__ == "__main__":
    # Example usage
    print("LOB Data Loader")
    print("=" * 50)
    
    # Example for LOBSTER data
    print("\nExample: Loading LOBSTER data")
    print("loader = create_lobster_loader(")
    print("    message_file='data/AAPL_2012-06-21_34200000_57600000_message_1.csv',")
    print("    orderbook_file='data/AAPL_2012-06-21_34200000_57600000_orderbook_1.csv',")
    print("    n_levels=10")
    print(")")
    print("data = loader.load()")
    print("print(data.head())")
    
    # Example for Binance data
    print("\nExample: Loading Binance data")
    print("loader = create_binance_loader(")
    print("    data_path='data/binance/',")
    print("    symbol='BTCUSDT',")
    print("    n_levels=10")
    print(")")
    print("data = loader.load()")
    print("print(data.head())")
