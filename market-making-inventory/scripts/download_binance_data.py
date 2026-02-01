"""
Download Binance Order Book Data

This script provides utilities to download order book depth data from Binance API.
The downloaded data can be used for training RL agents with real LOB data.

Usage:
    # Download current order book snapshot
    python download_binance_data.py --symbol BTCUSDT --output data/binance/
    
    # Download multiple snapshots
    python download_binance_data.py --symbol BTCUSDT --output data/binance/ --n-snapshots 100 --interval 60
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Install with: pip install requests")


class BinanceDataDownloader:
    """Downloader for Binance order book data."""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialize the downloader.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
        """
        self.symbol = symbol
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_order_book(self, limit: int = 1000) -> dict:
        """
        Get current order book depth.
        
        Args:
            limit: Number of price levels (max 5000)
            
        Returns:
            Dictionary with order book data
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required. Install with: pip install requests")
        
        url = f"{self.base_url}/depth"
        params = {"symbol": self.symbol, "limit": limit}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching order book: {e}")
            return {}
    
    def get_order_book_snapshot(self, limit: int = 1000) -> dict:
        """
        Get order book snapshot with metadata.
        
        Args:
            limit: Number of price levels
            
        Returns:
            Dictionary with snapshot data including timestamp
        """
        data = self.get_order_book(limit)
        
        if data:
            snapshot = {
                "symbol": self.symbol,
                "timestamp": datetime.now().isoformat(),
                "lastUpdateId": data.get("lastUpdateId", 0),
                "bids": data.get("bids", []),
                "asks": data.get("asks", [])
            }
            return snapshot
        
        return {}
    
    def download_snapshots(
        self,
        output_dir: str,
        n_snapshots: int = 1,
        interval: int = 60,
        limit: int = 1000
    ) -> List[str]:
        """
        Download multiple order book snapshots.
        
        Args:
            output_dir: Directory to save snapshots
            n_snapshots: Number of snapshots to download
            interval: Interval between snapshots in seconds
            limit: Number of price levels per snapshot
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i in range(n_snapshots):
            print(f"Downloading snapshot {i+1}/{n_snapshots}...")
            
            snapshot = self.get_order_book_snapshot(limit)
            
            if snapshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.symbol}_{timestamp}.json"
                filepath = output_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(snapshot, f, indent=2)
                
                saved_files.append(str(filepath))
                print(f"  Saved to {filepath}")
            else:
                print(f"  Failed to download snapshot {i+1}")
            
            # Wait before next snapshot
            if i < n_snapshots - 1:
                print(f"  Waiting {interval} seconds...")
                time.sleep(interval)
        
        return saved_files
    
    def get_server_time(self) -> Optional[dict]:
        """
        Get Binance server time.
        
        Returns:
            Dictionary with server time
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = requests.get(f"{self.base_url}/time", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching server time: {e}")
            return None
    
    def get_exchange_info(self) -> Optional[dict]:
        """
        Get exchange information for the symbol.
        
        Returns:
            Dictionary with exchange info
        """
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = requests.get(f"{self.base_url}/exchangeInfo", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find symbol info
            for symbol_info in data.get("symbols", []):
                if symbol_info["symbol"] == self.symbol:
                    return symbol_info
            
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching exchange info: {e}")
            return None


def print_order_book_summary(snapshot: dict):
    """
    Print a summary of the order book snapshot.
    
    Args:
        snapshot: Order book snapshot dictionary
    """
    print("\n" + "="*50)
    print(f"Order Book Summary - {snapshot.get('symbol', 'Unknown')}")
    print(f"Timestamp: {snapshot.get('timestamp', 'Unknown')}")
    print(f"Last Update ID: {snapshot.get('lastUpdateId', 'Unknown')}")
    print("="*50)
    
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    
    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        midprice = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / midprice) * 100
        
        print(f"\nBest Bid: {best_bid:.8f}")
        print(f"Best Ask: {best_ask:.8f}")
        print(f"Midprice: {midprice:.8f}")
        print(f"Spread: {spread:.8f} ({spread_pct:.4f}%)")
        
        # Top 5 levels
        print("\nTop 5 Bid Levels:")
        print(f"{'Price':<15} {'Volume':<15}")
        print("-" * 30)
        for i, (price, volume) in enumerate(bids[:5]):
            print(f"{float(price):<15.8f} {float(volume):<15.8f}")
        
        print("\nTop 5 Ask Levels:")
        print(f"{'Price':<15} {'Volume':<15}")
        print("-" * 30)
        for i, (price, volume) in enumerate(asks[:5]):
            print(f"{float(price):<15.8f} {float(volume):<15.8f}")
    else:
        print("\nNo order book data available.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download Binance Order Book Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single snapshot
  python download_binance_data.py --symbol BTCUSDT --output data/binance/
  
  # Download multiple snapshots
  python download_binance_data.py --symbol BTCUSDT --output data/binance/ \\
      --n-snapshots 100 --interval 60
  
  # Download with more price levels
  python download_binance_data.py --symbol ETHUSDT --output data/binance/ \\
      --limit 5000 --n-snapshots 10
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/binance/',
        help='Output directory for downloaded data'
    )
    
    parser.add_argument(
        '--n-snapshots',
        type=int,
        default=1,
        help='Number of snapshots to download'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Interval between snapshots in seconds'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='Number of price levels (max 5000)'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary of downloaded data'
    )
    
    parser.add_argument(
        '--check-connection',
        action='store_true',
        help='Check connection to Binance API'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = BinanceDataDownloader(args.symbol)
    
    # Check connection if requested
    if args.check_connection:
        print("Checking connection to Binance API...")
        
        server_time = downloader.get_server_time()
        if server_time:
            print(f"✓ Server time: {server_time}")
        else:
            print("✗ Failed to get server time")
            return
        
        exchange_info = downloader.get_exchange_info()
        if exchange_info:
            print(f"✓ Symbol {args.symbol} found")
            print(f"  Status: {exchange_info.get('status')}")
            print(f"  Base Asset: {exchange_info.get('baseAsset')}")
            print(f"  Quote Asset: {exchange_info.get('quoteAsset')}")
        else:
            print(f"✗ Symbol {args.symbol} not found")
            return
        
        print("\nConnection check successful!")
        return
    
    # Download snapshots
    print(f"Downloading {args.n_snapshots} snapshot(s) for {args.symbol}...")
    print(f"Output directory: {args.output}")
    print(f"Price levels: {args.limit}")
    
    saved_files = downloader.download_snapshots(
        output_dir=args.output,
        n_snapshots=args.n_snapshots,
        interval=args.interval,
        limit=args.limit
    )
    
    print(f"\nDownload complete! Saved {len(saved_files)} file(s).")
    
    # Print summary if requested
    if args.summary and saved_files:
        print("\nLoading first snapshot for summary...")
        with open(saved_files[0], 'r') as f:
            snapshot = json.load(f)
        print_order_book_summary(snapshot)


if __name__ == "__main__":
    main()
