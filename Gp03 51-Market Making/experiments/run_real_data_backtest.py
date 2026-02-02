import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.simulator import OrderBookSimulator, SimulationConfig
from src.data.real_data_fetcher import BinanceDataFetcher
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters
from src.solvers.hjb_solver import HJBSolver
from stable_baselines3 import PPO

def run_real_backtest():
    print("Starting Real Data Backtest (Binance BTC/USDT)...")
    
    # 1. Fetch Real Data
    fetcher = BinanceDataFetcher(symbol='BTC/USDT', timeframe='1m')
    try:
        # Fetch last 1000 minutes (~16 hours)
        price_data = fetcher.get_price_path(limit=1000)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    S0 = price_data[0]
    T_total = len(price_data) / (24*60) # Fraction of days? No, T usually in years.
    # 1 minute steps. T = N_minutes / (252 * 24 * 60) for annualization?
    # Let's approximate: 1 year = 525600 minutes.
    dt = 1.0 / 525600.0
    T_horizon = len(price_data) * dt
    
    # Estimate Volatility from data
    # log returns
    returns = np.diff(np.log(price_data))
    sigma_est = np.std(returns) / np.sqrt(dt)
    print(f"Estimated Annualized Volatility: {sigma_est:.2f}")

    # Configuration
    # Calibrate Parameters for BTC (High Price, High Frequency)
    scale_factor = S0 / 100.0
    k_btc = 1.5 / scale_factor
    A_btc = 500000.0 # High intensity for HFT frequency
    
    print(f"Calibrated Parameters: k={k_btc:.6f}, A={A_btc:.0f}")

    config = SimulationConfig(S0=S0, T=T_horizon, dt=dt, sigma=sigma_est, k=k_btc, A=A_btc) 
    params = MarketParameters(sigma=sigma_est, gamma=0.1, k=k_btc, A=A_btc)
    
    # Initialize Models
    hjb_model = InventoryHJBModel(params)
    solver_hjb = HJBSolver(params, max_inventory_Q=50) # Larger inventory for real BTC
    
    # Load RL Model (if exists)
    rl_model = None
    if os.path.exists("models/ppo_market_maker.zip"):
        print("Loading RL Model...")
        rl_model = PPO.load("models/ppo_market_maker.zip")
    else:
        print("Warning: RL Model not found. Skipping RL strategy.")

    # Strategies definitions
    def strategy_naive(q, t_left):
        fixed_spread = S0 * 0.0005 # 5 bps
        return fixed_spread, fixed_spread
        
    def strategy_hjb_exact(q, t_left):
        # Matrix Exponential exact solution
        # Note: t_left is small here, solver might be unstable if t -> 0
        # Use a minimum t_left to avoid singularity or keep it stationary
        safe_t = max(t_left, 0.01) 
        return solver_hjb.get_optimal_quotes(q, safe_t)
        
    def strategy_rl(q, t_left):
        if rl_model is None:
            return S0 * 0.01, S0 * 0.01 # Fallback
            
        current_time_norm = (T_horizon - t_left) / T_horizon
        # Clip inventory to training range if necessary [-50, 50]
        q_clipped = np.clip(q, -50, 50)
        
        obs = np.array([float(q_clipped), current_time_norm], dtype=np.float32)
        action, _ = rl_model.predict(obs, deterministic=True)
        # RL Output is spread [0, 5].
        # BUT RL was trained on S=100. BTC S=90000.
        # We need to scale the RL output spread to current price levels!
        # Assuming RL learned "spread in dollars" for S=100.
        # 1 unit spread at S=100 is 1%.
        # So we scale by scale_factor
        return action[0] * scale_factor, action[1] * scale_factor

    # Run Simulations
    results = {}
    strategies = {
        'Naive Real': strategy_naive,
        'HJB Real': strategy_hjb_exact
    }
    if rl_model:
        strategies['RL Agent'] = strategy_rl
    
    for name, strat in strategies.items():
        print(f"Running {name}...")
        # Pass the real price path!
        sim = OrderBookSimulator(config, price_path=price_data)
        df = sim.run_simulation(strat)
        results[name] = df
        
        print(f"{name} Final PnL: {df['pnl'].iloc[-1]:.2f}")
        
    # Plotting
    plot_results(results, price_data)
    print("Real Data Backtest Complete. Results saved in experiments/results/")

def plot_results(results, price_data):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    # 1. PnL Trajectories
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['pnl'], label=name)
    plt.title('PnL on Real BTC/USDT Data')
    plt.xlabel('Steps (Minutes)')
    plt.ylabel('PnL ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/results/real_pnl.png')
    plt.close()
    
    # 2. Price Path
    plt.figure(figsize=(10, 6))
    plt.plot(price_data, color='black', alpha=0.5, label='BTC Price')
    plt.title('Underlying Price Path')
    plt.legend()
    plt.savefig('experiments/results/real_price.png')
    plt.close()
    
    # 3. Inventory for HJB
    if 'HJB Real' in results:
        df = results['HJB Real']
        plt.figure(figsize=(10, 6))
        plt.plot(df['inventory'], color='purple')
        plt.title('HJB Inventory on Real Data')
        plt.grid(True)
        plt.savefig('experiments/results/real_inventory.png')
        plt.close()

if __name__ == "__main__":
    run_real_backtest()
