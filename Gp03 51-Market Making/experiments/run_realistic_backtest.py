import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.simulator import OrderBookSimulator, SimulationConfig
from src.data.real_data_fetcher import BinanceDataFetcher
from src.models.inventory_hjb import InventoryHJBModel, MarketParameters
from src.solvers.hjb_solver import HJBSolver

def run_realistic_backtest():
    print("Starting REALISTIC Backtest (Fees + Toxic Flow)...")
    print("Fetching data...")
    
    # 1. Fetch Real Data
    fetcher = BinanceDataFetcher(symbol='BTC/USDT', timeframe='1m')
    try:
        # Fetch last 10000 minutes (~7 days)
        price_data = fetcher.get_price_path(limit=10000)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    # 2. Parameters
    S0 = price_data[0]
    dt = 1.0 / 525600.0
    T_horizon = len(price_data) * dt
    returns = np.diff(np.log(price_data))
    sigma_est = np.std(returns) / np.sqrt(dt)

    # Calibrate
    scale_factor = S0 / 100.0
    k_btc = 1.5 / scale_factor
    A_btc = 500000.0
    
    # --- REALISTIC SETTINGS ---
    # Maker Fee: Binance Level 0 is 0.1% (taker) / 0.1% (maker).
    # VIP levels or rebates might be lower. Let's use 0.01% (1 basis point) as a "good" fee tier.
    MAKER_FEE = 0.0001 
    # Enable Sniping (Toxic Flow)
    ALLOW_SNIPING = True

    config = SimulationConfig(
        S0=S0, T=T_horizon, dt=dt, sigma=sigma_est, 
        k=k_btc, A=A_btc,
        maker_fee=MAKER_FEE,
        allow_sniping=ALLOW_SNIPING
    ) 
    
    params = MarketParameters(sigma=sigma_est, gamma=0.1, k=k_btc, A=A_btc)
    
    # Models
    solver_hjb = HJBSolver(params, max_inventory_Q=50)
    
    # RL
    rl_model = None
    model_path = "models/ppo_market_maker.zip"
    if not os.path.exists(model_path):
        # Try checking parent dir if running from experiments/
        model_path = "../models/ppo_market_maker.zip"
    
    if os.path.exists(model_path):
        print(f"Loading RL Model from {model_path}...")
        rl_model = PPO.load(model_path)
    else:
        print("Warning: RL Model not found at models/ppo_market_maker.zip. Skipping RL.")

    # Strategies
    def strategy_naive(q, t_left):
        fixed_spread = S0 * 0.0005 # 5 bps
        return fixed_spread, fixed_spread
        
    def strategy_hjb_exact(q, t_left):
        safe_t = max(t_left, 0.01) 
        return solver_hjb.get_optimal_quotes(q, safe_t)
        
    def strategy_rl(q, t_left):
        if rl_model is None: return S0*0.01, S0*0.01
        current_time_norm = (T_horizon - t_left) / T_horizon
        q_clipped = np.clip(q, -50, 50)
        obs = np.array([float(q_clipped), current_time_norm], dtype=np.float32)
        action, _ = rl_model.predict(obs, deterministic=True)
        return action[0] * scale_factor, action[1] * scale_factor

    strategies = {
        'Naive (Realistic)': strategy_naive,
        'HJB (Realistic)': strategy_hjb_exact
    }
    if rl_model:
        strategies['RL (Realistic)'] = strategy_rl

    # Run
    results = {}
    for name, strat in strategies.items():
        print(f"Running {name}...")
        sim = OrderBookSimulator(config, price_path=price_data)
        df = sim.run_simulation(strat)
        results[name] = df
        print(f"{name} Final PnL: {df['pnl'].iloc[-1]:.2f}")

    plot_results(results, price_data)
    print("Done. Results in experiments/results/realistic_*.png")

def plot_results(results, price_data):
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['pnl'], label=name)
    plt.title(f'Realistic PnL (Fees=1bp, Sniping=True)')
    plt.ylabel('PnL ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/results/realistic_pnl.png')
    plt.close()

if __name__ == "__main__":
    run_realistic_backtest()
