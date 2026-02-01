# Usage Guide

This guide provides detailed instructions for using the market making with inventory constraints package.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Running Backtests](#running-backtests)
5. [Training RL Agents](#training-rl-agents)
6. [Custom Strategies](#custom-strategies)
7. [Reproducing Results](#reproducing-results)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/market-making-inventory.git
cd market-making-inventory
```

### Step 2: Create Virtual Environment

```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt

# With RL support
pip install -r requirements.txt
pip install stable-baselines3
```

### Step 4: Verify Installation

```bash
python -c "from src.models import create_default_model; print('Installation successful!')"
```

---

## Quick Start

### Running Your First Backtest

```bash
# Run backtests with default strategies
python experiments/run_backtest.py --seed 42
```

This will:
1. Initialize the HJB model
2. Run simulations for HJB, constant spread, and inventory-aware strategies
3. Generate comparison plots in the `results/` directory
4. Print performance statistics

### Using the Python API

```python
from src.models import create_default_model
from src.data import create_default_simulator

# Create model and simulator
model = create_default_model()
simulator = create_default_simulator(seed=42)

# Define policy
def my_policy(t, q, S):
    delta_b, delta_a = model.get_optimal_spreads(t, q)
    return S - delta_b, S + delta_a

# Run simulation
results = simulator.run_simulation(my_policy)

# Print results
print(f"Final PnL: {results['final_pnl']:.4f}")
print(f"Trades: {results['n_trades']}")
```

---

## Configuration

### Model Parameters

The [`HJBParameters`](../src/models/inventory_hjb.py) class controls the model behavior:

| Parameter | Default | Description |
|-----------|----------|-------------|
| `sigma` | 0.2 | Volatility of the asset |
| `A` | 1.0 | Base order arrival intensity |
| `k` | 0.5 | Market depth (sensitivity to spread) |
| `gamma` | 0.1 | Risk aversion coefficient |
| `phi` | 0.01 | Terminal liquidation cost |
| `T` | 1.0 | Time horizon |
| `dt` | 0.01 | Time step for discretization |
| `Q_max` | 10 | Maximum inventory (absolute) |
| `delta_min` | 0.001 | Minimum half-spread |
| `delta_max` | 0.1 | Maximum half-spread |

### Custom Parameters

```python
from src.models import HJBParameters, InventoryHJB

# Create custom parameters
params = HJBParameters(
    sigma=0.3,      # Higher volatility
    k=0.3,           # Less liquid market
    gamma=0.2,        # More risk-averse
    Q_max=20,          # Larger inventory limit
    T=2.0             # Longer horizon
)

# Create model with custom parameters
model = InventoryHJB(params)
```

### Simulator Parameters

The [`SimulatorParameters`](../src/data/simulator.py) class controls the simulation:

| Parameter | Default | Description |
|-----------|----------|-------------|
| `sigma` | 0.2 | Price volatility |
| `mu` | 0.0 | Price drift |
| `S0` | 100.0 | Initial midprice |
| `A` | 1.0 | Base intensity |
| `k` | 0.5 | Market depth |
| `T` | 1.0 | Simulation duration |
| `dt` | 0.001 | Simulation time step |
| `Q_max` | 10 | Inventory limit |
| `order_size` | 1 | Size of each order |
| `seed` | None | Random seed for reproducibility |

---

## Running Backtests

### Command-Line Interface

```bash
# Run all default strategies
python cli.py backtest

# Run specific strategies
python cli.py backtest --strategies hjb constant

# Custom seed and output directory
python cli.py backtest --seed 123 --output-dir my_results

# Multiple runs
python cli.py backtest --n-runs 10
```

### Available Strategies

| Strategy | Description |
|-----------|-------------|
| `hjb` | HJB optimal strategy (Guéant-Lehalle-Fernandez-Tapia) |
| `constant` | Constant spread baseline |
| `inventory_aware` | Inventory-aware heuristic strategy |

### Python API

```python
from experiments.run_backtest import run_backtest, plot_results
from src.models import create_default_model
from src.data import create_default_simulator

# Create components
simulator = create_default_simulator(seed=42)
hjb_model = create_default_model()

# Define policy
def hjb_policy(t, q, S):
    delta_b, delta_a = hjb_model.get_optimal_spreads(t, q)
    return S - delta_b, S + delta_a

# Run backtest
backtest = run_backtest(simulator, hjb_policy, "HJB Optimal")

# Plot results
plot_results([backtest], output_dir="results")
```

### Output Files

Running backtests generates:

- `results/backtest_comparison.png` - Strategy comparison plots
- `results/statistics_table.png` - Performance statistics table

---

## Training RL Agents

### Prerequisites

Install RL dependencies:

```bash
pip install stable-baselines3 torch
```

### Basic Training

```python
from src.rl_env import create_default_env
from stable_baselines3 import PPO

# Create environment
env = create_default_env(reward_type="pnl", seed=42)

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)

# Train
model.learn(total_timesteps=100000)

# Save model
model.save("models/ppo_market_making")
```

### Evaluation

```python
# Load trained model
model = PPO.load("models/ppo_market_making")

# Create test environment
test_env = create_default_env(reward_type="pnl", seed=999)

# Evaluate
obs, info = test_env.reset()
total_reward = 0

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        break

print(f"Total reward: {total_reward:.4f}")
```

### Reward Types

The environment supports different reward functions:

| Reward Type | Description |
|-------------|-------------|
| `pnl` | Change in PnL (default) |
| `sharpe` | Sharpe ratio of recent returns |
| `inventory_penalty` | PnL minus inventory penalty |

```python
env = create_default_env(reward_type="inventory_penalty", seed=42)
```

---

## Custom Strategies

### Policy Function Signature

A policy function must have the following signature:

```python
def policy(t: float, q: int, S: float) -> Tuple[float, float]:
    """
    Compute bid and ask quotes.
    
    Args:
        t: Current time
        q: Current inventory
        S: Current midprice
        
    Returns:
        Tuple of (bid_price, ask_price)
    """
    # Your logic here
    return bid_price, ask_price
```

### Example: Mean-Reversion Strategy

```python
def mean_reversion_policy(t, q, S, S0=100.0, base_spread=0.02):
    """
    Strategy that widens spreads when price deviates from mean.
    """
    # Compute deviation from initial price
    deviation = abs(S - S0) / S0
    
    # Widen spread based on deviation
    spread = base_spread * (1 + deviation)
    
    return S - spread/2, S + spread/2
```

### Example: Volatility-Adjusted Strategy

```python
def volatility_adjusted_policy(t, q, S, base_spread=0.02, vol_window=100):
    """
    Strategy that adjusts spreads based on recent volatility.
    """
    # This would require tracking price history
    # For simplicity, use a fixed adjustment
    spread = base_spread * 1.5  # Assume high volatility
    
    return S - spread/2, S + spread/2
```

### Using Custom Policies

```python
from src.data import create_default_simulator

simulator = create_default_simulator(seed=42)

# Use custom policy
results = simulator.run_simulation(mean_reversion_policy)

print(f"Final PnL: {results['final_pnl']:.4f}")
```

---

## Reproducing Results

### Reproducing Paper Results

To reproduce results from the Guéant et al. (2013) paper:

```python
from src.models import HJBParameters, InventoryHJB
from src.data import SimulatorParameters, MarketMakingSimulator

# Paper parameters
model_params = HJBParameters(
    sigma=0.2,
    A=1.0,
    k=0.5,
    gamma=0.1,
    phi=0.01,
    T=1.0,
    dt=0.01,
    Q_max=10
)

sim_params = SimulatorParameters(
    sigma=0.2,
    S0=100.0,
    A=1.0,
    k=0.5,
    T=1.0,
    dt=0.001,
    Q_max=10,
    seed=42  # For reproducibility
)

# Create model and simulator
model = InventoryHJB(model_params)
simulator = MarketMakingSimulator(sim_params)

# Run simulation
results = simulator.run_simulation(
    lambda t, q, S: model.get_quotes(t, q, S)
)

print(f"Reproduced PnL: {results['final_pnl']:.4f}")
```

### Monte Carlo Simulation

For statistical significance, run multiple simulations:

```python
import numpy as np

n_simulations = 1000
pnls = []

for i in range(n_simulations):
    simulator = create_default_simulator(seed=i)
    results = simulator.run_simulation(policy)
    pnls.append(results['final_pnl'])

# Compute statistics
mean_pnl = np.mean(pnls)
std_pnl = np.std(pnls)
percentile_5 = np.percentile(pnls, 5)
percentile_95 = np.percentile(pnls, 95)

print(f"Mean PnL: {mean_pnl:.4f}")
print(f"Std PnL: {std_pnl:.4f}")
print(f"5th percentile: {percentile_5:.4f}")
print(f"95th percentile: {percentile_95:.4f}")
```

---

## Troubleshooting

### Common Issues

#### Issue: Import Error

```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Make sure you're running from the project root directory:

```bash
cd market-making-inventory
python experiments/run_backtest.py
```

#### Issue: OR-Tools Not Found

```
ModuleNotFoundError: No module named 'ortools'
```

**Solution:** Install OR-Tools:

```bash
pip install ortools
```

#### Issue: Gymnasium Not Found

```
ModuleNotFoundError: No module named 'gymnasium'
```

**Solution:** Install Gymnasium:

```bash
pip install gymnasium
```

#### Issue: Slow Performance

**Solution:** Reduce discretization resolution:

```python
params = HJBParameters(
    dt=0.05,  # Larger time step
    Q_max=5     # Smaller inventory range
)
```

### Debug Mode

Enable verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

### Getting Help

If you encounter issues:

1. Check the [documentation](../docs/)
2. Search existing [GitHub issues](https://github.com/yourusername/market-making-inventory/issues)
3. Open a new issue with:
   - Python version
   - Error message
   - Minimal reproducible example

---

## Advanced Usage

### Multi-Asset Market Making

Extend to multiple assets:

```python
class MultiAssetMarketMaker:
    def __init__(self, n_assets=2):
        self.models = [create_default_model() for _ in range(n_assets)]
        self.simulators = [create_default_simulator() for _ in range(n_assets)]
    
    def policy(self, t, q_vec, S_vec):
        quotes = []
        for i, (q, S) in enumerate(zip(q_vec, S_vec)):
            delta_b, delta_a = self.models[i].get_optimal_spreads(t, q)
            quotes.append((S - delta_b, S + delta_a))
        return quotes
```

### Stochastic Volatility

Implement Heston model:

```python
def heston_price_step(S, sigma, kappa, theta, xi, dt):
    """
    Generate price step with stochastic volatility.
    """
    # Volatility dynamics
    d_sigma = kappa * (theta - sigma) * dt + xi * np.sqrt(sigma) * np.sqrt(dt) * np.random.randn()
    sigma_new = max(sigma + d_sigma, 0.01)
    
    # Price dynamics
    dS = sigma_new * S * np.sqrt(dt) * np.random.randn()
    
    return S + dS, sigma_new
```

### Custom Risk Measures

Implement CVaR constraint:

```python
def cvar_constraint(pnl_history, alpha=0.95, max_cvar=5.0):
    """
    Check if CVaR constraint is satisfied.
    """
    from scipy.stats import norm
    
    returns = np.diff(pnl_history)
    if len(returns) < 2:
        return True
    
    z_alpha = norm.ppf(alpha)
    cvar = np.mean(returns[returns <= np.percentile(returns, alpha * 100)])
    
    return abs(cvar) <= max_cvar
```

---

## Next Steps

- Explore the [notebooks/](../notebooks/) directory for interactive examples
- Read the [mathematical model documentation](math_model.md)
- Check the [state of the art review](state_of_the_art.md)
- Contribute your own strategies!
