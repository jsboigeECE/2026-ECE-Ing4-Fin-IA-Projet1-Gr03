#!/usr/bin/env python3
"""
Market Making CLI

Command-line interface for the market making with inventory constraints project.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def cmd_backtest(args):
    """Run backtests."""
    from experiments.run_backtest import main as backtest_main
    
    # Override sys.argv for backtest
    sys.argv = ['run_backtest.py', '--strategies'] + args.strategies
    if args.seed:
        sys.argv.extend(['--seed', str(args.seed)])
    if args.n_runs:
        sys.argv.extend(['--n-runs', str(args.n_runs)])
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    
    backtest_main()


def cmd_train(args):
    """Train an RL agent."""
    print("Training RL agent...")
    print("This feature requires stable-baselines3 to be installed.")
    print("Install with: pip install stable-baselines3")
    
    # Placeholder for RL training
    # from rl_env import create_default_env
    # from stable_baselines3 import PPO
    # 
    # env = create_default_env(reward_type="pnl")
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100000)
    # model.save("models/ppo_market_making")


def cmd_simulate(args):
    """Run a simulation."""
    from data import create_default_simulator
    from models import create_default_model
    
    print(f"Running simulation with seed={args.seed}...")
    
    # Create simulator
    simulator = create_default_simulator(seed=args.seed)
    
    # Create HJB model for optimal policy
    hjb_model = create_default_model()
    
    # Create policy
    def policy(t: float, q: int, S: float) -> tuple:
        delta_b, delta_a = hjb_model.get_optimal_spreads(t, q)
        return S - delta_b, S + delta_a
    
    # Run simulation
    results = simulator.run_simulation(policy)
    
    print(f"\nSimulation Results:")
    print(f"  Final PnL: {results['final_pnl']:.4f}")
    print(f"  Final Inventory: {results['final_inventory']}")
    print(f"  Number of Trades: {results['n_trades']}")


def cmd_analyze(args):
    """Analyze results."""
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(f"Analyzing results from {args.input}...")
    
    # Placeholder for analysis
    # Load results and generate plots
    print("Analysis feature coming soon!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Market Making with Inventory Constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtests with default strategies
  python cli.py backtest
  
  # Run backtests with specific strategies
  python cli.py backtest --strategies hjb constant
  
  # Run simulation with custom seed
  python cli.py simulate --seed 123
  
  # Train RL agent
  python cli.py train --timesteps 100000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run backtests for market making strategies'
    )
    backtest_parser.add_argument(
        '--strategies',
        nargs='+',
        default=['hjb', 'constant', 'inventory_aware'],
        choices=['hjb', 'constant', 'inventory_aware'],
        help='Strategies to test'
    )
    backtest_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    backtest_parser.add_argument(
        '--n-runs',
        type=int,
        default=1,
        help='Number of simulation runs'
    )
    backtest_parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train a reinforcement learning agent'
    )
    train_parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of training timesteps'
    )
    train_parser.add_argument(
        '--algorithm',
        type=str,
        default='ppo',
        choices=['ppo', 'dqn', 'sac'],
        help='RL algorithm to use'
    )
    train_parser.add_argument(
        '--reward-type',
        type=str,
        default='pnl',
        choices=['pnl', 'sharpe', 'inventory_penalty'],
        help='Reward function type'
    )
    train_parser.set_defaults(func=cmd_train)
    
    # Simulate command
    simulate_parser = subparsers.add_parser(
        'simulate',
        help='Run a single simulation'
    )
    simulate_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    simulate_parser.add_argument(
        '--strategy',
        type=str,
        default='hjb',
        choices=['hjb', 'constant', 'inventory_aware'],
        help='Strategy to use'
    )
    simulate_parser.set_defaults(func=cmd_simulate)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze simulation results'
    )
    analyze_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input file or directory'
    )
    analyze_parser.add_argument(
        '--output',
        type=str,
        default='analysis',
        help='Output directory'
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
