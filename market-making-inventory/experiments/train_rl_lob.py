"""
Train RL Agent with Real LOB Data

This script provides a complete training pipeline for reinforcement learning
agents using real Limit Order Book (LOB) data from LOBSTER or Binance.

Usage:
    # Train with LOBSTER data
    python train_rl_lob.py --source lobster --message-file data/AAPL_message.csv \
        --orderbook-file data/AAPL_orderbook.csv --timesteps 100000
    
    # Train with Binance data
    python train_rl_lob.py --source binance --data-path data/binance/ \
        --symbol BTCUSDT --timesteps 100000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rl_env.lob_market_making_env import (
        LOBMarketMakingEnv,
        LOBMarketMakingEnvConfig,
        create_lob_env
    )
    from data.lob_loader import LOBDataLoader, LOBDataConfig
except ImportError:
    print("Error: Could not import required modules.")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

try:
    from stable_baselines3 import PPO, DQN, SAC, A2C
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import plot_results
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Get current reward
        if len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Reward={self.episode_rewards[-1]:.2f}, "
                      f"Length={self.episode_lengths[-1]}")
        
        return True


class PlottingCallback(BaseCallback):
    """Callback for plotting training progress."""
    
    def __init__(self, plot_freq: int = 1000, save_path: str = "plots"):
        super().__init__()
        self.plot_freq = plot_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.episode_rewards = []
        self.episode_pnls = []
    
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_pnls.append(info.get('pnl', 0))
            
            # Plot every plot_freq episodes
            if len(self.episode_rewards) % self.plot_freq == 0:
                self._plot_progress()
        
        return True
    
    def _plot_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        axes[0].plot(self.episode_rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].grid(True)
        
        # Plot PnL
        axes[1].plot(self.episode_pnls)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Final PnL')
        axes[1].set_title('Final PnL per Episode')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_path / f'training_progress_{len(self.episode_rewards)}.png')
        plt.close()


def create_env(
    config: LOBMarketMakingEnvConfig,
    monitor_dir: Optional[str] = None
):
    """
    Create a monitored environment.
    
    Args:
        config: Environment configuration
        monitor_dir: Directory for monitoring logs
        
    Returns:
        Monitored environment
    """
    env = LOBMarketMakingEnv(config)
    
    if monitor_dir is not None:
        monitor_path = Path(monitor_dir)
        monitor_path.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_path))
    
    return env


def train_agent(
    env_config: LOBMarketMakingEnvConfig,
    algorithm: str = "ppo",
    total_timesteps: int = 100000,
    model_path: str = "models/rl_lob_model",
    log_dir: str = "logs",
    checkpoint_freq: int = 10000,
    verbose: int = 1
):
    """
    Train an RL agent on LOB data.
    
    Args:
        env_config: Environment configuration
        algorithm: RL algorithm to use (ppo, dqn, sac, a2c)
        total_timesteps: Total training timesteps
        model_path: Path to save the trained model
        log_dir: Directory for training logs
        checkpoint_freq: Frequency of checkpoint saving
        verbose: Verbosity level
        
    Returns:
        Trained model
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        )
    
    # Create environment
    env = create_env(env_config, monitor_dir=log_dir)
    
    # Select algorithm
    algorithm_map = {
        'ppo': PPO,
        'dqn': DQN,
        'sac': SAC,
        'a2c': A2C
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                         f"Available: {list(algorithm_map.keys())}")
    
    algo_class = algorithm_map[algorithm]
    
    # Create model
    if algorithm in ['ppo', 'a2c']:
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir
        )
    elif algorithm == 'dqn':
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            target_update_interval=1000,
            tensorboard_log=log_dir
        )
    elif algorithm == 'sac':
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=1000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            tensorboard_log=log_dir
        )
    
    # Create callbacks
    callbacks = []
    
    # Training callback
    training_callback = TrainingCallback(verbose=verbose)
    callbacks.append(training_callback)
    
    # Checkpoint callback
    checkpoint_path = Path(model_path).parent
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix=f"rl_lob_{algorithm}"
    )
    callbacks.append(checkpoint_callback)
    
    # Plotting callback
    plotting_callback = PlottingCallback(
        plot_freq=100,
        save_path=str(checkpoint_path / "plots")
    )
    callbacks.append(plotting_callback)
    
    # Train model
    print(f"\nTraining {algorithm.upper()} agent for {total_timesteps} timesteps...")
    print(f"Environment: {env_config.data_source} data")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=verbose > 0
    )
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Save training statistics
    stats = {
        'episode_rewards': training_callback.episode_rewards,
        'episode_lengths': training_callback.episode_lengths,
        'total_timesteps': total_timesteps,
        'algorithm': algorithm,
        'env_config': env_config.__dict__
    }
    
    stats_path = model_path.parent / f"{model_path.stem}_stats.npz"
    np.savez(stats_path, **stats)
    print(f"Training statistics saved to {stats_path}")
    
    return model


def evaluate_agent(
    model,
    env_config: LOBMarketMakingEnvConfig,
    n_episodes: int = 10,
    render: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a trained agent.
    
    Args:
        model: Trained RL model
        env_config: Environment configuration
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        Dictionary with evaluation results
    """
    env = LOBMarketMakingEnv(env_config)
    
    episode_rewards = []
    episode_pnls = []
    episode_trades = []
    episode_inventories = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        episode_pnls.append(info['pnl'])
        episode_trades.append(info['n_trades'])
        episode_inventories.append(info['inventory'])
        
        print(f"Episode {episode + 1}: "
              f"Reward={total_reward:.2f}, "
              f"PnL={info['pnl']:.2f}, "
              f"Trades={info['n_trades']}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_pnl': np.mean(episode_pnls),
        'std_pnl': np.std(episode_pnls),
        'mean_trades': np.mean(episode_trades),
        'mean_inventory': np.mean(episode_inventories),
        'episode_rewards': episode_rewards,
        'episode_pnls': episode_pnls
    }
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean PnL: {results['mean_pnl']:.2f} ± {results['std_pnl']:.2f}")
    print(f"  Mean Trades: {results['mean_trades']:.2f}")
    print(f"  Mean Final Inventory: {results['mean_inventory']:.2f}")
    
    return results


def plot_results(results: Dict[str, Any], save_path: str = "plots/evaluation.png"):
    """
    Plot evaluation results.
    
    Args:
        results: Evaluation results dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    axes[0].plot(results['episode_rewards'], marker='o')
    axes[0].axhline(results['mean_reward'], color='r', linestyle='--', 
                    label=f"Mean: {results['mean_reward']:.2f}")
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot PnL
    axes[1].plot(results['episode_pnls'], marker='o', color='green')
    axes[1].axhline(results['mean_pnl'], color='r', linestyle='--',
                    label=f"Mean: {results['mean_pnl']:.2f}")
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Final PnL')
    axes[1].set_title('Episode PnL')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train RL Agent with Real LOB Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with LOBSTER data
  python train_rl_lob.py --source lobster \\
      --message-file data/AAPL_message.csv \\
      --orderbook-file data/AAPL_orderbook.csv \\
      --timesteps 100000
  
  # Train with Binance data
  python train_rl_lob.py --source binance \\
      --data-path data/binance/ \\
      --symbol BTCUSDT \\
      --timesteps 100000
  
  # Evaluate a trained model
  python train_rl_lob.py --mode evaluate \\
      --model models/rl_lob_model.zip \\
      --source lobster \\
      --message-file data/AAPL_message.csv \\
      --orderbook-file data/AAPL_orderbook.csv
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'evaluate'],
        help='Mode: train or evaluate'
    )
    
    # Data source
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['lobster', 'binance'],
        help='Data source'
    )
    
    # LOBSTER specific
    parser.add_argument(
        '--message-file',
        type=str,
        help='Path to LOBSTER message file'
    )
    parser.add_argument(
        '--orderbook-file',
        type=str,
        help='Path to LOBSTER orderbook file'
    )
    
    # Binance specific
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to Binance data directory'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (for Binance)'
    )
    
    # Training parameters
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ppo',
        choices=['ppo', 'dqn', 'sac', 'a2c'],
        help='RL algorithm'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--episode-length',
        type=int,
        default=1000,
        help='Episode length in timesteps'
    )
    
    # Reward parameters
    parser.add_argument(
        '--reward-type',
        type=str,
        default='pnl',
        choices=['pnl', 'sharpe', 'inventory_penalty', 'spread_profit'],
        help='Reward function type'
    )
    
    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default='models/rl_lob_model.zip',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation'
    )
    
    # Other parameters
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level'
    )
    
    args = parser.parse_args()
    
    # Create environment configuration
    env_config = LOBMarketMakingEnvConfig(
        data_source=args.source,
        data_path=args.data_path,
        message_file=args.message_file,
        orderbook_file=args.orderbook_file,
        symbol=args.symbol,
        episode_length=args.episode_length,
        reward_type=args.reward_type,
        seed=args.seed
    )
    
    # Validate data parameters
    if args.source == 'lobster':
        if args.message_file is None or args.orderbook_file is None:
            parser.error("--message-file and --orderbook-file are required for LOBSTER data")
    elif args.source == 'binance':
        if args.data_path is None:
            parser.error("--data-path is required for Binance data")
    
    if args.mode == 'train':
        # Train agent
        model = train_agent(
            env_config=env_config,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            model_path=args.model,
            log_dir=args.log_dir,
            verbose=args.verbose
        )
        
        # Evaluate trained model
        print("\n" + "="*50)
        print("Evaluating trained model...")
        print("="*50)
        results = evaluate_agent(
            model=model,
            env_config=env_config,
            n_episodes=args.n_episodes,
            render=args.render
        )
        
        # Plot results
        plot_path = Path(args.model).parent / "evaluation.png"
        plot_results(results, save_path=str(plot_path))
        
    elif args.mode == 'evaluate':
        # Load and evaluate model
        if not SB3_AVAILABLE:
            parser.error("stable-baselines3 is required for evaluation")
        
        # Load model
        algorithm_map = {'ppo': PPO, 'dqn': DQN, 'sac': SAC, 'a2c': A2C}
        
        # Try to detect algorithm from model path
        model_path = Path(args.model)
        for algo_name, algo_class in algorithm_map.items():
            if algo_name in model_path.stem.lower():
                model = algo_class.load(args.model)
                break
        else:
            # Default to PPO
            model = PPO.load(args.model)
        
        print(f"Loaded model from {args.model}")
        
        # Evaluate
        results = evaluate_agent(
            model=model,
            env_config=env_config,
            n_episodes=args.n_episodes,
            render=args.render
        )
        
        # Plot results
        plot_path = Path(args.model).parent / "evaluation.png"
        plot_results(results, save_path=str(plot_path))


if __name__ == "__main__":
    main()
