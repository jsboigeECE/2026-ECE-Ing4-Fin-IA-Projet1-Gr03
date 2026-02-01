"""
Simple test script for market making project.
Tests basic functionality without scipy dependency.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models import create_default_model
        print("  ✓ models imported")
    except Exception as e:
        print(f"  ✗ models import failed: {e}")
        return False
    
    try:
        from data import create_default_simulator
        print("  ✓ data imported")
    except Exception as e:
        print(f"  ✗ data import failed: {e}")
        return False
    
    try:
        from solvers import create_default_solver
        print("  ✓ solvers imported")
    except Exception as e:
        print(f"  ✗ solvers import failed: {e}")
        return False
    
    try:
        from rl_env import create_default_env
        print("  ✓ rl_env imported")
    except Exception as e:
        print(f"  ✗ rl_env import failed: {e}")
        return False
    
    return True

def test_model():
    """Test HJB model creation."""
    print("\nTesting HJB model...")
    
    try:
        from models import create_default_model
        model = create_default_model()
        print(f"  ✓ Model created")
        print(f"    Time horizon: {model.params.T}")
        print(f"    Max inventory: {model.params.Q_max}")
        
        # Test optimal spreads
        delta_b, delta_a = model.get_optimal_spreads(0.5, 3)
        print(f"  ✓ Optimal spreads computed: delta_b={delta_b:.4f}, delta_a={delta_a:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simulator():
    """Test simulator creation."""
    print("\nTesting simulator...")
    
    try:
        from data import create_default_simulator
        simulator = create_default_simulator(seed=42)
        print(f"  ✓ Simulator created")
        print(f"    Time horizon: {simulator.params.T}")
        print(f"    Max inventory: {simulator.params.Q_max}")
        
        # Test simple policy
        def simple_policy(t, q, S):
            return S - 0.01, S + 0.01
        
        # Run a short simulation
        results = simulator.run_simulation(simple_policy)
        print(f"  ✓ Simulation completed")
        print(f"    Final PnL: {results['final_pnl']:.4f}")
        print(f"    Number of trades: {results['n_trades']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_solver():
    """Test HJB solver."""
    print("\nTesting HJB solver...")
    
    try:
        from solvers import create_default_solver
        solver, policy = create_default_solver("finite_difference")
        print(f"  ✓ Solver created")
        print(f"    Time steps: {solver.N}")
        
        # Test policy
        delta_b, delta_a = policy(0.5, 3)
        print(f"  ✓ Policy works: delta_b={delta_b:.4f}, delta_a={delta_a:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Solver test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_env():
    """Test RL environment."""
    print("\nTesting RL environment...")
    
    try:
        from rl_env import create_default_env
        env = create_default_env(reward_type="pnl", seed=42)
        print(f"  ✓ Environment created")
        print(f"    Action space: {env.action_space}")
        print(f"    Observation space: {env.observation_space}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"  ✓ Reset successful")
        print(f"    Observation: {obs}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step successful")
        print(f"    Reward: {reward:.4f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"  ✗ RL env test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Market Making Project - Simple Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test model
    if not test_model():
        all_passed = False
    
    # Test simulator
    if not test_simulator():
        all_passed = False
    
    # Test solver
    if not test_solver():
        all_passed = False
    
    # Test RL env
    if not test_rl_env():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
