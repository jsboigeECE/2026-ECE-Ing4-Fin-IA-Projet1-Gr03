"""
Market Making with Inventory Constraints

This package provides tools for optimal market making with inventory constraints,
including mathematical models, numerical solvers, data simulators, and
RL environments.
"""

__version__ = "1.0.0"
__author__ = "Market Making Team"

from .models import (
    InventoryHJB,
    HJBParameters,
    create_default_model
)

from .data import (
    MarketMakingSimulator,
    LimitOrderBook,
    SimulatorParameters,
    Trade,
    OrderSide,
    create_default_simulator
)

from .solvers import (
    FiniteDifferenceSolver,
    PolicyIterationSolver,
    HJBSolverConfig,
    create_default_solver,
    CSPMarketMakingSolver,
    SimplifiedCSPSolver,
    CSPSolverConfig,
    create_default_csp_solver
)

from .rl_env import (
    MarketMakingEnv,
    MarketMakingEnvConfig,
    create_default_env
)

__all__ = [
    # Models
    'InventoryHJB',
    'HJBParameters',
    'create_default_model',
    # Data
    'MarketMakingSimulator',
    'LimitOrderBook',
    'SimulatorParameters',
    'Trade',
    'OrderSide',
    'create_default_simulator',
    # Solvers
    'FiniteDifferenceSolver',
    'PolicyIterationSolver',
    'HJBSolverConfig',
    'create_default_solver',
    'CSPMarketMakingSolver',
    'SimplifiedCSPSolver',
    'CSPSolverConfig',
    'create_default_csp_solver',
    # RL Environment
    'MarketMakingEnv',
    'MarketMakingEnvConfig',
    'create_default_env'
]
