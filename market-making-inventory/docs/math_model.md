# Mathematical Model: Optimal Market Making with Inventory Constraints

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Price Dynamics](#price-dynamics)
3. [Order Flow Model](#order-flow-model)
4. [HJB Equation](#hjb-equation)
5. [Optimal Quotes](#optimal-quotes)
6. [Inventory Constraints](#inventory-constraints)
7. [Risk Measures](#risk-measures)
8. [CSP Formulation](#csp-formulation)

---

## Problem Formulation

### Market Maker's Objective

A market maker continuously posts bid and ask quotes to provide liquidity. The objective is to maximize the expected utility of the terminal P&L over a finite time horizon $[0, T]$.

**Value Function:**

$$u(t, x, q, S) = \sup_{\delta^b, \delta^a} \mathbb{E}_{t,x,q,S}\left[ X_T + q_T S_T - \phi(q_T^2) \right]$$

Where:
- $t$: current time
- $x$: cash position
- $q$: inventory position (number of shares held)
- $S$: midprice of the asset
- $X_T$: terminal cash position
- $q_T$: terminal inventory
- $\phi$: terminal liquidation cost parameter
- $\delta^b, \delta^a$: bid and ask half-spreads

### State Variables

The state of the system at time $t$ is characterized by:

$$\mathcal{S}_t = (t, q_t, S_t)$$

### Control Variables

The market maker controls the bid and ask half-spreads:

$$\mathcal{A}_t = (\delta^b_t, \delta^a_t)$$

---

## Price Dynamics

### Brownian Motion Model

The midprice follows a geometric Brownian motion:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

For simplicity in the analytical solution, we often use the arithmetic Brownian motion:

$$dS_t = \sigma dW_t$$

Where:
- $\mu$: drift (often assumed zero for market making)
- $\sigma$: volatility
- $W_t$: standard Brownian motion

### Discretization

For numerical implementation, we discretize the price process:

$$S_{t+\Delta t} = S_t + \sigma \sqrt{\Delta t} \cdot Z_t$$

Where $Z_t \sim \mathcal{N}(0, 1)$.

---

## Order Flow Model

### Poisson Arrival Process

Market orders arrive according to Poisson processes with intensity functions that depend on the quoted spreads.

**Buy Market Orders (fills the ask):**

$$\lambda^a(\delta^a) = A \exp(-k \delta^a)$$

**Sell Market Orders (fills the bid):**

$$\lambda^b(\delta^b) = A \exp(-k \delta^b)$$

Where:
- $A$: base intensity parameter
- $k$: market depth parameter (sensitivity to spread)
- $\delta^a, \delta^b$: ask and bid half-spreads

### Inventory Dynamics

When a market order arrives, the inventory changes:

$$dq_t = \begin{cases}
+1 & \text{with probability } \lambda^b(\delta^b) dt \quad \text{(buy order fills bid)} \\
-1 & \text{with probability } \lambda^a(\delta^a) dt \quad \text{(sell order fills ask)} \\
0 & \text{otherwise}
\end{cases}$$

### Cash Dynamics

The cash position changes when trades occur:

$$dX_t = \begin{cases}
(S_t - \delta^b_t) & \text{when } dq_t = +1 \\
(S_t + \delta^a_t) & \text{when } dq_t = -1 \\
0 & \text{otherwise}
\end{cases}$$

---

## HJB Equation

### Hamilton-Jacobi-Bellman Equation

The value function satisfies the HJB equation:

$$\frac{\partial u}{\partial t} + \frac{\sigma^2}{2} \frac{\partial^2 u}{\partial S^2} + \sup_{\delta^b, \delta^a} \mathcal{L}(\delta^b, \delta^a) = 0$$

Where the generator $\mathcal{L}$ is:

$$\mathcal{L}(\delta^b, \delta^a) = \lambda^b(\delta^b) [u(t, x + S - \delta^b, q+1, S) - u(t, x, q, S)] + \lambda^a(\delta^a) [u(t, x + S + \delta^a, q-1, S) - u(t, x, q, S)]$$

### Simplified Form

Since the value function is linear in $x$ and $S$, we can write:

$$u(t, x, q, S) = x + qS + v(t, q)$$

Where $v(t, q)$ is the "indifference value" representing the cost of holding inventory.

The HJB equation simplifies to:

$$\frac{\partial v}{\partial t} + \sup_{\delta^b, \delta^a} \left\{ \lambda^b(\delta^b)[v(t, q+1) - v(t, q) - \delta^b] + \lambda^a(\delta^a)[v(t, q-1) - v(t, q) + \delta^a] \right\} = 0$$

### First-Order Conditions

Taking derivatives with respect to $\delta^b$ and $\delta^a$:

$$\frac{\partial}{\partial \delta^b}: -k \lambda^b(\delta^b)[v(t, q+1) - v(t, q) - \delta^b] - \lambda^b(\delta^b) = 0$$

$$\frac{\partial}{\partial \delta^a}: -k \lambda^a(\delta^a)[v(t, q-1) - v(t, q) + \delta^a] + \lambda^a(\delta^a) = 0$$

---

## Optimal Quotes

### Closed-Form Solution

Solving the first-order conditions gives the optimal spreads:

$$\delta^{b*}(t, q) = \frac{1}{2k} + v(t, q+1) - v(t, q)$$

$$\delta^{a*}(t, q) = \frac{1}{2k} + v(t, q) - v(t, q-1)$$

### System of ODEs

The indifference values satisfy a system of linear ODEs:

$$\frac{dv}{dt}(t, q) = -\frac{A}{2k} \left[ \exp(-k[v(t, q+1) - v(t, q)]) + \exp(-k[v(t, q) - v(t, q-1)]) \right]$$

With terminal condition:

$$v(T, q) = -\phi q^2$$

### Asymptotic Approximation (Large T)

For large time horizons, we obtain the famous Guéant-Lehalle-Fernandez-Tapia formulas:

$$\delta^{b*}(q) \approx \frac{1}{2k} + \gamma \sigma^2 (T-t) \left(q + \frac{1}{2}\right)$$

$$\delta^{a*}(q) \approx \frac{1}{2k} + \gamma \sigma^2 (T-t) \left(q - \frac{1}{2}\right)$$

Where $\gamma$ is the risk aversion parameter related to $\phi$.

### Interpretation

1. **Base Spread:** $\frac{1}{2k}$ is the minimum spread to compensate for providing liquidity
2. **Inventory Adjustment:** The term $\gamma \sigma^2 (T-t)q$ adjusts the spread based on inventory
   - Positive inventory: widen ask, narrow bid (to sell)
   - Negative inventory: widen bid, narrow ask (to buy)
3. **Time Decay:** The adjustment decreases as $t \to T$

---

## Inventory Constraints

### Hard Constraints

The inventory is bounded:

$$q \in \{-Q_{max}, -Q_{max}+1, \ldots, 0, \ldots, Q_{max}-1, Q_{max}\}$$

### Boundary Conditions

At the boundaries, the market maker cannot take more inventory:

$$\delta^{b*}(t, Q_{max}) = +\infty \quad \text{(cannot buy more)}$$
$$\delta^{a*}(t, -Q_{max}) = +\infty \quad \text{(cannot sell more)}$$

In practice, we set very large spreads at boundaries.

### Terminal Liquidation

At time $T$, the market maker must liquidate the inventory:

$$v(T, q) = -\phi q^2$$

This represents the cost of liquidating $q$ shares at the midprice.

---

## Risk Measures

### Variance-Based Risk Aversion

The parameter $\gamma$ represents risk aversion:

$$\gamma = \frac{\phi}{\sigma^2}$$

Higher $\gamma$ means more conservative quoting.

### Value at Risk (VaR)

For a confidence level $\alpha$, the VaR is:

$$\text{VaR}_\alpha = \inf\{x \in \mathbb{R} : P(\text{Loss} \leq x) \geq \alpha\}$$

For a Brownian price model:

$$\text{VaR}_\alpha = q \sigma \sqrt{T-t} \cdot \Phi^{-1}(\alpha)$$

Where $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

### Conditional VaR (CVaR)

$$\text{CVaR}_\alpha = \mathbb{E}[\text{Loss} \mid \text{Loss} \geq \text{VaR}_\alpha]$$

For the Brownian model:

$$\text{CVaR}_\alpha = q \sigma \sqrt{T-t} \cdot \frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}$$

Where $\phi$ is the standard normal PDF.

### Maximum Drawdown Constraint

To limit maximum drawdown, we can impose:

$$\max_{t \in [0,T]} (X_t + q_t S_t - X_0) \geq -D_{max}$$

This is typically enforced through simulation and policy adjustment.

---

## CSP Formulation

### Constraint Satisfaction Problem

An alternative formulation treats the problem as a CSP:

**Variables:**
- $\delta^b_t, \delta^a_t$ for each time step $t$

**Constraints:**
1. Inventory bounds: $-Q_{max} \leq q_t \leq Q_{max}$
2. VaR constraint: $P(\text{Loss} > \text{VaR}_{max}) \leq \alpha$
3. Drawdown constraint: $\text{Drawdown}_t \leq D_{max}$
4. Spread bounds: $\delta_{min} \leq \delta^b_t, \delta^a_t \leq \delta_{max}$

**Objective:**
Minimize expected cost or maximize expected utility.

### OR-Tools Formulation

```python
from ortools.sat.python import cp_model

# Create model
model = cp_model.CpModel()

# Variables
delta_b = [model.NewIntVar(0, 100, f'delta_b_{t}') for t in range(T)]
delta_a = [model.NewIntVar(0, 100, f'delta_a_{t}') for t in range(T)]
inventory = [model.NewIntVar(-Q_max, Q_max, f'q_{t}') for t in range(T+1)]

# Initial inventory
model.Add(inventory[0] == 0)

# Inventory dynamics
for t in range(T):
    # This is a simplified version - actual implementation requires
    # modeling the stochastic nature of order arrivals
    model.Add(inventory[t+1] == inventory[t] + buy[t] - sell[t])

# VaR constraint (simplified)
model.Add(expected_loss <= VaR_max)

# Objective
model.Minimize(total_cost)
```

---

## Numerical Implementation

### Finite Difference Method

For solving the HJB equation numerically:

1. **Discretize time:** $t_n = n \Delta t$, $n = 0, \ldots, N$
2. **Discretize inventory:** $q \in \{-Q_{max}, \ldots, Q_{max}\}$
3. **Backward induction:** Start from terminal condition and solve backward

```python
# Pseudocode
v = np.zeros((N+1, 2*Q_max+1))
v[N, :] = -phi * q**2  # Terminal condition

for n in range(N-1, -1, -1):
    for q in range(-Q_max, Q_max+1):
        # Solve for optimal spreads
        delta_b = 1/(2*k) + v[n+1, q+1] - v[n+1, q]
        delta_a = 1/(2*k) + v[n+1, q] - v[n+1, q-1]
        
        # Update value function
        lambda_b = A * np.exp(-k * delta_b)
        lambda_a = A * np.exp(-k * delta_a)
        
        v[n, q] = v[n+1, q] - dt * (
            lambda_b * (v[n+1, q+1] - v[n+1, q] - delta_b) +
            lambda_a * (v[n+1, q-1] - v[n+1, q] + delta_a)
        )
```

### Policy Extraction

Once the value function is computed, the optimal policy is:

$$\pi^*(t, q) = (\delta^{b*}(t, q), \delta^{a*}(t, q))$$

---

## Extensions

### Stochastic Volatility

Replace constant $\sigma$ with $\sigma_t$ following:

$$d\sigma_t^2 = \kappa(\theta - \sigma_t^2)dt + \xi \sigma_t dZ_t$$

### Jump Processes

Add jumps to the price process:

$$dS_t = \sigma S_t dW_t + J_t dN_t$$

Where $N_t$ is a Poisson process and $J_t$ is the jump size.

### Multi-Asset

Extend to multiple correlated assets:

$$dS_t^i = \sigma^i dW_t^i$$

With correlation matrix $\Sigma$ where $\Sigma_{ij} = \rho_{ij}$.

---

## Summary

The mathematical model for optimal market making with inventory constraints consists of:

1. **Price dynamics:** Brownian motion (or extensions)
2. **Order flow:** Poisson processes with exponential intensity
3. **HJB equation:** Stochastic optimal control formulation
4. **Optimal quotes:** Closed-form or numerical solution
5. **Inventory constraints:** Bounds and terminal liquidation
6. **Risk measures:** VaR, CVaR, drawdown limits
7. **CSP formulation:** Alternative constraint-based approach

This model provides a rigorous foundation for implementing market making strategies that balance profit maximization with risk management.
