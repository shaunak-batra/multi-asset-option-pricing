# Multi-Asset Option Pricing System

A comprehensive C++ implementation for pricing multi-asset exotic options using Monte Carlo simulation with correlated asset dynamics.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Framework](#mathematical-framework)
   - [Geometric Brownian Motion](#geometric-brownian-motion)
   - [Correlated Asset Dynamics](#correlated-asset-dynamics)
   - [Cholesky Decomposition](#cholesky-decomposition)
   - [Risk-Neutral Valuation](#risk-neutral-valuation)
3. [Multi-Asset Option Types](#multi-asset-option-types)
   - [Basket Options](#basket-options)
   - [Rainbow Options](#rainbow-options)
   - [Exchange Options](#exchange-options)
   - [Spread Options](#spread-options)
4. [Monte Carlo Simulation](#monte-carlo-simulation)
5. [Statistical Analysis](#statistical-analysis)
6. [Option Greeks](#option-greeks)
7. [Implementation Details](#implementation-details)
8. [Usage Guide](#usage-guide)
9. [Compilation and Execution](#compilation-and-execution)

---

## Overview

This system implements a sophisticated multi-asset option pricing engine that supports various exotic options. The core methodology employs **Monte Carlo simulation** with **correlated geometric Brownian motion** to model asset price evolution under the risk-neutral measure.

### Key Features

- Multiple exotic option types (Basket, Rainbow, Exchange, Spread)
- Correlated multi-asset price simulation
- Monte Carlo pricing with confidence intervals
- Interactive command-line interface
- Correlation sensitivity analysis
- Greeks calculation framework

---

## Mathematical Framework

### Geometric Brownian Motion

The fundamental model for asset price evolution is **Geometric Brownian Motion (GBM)**, which describes the stochastic behavior of asset prices in continuous time.

#### Single Asset Dynamics

For a single asset *S*, the price evolution under the physical measure follows:

```
dS(t) = μ S(t) dt + σ S(t) dW(t)
```

Where:
- **S(t)**: Asset price at time *t*
- **μ**: Drift rate (expected return)
- **σ**: Volatility (standard deviation of returns)
- **dW(t)**: Increment of a standard Wiener process (Brownian motion)

#### Closed-Form Solution

Using Itô's lemma, the solution to the GBM stochastic differential equation is:

```
S(T) = S(0) · exp[(μ - σ²/2)T + σ√T · Z]
```

Where:
- **S(0)**: Initial spot price
- **T**: Time to maturity
- **Z**: Standard normal random variable ~ N(0,1)
- **(μ - σ²/2)**: Drift adjustment (Itô correction)

**Implementation** (multi_asset_option.cpp:176-178):
```cpp
double drift = (mu[i] - 0.5 * sigma[i] * sigma[i]) * T;
double diffusion = sigma[i] * std::sqrt(T) * Z[i];
final_prices[i] = S0[i] * std::exp(drift + diffusion);
```

The term **-σ²/2** is the **Itô correction**, which accounts for the convexity adjustment when converting from the SDE to the exact solution.

---

### Correlated Asset Dynamics

For *n* assets, we model their joint evolution with a correlation structure.

#### Multi-Dimensional GBM

For *i*-th asset:

```
dSᵢ(t) = μᵢ Sᵢ(t) dt + σᵢ Sᵢ(t) dWᵢ(t)
```

Where the Brownian motions satisfy:

```
dWᵢ(t) · dWⱼ(t) = ρᵢⱼ dt
```

Here **ρᵢⱼ** is the correlation coefficient between assets *i* and *j*.

#### Correlation Matrix

The correlation matrix **Ρ** is symmetric and positive semi-definite:

```
     ⎡ 1    ρ₁₂  ρ₁₃  ... ⎤
Ρ =  ⎢ ρ₂₁   1   ρ₂₃  ... ⎥
     ⎢ ρ₃₁  ρ₃₂   1   ... ⎥
     ⎣ ...  ...  ...   1  ⎦
```

Properties:
- Diagonal elements are 1 (each asset is perfectly correlated with itself)
- **ρᵢⱼ = ρⱼᵢ** (symmetry)
- **-1 ≤ ρᵢⱼ ≤ 1** (correlation bounds)
- All eigenvalues ≥ 0 (positive semi-definite)

---

### Cholesky Decomposition

To generate correlated random variables, we use **Cholesky decomposition** of the correlation matrix.

#### Mathematical Theory

Given a positive semi-definite matrix **Ρ**, Cholesky decomposition finds a lower triangular matrix **L** such that:

```
Ρ = L · Lᵀ
```

#### Two-Asset Case (Simplified)

For two assets with correlation **ρ**:

```
Ρ = ⎡ 1   ρ ⎤
    ⎣ ρ   1 ⎦
```

The Cholesky decomposition is:

```
L = ⎡ 1       0      ⎤
    ⎣ ρ   √(1-ρ²)   ⎦
```

#### Generating Correlated Random Variables

1. Generate independent standard normal variables: **Z₁, Z₂ ~ N(0,1)**
2. Transform using Cholesky matrix:

```
Z₁' = Z₁
Z₂' = ρ · Z₁ + √(1 - ρ²) · Z₂
```

Now **Z₁'** and **Z₂'** are standard normal with correlation **ρ**.

**Implementation** (multi_asset_option.cpp:157-161):
```cpp
if (n == 2) {
    double rho = correlation[0][1];
    correlated_randoms[0] = independent_randoms[0];
    correlated_randoms[1] = rho * independent_randoms[0] +
                           std::sqrt(1 - rho * rho) * independent_randoms[1];
}
```

#### Verification

The correlation can be verified:

```
Corr(Z₁', Z₂') = E[Z₁' · Z₂']
                = E[Z₁ · (ρZ₁ + √(1-ρ²)Z₂)]
                = ρ E[Z₁²] + √(1-ρ²) E[Z₁Z₂]
                = ρ · 1 + √(1-ρ²) · 0
                = ρ
```

---

### Risk-Neutral Valuation

Under the **risk-neutral measure** (also called the **martingale measure**), option prices are computed as discounted expected payoffs.

#### Risk-Neutral Dynamics

Under the risk-neutral measure **Q**, the drift of each asset equals the risk-free rate:

```
dSᵢ(t) = r Sᵢ(t) dt + σᵢ Sᵢ(t) dWᵢᴼ(t)
```

Where **r** is the risk-free rate and **Wᴼ** denotes Brownian motion under **Q**.

#### Option Pricing Formula

The price of an option with payoff **h(S₁(T), ..., Sₙ(T))** at maturity **T** is:

```
V(0) = e⁻ʳᵀ · Eᴼ[h(S₁(T), ..., Sₙ(T))]
```

Where:
- **e⁻ʳᵀ**: Discount factor (present value)
- **Eᴼ[·]**: Expectation under risk-neutral measure
- **h(·)**: Payoff function

**Implementation** (multi_asset_option.cpp:228-229):
```cpp
double average_payoff = sum_payoffs / num_simulations;
return std::exp(-option.r * option.T) * average_payoff;
```

---

## Multi-Asset Option Types

### Basket Options

A **basket option** is an option on a weighted portfolio (basket) of underlying assets.

#### Basket Value

The basket value at time *t* is defined as:

```
B(t) = Σᵢ wᵢ · Sᵢ(t)
```

Where:
- **wᵢ**: Weight of asset *i* in the basket
- **Σᵢ wᵢ = 1**: Weights typically sum to 1
- **Sᵢ(t)**: Price of asset *i* at time *t*

#### Payoff Functions

**Basket Call Option:**
```
Payoff = max(B(T) - K, 0)
       = max(Σᵢ wᵢ·Sᵢ(T) - K, 0)
```

**Basket Put Option:**
```
Payoff = max(K - B(T), 0)
       = max(K - Σᵢ wᵢ·Sᵢ(T), 0)
```

Where **K** is the strike price.

**Implementation** (multi_asset_option.cpp:37-47):
```cpp
double payoff(const std::vector<double>& spot_prices) const override {
    double basket_value = 0.0;
    for (size_t i = 0; i < spot_prices.size(); ++i) {
        basket_value += weights[i] * spot_prices[i];
    }

    if (is_call) {
        return std::max(basket_value - K[0], 0.0);
    } else {
        return std::max(K[0] - basket_value, 0.0);
    }
}
```

#### Properties

- **Diversification Effect**: Basket volatility is typically lower than individual asset volatilities due to correlation < 1
- **Correlation Sensitivity**: Price increases with correlation for call options
- **Commonly Used**: For index options, portfolio hedging, structured products

---

### Rainbow Options

**Rainbow options** are path-independent multi-asset options whose payoff depends on the best-performing or worst-performing asset.

#### Best-of Option

Payoff based on the maximum asset value:

**Best-of Call:**
```
Payoff = max(max{S₁(T), S₂(T), ..., Sₙ(T)} - K, 0)
```

**Best-of Put:**
```
Payoff = max(K - max{S₁(T), S₂(T), ..., Sₙ(T)}, 0)
```

#### Worst-of Option

Payoff based on the minimum asset value:

**Worst-of Call:**
```
Payoff = max(min{S₁(T), S₂(T), ..., Sₙ(T)} - K, 0)
```

**Worst-of Put:**
```
Payoff = max(K - min{S₁(T), S₂(T), ..., Sₙ(T)}, 0)
```

**Implementation** (multi_asset_option.cpp:65-77):
```cpp
double payoff(const std::vector<double>& spot_prices) const override {
    double extreme_value;
    if (is_best_of) {
        extreme_value = *std::max_element(spot_prices.begin(), spot_prices.end());
    } else {
        extreme_value = *std::min_element(spot_prices.begin(), spot_prices.end());
    }

    if (is_call) {
        return std::max(extreme_value - K[0], 0.0);
    } else {
        return std::max(K[0] - extreme_value, 0.0);
    }
}
```

#### Properties

- **Best-of options**: More valuable than single-asset options (holder chooses best performer)
- **Worst-of options**: Less valuable than single-asset options (forced to take worst performer)
- **Correlation Sensitivity**: Best-of calls decrease in value with higher correlation
- **Alternative Names**: "Altiplano options", "Outperformance options"

---

### Exchange Options

An **exchange option** gives the holder the right to exchange one asset for another.

#### Payoff Function

```
Payoff = max(S₁(T) - S₂(T), 0)
```

The holder exchanges asset 2 for asset 1 if asset 1 is worth more.

**Implementation** (multi_asset_option.cpp:92-94):
```cpp
double payoff(const std::vector<double>& spot_prices) const override {
    if (spot_prices.size() < 2) return 0.0;
    return std::max(spot_prices[0] - spot_prices[1], 0.0);
}
```

#### Margrabe Formula

Exchange options have a closed-form solution known as the **Margrabe formula** (1978):

```
V = S₁(0) · N(d₁) - S₂(0) · N(d₂)
```

Where:

```
d₁ = [ln(S₁(0)/S₂(0)) + (σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

σ² = σ₁² + σ₂² - 2ρ₁₂σ₁σ₂
```

Here:
- **σ**: Volatility of the ratio S₁/S₂
- **N(·)**: Cumulative standard normal distribution
- **ρ₁₂**: Correlation between the two assets

#### Properties

- **No strike price**: The strike is effectively the second asset
- **Symmetric structure**: Similar to call option on ratio S₁/S₂
- **Applications**: Mergers & acquisitions, asset swaps, portfolio rebalancing

---

### Spread Options

A **spread option** has payoff based on the difference (spread) between two asset prices relative to a strike.

#### Payoff Functions

**Spread Call:**
```
Payoff = max(S₁(T) - S₂(T) - K, 0)
```

**Spread Put:**
```
Payoff = max(K - (S₁(T) - S₂(T)), 0)
```

Where **K** is the strike on the spread.

**Implementation** (multi_asset_option.cpp:111-119):
```cpp
double payoff(const std::vector<double>& spot_prices) const override {
    if (spot_prices.size() < 2) return 0.0;
    double spread = spot_prices[0] - spot_prices[1];

    if (is_call) {
        return std::max(spread - K[0], 0.0);
    } else {
        return std::max(K[0] - spread, 0.0);
    }
}
```

#### Relation to Exchange Options

- When **K = 0**, spread call equals exchange option: max(S₁ - S₂, 0)
- General case adds strike price to the spread

#### Applications

- **Energy markets**: Crack spreads (crude oil vs. refined products)
- **Fixed income**: Yield curve spreads
- **Commodities**: Location spreads (same commodity, different locations)
- **Equities**: Pairs trading strategies

---

## Monte Carlo Simulation

**Monte Carlo simulation** is a numerical method for pricing options by simulating random paths of underlying assets and averaging payoffs.

### Algorithm Overview

1. **Simulate** asset prices at maturity under risk-neutral measure
2. **Compute** option payoff for each simulation
3. **Average** all payoffs
4. **Discount** average payoff to present value

### Mathematical Foundation

By the Law of Large Numbers, as the number of simulations *N* → ∞:

```
(1/N) Σᵢ₌₁ᴺ h(Sᵢ(T)) → Eᴼ[h(S(T))]
```

The Monte Carlo estimator for option price is:

```
V̂(0) = e⁻ʳᵀ · (1/N) Σᵢ₌₁ᴺ h(Sⁱ₁(T), ..., Sⁱₙ(T))
```

Where the superscript *i* denotes the *i*-th simulation path.

**Implementation** (multi_asset_option.cpp:219-229):
```cpp
double price_option(const MultiAssetOption& option) {
    double sum_payoffs = 0.0;

    for (int i = 0; i < num_simulations; ++i) {
        std::vector<double> final_prices = model.simulate_final_prices(option.T);
        double payoff = option.payoff(final_prices);
        sum_payoffs += payoff;
    }

    double average_payoff = sum_payoffs / num_simulations;
    return std::exp(-option.r * option.T) * average_payoff;
}
```

### Convergence Rate

The standard error of the Monte Carlo estimator is:

```
SE = σ̂/√N
```

Where:
- **σ̂**: Sample standard deviation of payoffs
- **N**: Number of simulations

Key insight: Error decreases as **1/√N**, so:
- To halve the error, need **4× more simulations**
- To achieve 10× better accuracy, need **100× more simulations**

### Advantages of Monte Carlo

1. **Flexibility**: Can price any payoff structure
2. **High Dimensions**: Efficiency doesn't degrade much with number of assets
3. **Path Dependence**: Can handle path-dependent features
4. **Intuitive**: Directly simulates real-world scenarios

### Disadvantages

1. **Slow Convergence**: O(1/√N) rate
2. **Computational Cost**: Requires many simulations for accuracy
3. **Early Exercise**: Difficult for American options (requires backward induction)

---

## Statistical Analysis

### Confidence Intervals

A **confidence interval** provides a range within which the true option price likely falls.

#### Standard Error Calculation

For *N* simulations with discounted payoffs {V₁, V₂, ..., Vₙ}:

**Sample Mean:**
```
V̄ = (1/N) Σᵢ Vᵢ
```

**Sample Variance:**
```
s² = [1/(N-1)] Σᵢ (Vᵢ - V̄)²
```

**Standard Error:**
```
SE = s/√N
```

**Implementation** (multi_asset_option.cpp:244-249):
```cpp
double variance = 0.0;
for (double payoff : payoffs) {
    variance += (payoff - mean_payoff) * (payoff - mean_payoff);
}
variance /= (num_simulations - 1);
double std_error = std::sqrt(variance / num_simulations);
```

#### Confidence Interval Formula

For a confidence level α (typically 95%), the confidence interval is:

```
CI = V̄ ± z_(α/2) · SE
```

For 95% confidence: **z₀.₀₂₅ = 1.96**

```
CI₉₅% = V̄ ± 1.96 · SE
```

**Implementation** (multi_asset_option.cpp:254-256):
```cpp
double z_score = 1.96;  // for 95% confidence
double margin_error = z_score * price_std_error;
return {price, margin_error};
```

#### Interpretation

"We are 95% confident that the true option price lies within [V̄ - 1.96·SE, V̄ + 1.96·SE]"

### Central Limit Theorem

The justification for using the normal distribution is the **Central Limit Theorem**:

```
(V̄ - E[V]) / (σ/√N) → N(0, 1)  as N → ∞
```

Even if individual payoffs are not normally distributed, their average approaches a normal distribution for large *N*.

---

## Option Greeks

**Greeks** measure the sensitivity of option prices to changes in input parameters.

### Delta (Δ)

**Delta** measures the rate of change of option price with respect to the underlying asset price.

#### Definition

For asset *i*:

```
Δᵢ = ∂V/∂Sᵢ
```

#### Finite Difference Approximation

For small bump Δ*S*:

```
Δᵢ ≈ [V(S₁, ..., Sᵢ + ΔS, ..., Sₙ) - V(S₁, ..., Sᵢ, ..., Sₙ)] / ΔS
```

Alternatively, **central difference** (more accurate):

```
Δᵢ ≈ [V(S₁, ..., Sᵢ + ΔS, ..., Sₙ) - V(S₁, ..., Sᵢ - ΔS, ..., Sₙ)] / (2ΔS)
```

**Framework** (multi_asset_option.cpp:273-283):
```cpp
std::vector<double> calculate_delta(const MultiAssetOption& option) {
    std::vector<double> deltas;
    double base_price = pricer.price_option(option);

    // Finite difference approximation for each underlying
    // Note: This is a simplified approach - in practice, you'd need to modify the model
    std::cout << "Delta calculation requires model modification for proper implementation.\n";
    std::cout << "Base price: " << base_price << std::endl;

    return deltas;
}
```

### Other Greeks

While not fully implemented, the framework can be extended for:

**Gamma (Γ)**: Second derivative with respect to spot price
```
Γᵢ = ∂²V/∂Sᵢ²
```

**Vega (ν)**: Sensitivity to volatility
```
νᵢ = ∂V/∂σᵢ
```

**Rho (ρ)**: Sensitivity to interest rate
```
ρ = ∂V/∂r
```

**Theta (Θ)**: Time decay
```
Θ = -∂V/∂t
```

---

## Implementation Details

### Class Hierarchy

```
MultiAssetOption (Abstract Base Class)
    ├── BasketOption
    ├── RainbowOption
    ├── ExchangeOption
    └── SpreadOption
```

Each derived class implements:
- `payoff()`: Computes option payoff given final asset prices
- `get_type()`: Returns descriptive string of option type

### TwoFactorModel Class

Handles simulation of correlated asset prices:

**Key Methods:**
- `generate_correlated_randoms()`: Uses Cholesky decomposition
- `simulate_final_prices(T)`: Generates prices at maturity *T*
- `simulate_paths(T, steps)`: Generates full price paths (for path-dependent options)

**Path Simulation** (multi_asset_option.cpp:185-206):
```cpp
std::vector<std::vector<double>> simulate_paths(double T, int time_steps) {
    double dt = T / time_steps;
    std::vector<std::vector<double>> paths(S0.size(), std::vector<double>(time_steps + 1));

    // Initialize with spot prices
    for (size_t i = 0; i < S0.size(); ++i) {
        paths[i][0] = S0[i];
    }

    // Generate path
    for (int t = 1; t <= time_steps; ++t) {
        std::vector<double> Z = generate_correlated_randoms();

        for (size_t i = 0; i < S0.size(); ++i) {
            double drift = (mu[i] - 0.5 * sigma[i] * sigma[i]) * dt;
            double diffusion = sigma[i] * std::sqrt(dt) * Z[i];
            paths[i][t] = paths[i][t-1] * std::exp(drift + diffusion);
        }
    }

    return paths;
}
```

### MonteCarloMultiAssetPricer Class

Performs Monte Carlo pricing:

**Key Methods:**
- `price_option()`: Returns point estimate of option price
- `price_with_confidence()`: Returns price with confidence interval

### Random Number Generation

Uses C++ `<random>` library:
- **Engine**: `std::mt19937` (Mersenne Twister)
- **Distribution**: `std::normal_distribution<double>`
- **Seeding**: `std::random_device` for non-deterministic seed

---

## Usage Guide

### Interactive Mode

The program provides an interactive command-line interface:

1. **Asset Parameters**: Enter number of assets and their properties (S₀, μ, σ)
2. **Correlation Matrix**: Specify correlations between assets
3. **Option Parameters**: Set time to maturity *T* and risk-free rate *r*
4. **Simulation Parameters**: Choose number of Monte Carlo simulations
5. **Option Selection**: Select which option type to price

### Menu Options

1. **Basket Option**: Specify weights and strike
2. **Rainbow Option (Best-of)**: Select best-performing asset
3. **Rainbow Option (Worst-of)**: Select worst-performing asset
4. **Exchange Option**: Option to exchange asset 1 for asset 2
5. **Spread Option**: Option on spread between two assets
6. **Price All Options**: Compute prices for all option types simultaneously
7. **Correlation Sensitivity Analysis**: Analyze how prices vary with correlation

### Example Workflow

```
Number of assets: 2

Asset 1:
  Initial price: $100
  Expected return: 0.05
  Volatility: 0.20

Asset 2:
  Initial price: $95
  Expected return: 0.06
  Volatility: 0.25

Correlation between Asset 1 and Asset 2: 0.50

Time to maturity: 1.0 years
Risk-free rate: 0.03
Number of simulations: 100000

Select option type: 1 (Basket Option)
Strike price: $100
Call or Put: C (Call)
Weights: 0.5, 0.5

Result: Price: $8.45 ± $0.12
```

---

## Compilation and Execution

### Requirements

- C++17 or later
- Standard library support for `<random>`, `<vector>`, `<algorithm>`

### Compilation

Using g++:
```bash
g++ -std=c++17 -O3 -o multi_asset_option multi_asset_option.cpp
```

Using clang++:
```bash
clang++ -std=c++17 -O3 -o multi_asset_option multi_asset_option.cpp
```

### Optimization Flags

- `-O3`: Maximum optimization for speed
- `-march=native`: Optimize for local CPU architecture
- `-flto`: Link-time optimization

Full optimized compilation:
```bash
g++ -std=c++17 -O3 -march=native -flto -o multi_asset_option multi_asset_option.cpp
```

### Execution

```bash
./multi_asset_option
```

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| **S(t)** or **Sᵢ(t)** | Price of asset (or asset *i*) at time *t* |
| **S(0)** or **S₀** | Initial spot price |
| **S(T)** | Final price at maturity |
| **μ** or **μᵢ** | Drift rate (expected return) |
| **σ** or **σᵢ** | Volatility (standard deviation of returns) |
| **r** | Risk-free interest rate |
| **T** | Time to maturity (in years) |
| **K** | Strike price |
| **ρᵢⱼ** | Correlation between asset *i* and *j* |
| **W(t)** or **Wᵢ(t)** | Wiener process (Brownian motion) |
| **Z** or **Zᵢ** | Standard normal random variable |
| **V** or **V(t)** | Option value/price |
| **h(·)** | Payoff function |
| **E[·]** or **Eᴼ[·]** | Expectation (under risk-neutral measure) |
| **N(·)** | Cumulative standard normal distribution |
| **Ρ** | Correlation matrix |
| **L** | Cholesky decomposition matrix |

---

## References and Further Reading

### Key Papers

1. **Black, F., & Scholes, M. (1973)**. "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. **Margrabe, W. (1978)**. "The Value of an Option to Exchange One Asset for Another." *Journal of Finance*, 33(1), 177-186.

3. **Boyle, P. P. (1977)**. "Options: A Monte Carlo approach." *Journal of Financial Economics*, 4(3), 323-338.

### Textbooks

- **Hull, J. C.** *Options, Futures, and Other Derivatives* (Chapter on Exotic Options)
- **Glasserman, P.** *Monte Carlo Methods in Financial Engineering* (Springer)
- **Joshi, M. S.** *The Concepts and Practice of Mathematical Finance*

### Mathematical Concepts

- **Stochastic Calculus**: Itô's Lemma, Stochastic Differential Equations
- **Numerical Methods**: Monte Carlo Simulation, Variance Reduction Techniques
- **Linear Algebra**: Cholesky Decomposition, Positive Definite Matrices
- **Probability Theory**: Central Limit Theorem, Law of Large Numbers

---

## License

This implementation is provided for educational and research purposes.

---

## Notes

This system demonstrates:
- Object-oriented design in C++ for financial derivatives
- Implementation of advanced stochastic models
- Numerical methods for high-dimensional problems
- Statistical analysis of simulation results

The code prioritizes clarity and correctness over performance optimization, making it suitable for learning and prototyping.

**Note**: For production use, consider:
- Variance reduction techniques (antithetic variates, control variates)
- Parallel computation (OpenMP, CUDA)
- More sophisticated correlation handling (spectral decomposition)
- Historical calibration and model validation

---

**Disclaimer**: This implementation is for educational and research purposes. Financial decisions should not be made solely based on this model without considering its limitations and consulting with qualified financial professionals.
