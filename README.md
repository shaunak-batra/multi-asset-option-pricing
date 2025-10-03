# README: Multi-Asset Option Pricing with Monte Carlo Simulation

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
   - [Options and Multi-Asset Options](#options-and-multi-asset-options)
   - [Geometric Brownian Motion](#geometric-brownian-motion)
   - [Monte Carlo Simulation](#monte-carlo-simulation)
   - [Correlation and Cholesky Decomposition](#correlation-and-cholesky-decomposition)
3. [Option Pricing Models](#option-pricing-models)
   - [Basket Option](#basket-option)
   - [Rainbow Option](#rainbow-option)
   - [Exchange Option](#exchange-option)
   - [Spread Option](#spread-option)
4. [Code Architecture](#code-architecture)
   - [Class Hierarchy](#class-hierarchy)
   - [Key Components](#key-components)
5. [Implementation Details](#implementation-details)
   - [TwoFactorModel](#twofactormodel)
   - [MonteCarloMultiAssetPricer](#montecarlomultiassetpricer)
   - [GreeksCalculator](#greekscalculator)
   - [Input Validation](#input-validation)
   - [Interactive Pricing](#interactive-pricing)
6. [Optimization Techniques](#optimization-techniques)
   - [Efficient Random Number Generation](#efficient-random-number-generation)
   - [Simplified Cholesky Decomposition](#simplified-cholesky-decomposition)
   - [Monte Carlo Efficiency](#monte-carlo-efficiency)
7. [Usage Examples](#usage-examples)
   - [Pricing a Basket Option](#pricing-a-basket-option)
   - [Correlation Sensitivity Analysis](#correlation-sensitivity-analysis)
8. [Limitations and Assumptions](#limitations-and-assumptions)
   - [Model Assumptions](#model-assumptions)
   - [Implementation Limitations](#implementation-limitations)
9. [Compilation and Execution](#compilation-and-execution)
   - [Dependencies](#dependencies)
   - [How to Compile](#how-to-compile)
   - [Running the Program](#running-the-program)
10. [Testing and Validation](#testing-and-validation)
    - [Unit Testing](#unit-testing)
    - [Monte Carlo Convergence](#monte-carlo-convergence)

---

## Overview

This C++ program is designed to price **multi-asset options**, which are financial instruments whose value depends on the performance of multiple underlying assets (like stocks, commodities, or indices). Unlike single-asset options (e.g., a stock option), multi-asset options are more complex because their payoff depends on the combined behavior of several assets, which may move together or independently.

The program uses a **Monte Carlo simulation** to estimate option prices. Monte Carlo simulation is like rolling a dice many times to predict possible outcomes—in this case, to simulate how asset prices might move in the future and calculate the option's value based on those scenarios. The code supports four types of multi-asset options: **Basket**, **Rainbow**, **Exchange**, and **Spread** options, each with different payoff structures.

The program is interactive, allowing users to input parameters like asset prices, volatilities, correlations, and the number of simulations. It also provides confidence intervals for price estimates and supports sensitivity analysis to understand how changes in asset correlations affect option prices. The code is modular, object-oriented, and includes input validation to ensure robustness.

This README explains the mathematics behind the options and the simulation, the code structure, optimizations, and how to use the program, making it accessible even if you're not a math or finance expert.

---

## Mathematical Foundation

This section explains the key mathematical concepts used in the code, broken down into simple terms for clarity.

### Options and Multi-Asset Options

An **option** is a financial contract that gives you the right (but not the obligation) to buy or sell an asset at a specific price (called the **strike price**) by a certain date (the **expiration date**). For example, a **call option** lets you buy an asset, while a **put option** lets you sell it.

**Multi-asset options** depend on multiple assets. For instance:
- A **Basket Option** depends on the weighted average of several asset prices.
- A **Rainbow Option** depends on the best or worst-performing asset.
- An **Exchange Option** lets you swap one asset for another.
- A **Spread Option** depends on the difference between two asset prices.

The challenge is that these assets don’t move independently—their prices are often **correlated** (e.g., if one stock rises, another might rise too). The program accounts for this correlation when simulating future prices.

### Geometric Brownian Motion

The program assumes that asset prices follow a **Geometric Brownian Motion (GBM)** model. Imagine asset prices as a wiggly line on a graph that tends to drift upward or downward over time but also has random fluctuations.

- **Drift**: This is the expected average movement of the asset price, like a gentle push upward if the asset is expected to grow.
- **Volatility**: This measures how much the price wiggles randomly. High volatility means big, unpredictable swings.
- **Random Component**: The wiggles are modeled as random numbers drawn from a **normal distribution** (a bell-shaped curve where most values cluster around the average).

Mathematically, for an asset with price \( S_t \) at time \( t \), GBM is described by:

\[ S_t = S_0 \exp\left( \left( \mu - \frac{\sigma^2}{2} \right)t + \sigma \sqrt{t} Z \right) \]

Where:
- \( S_0 \): Initial price of the asset.
- \( \mu \): Expected return (drift).
- \( \sigma \): Volatility (how much the price fluctuates).
- \( t \): Time.
- \( Z \): A random number from a normal distribution (mean 0, standard deviation 1).
- \( \exp \): The exponential function (e raised to a power).

The program uses this formula to simulate future asset prices at the option’s expiration.

### Monte Carlo Simulation

Monte Carlo simulation is a method to estimate outcomes by running many random scenarios. Here’s how it works for option pricing:

1. **Simulate Asset Prices**: Generate thousands of possible future price paths for all assets using GBM, accounting for their correlations.
2. **Calculate Payoff**: For each scenario, compute the option’s payoff (how much money you’d make if the option were exercised).
3. **Average and Discount**: Average the payoffs across all scenarios and discount them back to today’s value using the **risk-free rate** (like the interest rate on a safe investment, e.g., government bonds).

The formula for the option price is:

\[ \text{Option Price} = e^{-rT} \cdot \text{Average Payoff} \]

Where:
- \( r \): Risk-free rate.
- \( T \): Time to maturity (in years).
- \( e^{-rT} \): Discounts future value to present value (because money today is worth more than money in the future).

The more simulations you run, the more accurate the price estimate, but it takes longer to compute.

### Correlation and Cholesky Decomposition

When assets are correlated, their price movements are linked. For example, if two stocks are in the same industry, they might rise or fall together. This is captured by a **correlation matrix**, where each entry (between -1 and 1) shows how strongly two assets move together:
- \( 1 \): Perfectly correlated (move exactly together).
- \( -1 \): Perfectly negatively correlated (move in opposite directions).
- \( 0 \): Independent.

To simulate correlated asset prices, the program uses **Cholesky decomposition**. Think of it as a recipe to transform independent random numbers (like rolling separate dice) into correlated random numbers (like dice that influence each other).

For two assets with correlation \( \rho \), the Cholesky decomposition creates correlated random numbers \( Z_1 \) and \( Z_2 \):

\[ Z_1 = W_1 \]
\[ Z_2 = \rho W_1 + \sqrt{1 - \rho^2} W_2 \]

Where \( W_1 \) and \( W_2 \) are independent random numbers from a normal distribution. This ensures the simulated price movements respect the correlation structure.

---

## Option Pricing Models

The code supports four types of multi-asset options, each with a unique payoff structure. Let’s break them down.

### Basket Option

A **Basket Option** depends on the weighted average of multiple asset prices. For example, if you have two stocks priced at $100 and $50 with weights 0.6 and 0.4, the basket value is:

\[ 0.6 \times 100 + 0.4 \times 50 = 60 + 20 = 80 \]

The payoff for a call option is:

\[ \text{Payoff} = \max(\text{Basket Value} - \text{Strike}, 0) \]

For a put option, it’s:

\[ \text{Payoff} = \max(\text{Strike} - \text{Basket Value}, 0) \]

The code allows users to specify weights and whether it’s a call or put option.

### Rainbow Option

A **Rainbow Option** focuses on either the **best** or **worst** performing asset among a group. For example:
- **Best-of Call**: Pays off based on the highest asset price.
- **Worst-of Put**: Pays off based on the lowest asset price.

The payoff for a best-of call is:

\[ \text{Payoff} = \max(\text{Max Asset Price} - \text{Strike}, 0) \]

For a worst-of put:

\[ \text{Payoff} = \max(\text{Strike} - \text{Min Asset Price}, 0) \]

The code uses `std::max_element` and `std::min_element` to find the extreme values.

### Exchange Option

An **Exchange Option** lets you swap one asset for another. If you have two assets with prices \( S_1 \) and \( S_2 \), the payoff is:

\[ \text{Payoff} = \max(S_1 - S_2, 0) \]

This means you profit if the first asset is worth more than the second at expiration. No strike price is needed.

### Spread Option

A **Spread Option** depends on the difference (spread) between two asset prices. For a call option:

\[ \text{Payoff} = \max(S_1 - S_2 - \text{Strike}, 0) \]

For a put option:

\[ \text{Payoff} = \max(\text{Strike} - (S_1 - S_2), 0) \]

This is useful for betting on the relative performance of two assets.

---

## Code Architecture

The code is designed using **object-oriented programming (OOP)** principles for modularity and extensibility. Below is an overview of its structure.

### Class Hierarchy

The program uses a class hierarchy to model options and their pricing:

- **MultiAssetOption**: An abstract base class for all multi-asset options. It defines:
  - Common attributes: Time to maturity (\( T \)), risk-free rate (\( r \)), strike prices (\( K \)).
  - Pure virtual functions: `payoff` (to compute the option’s payoff) and `get_type` (to identify the option type).
- **Derived Classes**: Specific option types inherit from `MultiAssetOption`:
  - `BasketOption`: Handles weighted basket options.
  - `RainbowOption`: Handles best-of/worst-of options.
  - `ExchangeOption`: Handles asset swaps.
  - `SpreadOption`: Handles price spreads.
- **TwoFactorModel**: Models the evolution of multiple correlated assets using GBM.
- **MonteCarloMultiAssetPricer**: Performs Monte Carlo simulations to price options.
- **GreeksCalculator**: Computes option sensitivities (though delta calculation is incomplete in the code).

### Key Components

- **Random Number Generation**: Uses `std::mt19937` (Mersenne Twister) for high-quality random numbers and `std::normal_distribution` for normal random variables.
- **Input Validation**: Functions like `get_positive_double` and `get_correlation` ensure robust user input.
- **Interactive Interface**: The `run_interactive_pricing` function guides users through parameter input and option selection.

---

## Implementation Details

This section dives into the code’s key components, explaining how they work and why they’re implemented that way.

### TwoFactorModel

The `TwoFactorModel` class simulates asset price paths using GBM, accounting for correlations.

- **Constructor**: Takes initial prices (\( S_0 \)), drifts (\( \mu \)), volatilities (\( \sigma \)), and a correlation matrix.
- **generate_correlated_randoms**: Uses Cholesky decomposition to produce correlated random numbers. For two assets, it implements a simplified formula; for more assets, it falls back to using independent randoms (a limitation).
- **simulate_final_prices**: Generates asset prices at maturity using the GBM formula.
- **simulate_paths**: Generates full price paths over multiple time steps, useful for path-dependent options (though not used in the current code).

**Why It Matters**: This class is the core of the simulation, ensuring realistic price movements that respect correlations.

### MonteCarloMultiAssetPricer

This class runs Monte Carlo simulations to price options.

- **price_option**: Runs `num_simulations` scenarios, computes payoffs, and discounts the average to get the option price.
- **price_with_confidence**: Computes the option price and a 95% confidence interval by calculating the standard error of the payoffs. The confidence interval tells you how reliable the price estimate is.

**Math Behind Confidence Interval**:
- The standard error measures the variability of the average payoff.
- For a 95% confidence interval, the code uses a z-score of 1.96 (from the normal distribution) to estimate the margin of error:

\[ \text{Margin of Error} = 1.96 \times \text{Standard Error} \]

\[ \text{Standard Error} = \sqrt{\frac{\text{Variance of Payoffs}}{\text{Number of Simulations}}} \]

This helps users understand the precision of the Monte Carlo estimate.

### GreeksCalculator

The `GreeksCalculator` class is meant to compute **Greeks** (sensitivities of the option price to various factors, like asset prices). Currently, it only has a placeholder for delta (sensitivity to asset price changes) using a finite difference method, but it’s incomplete.

**Why Greeks Matter**: Greeks help traders understand how option prices change with market conditions, aiding in risk management. The incomplete implementation is a limitation.

### Input Validation

Functions like `get_positive_double`, `get_correlation`, and `get_option_choice` ensure that user inputs are valid (e.g., positive numbers for prices, correlations between -1 and 1). They use loops to prompt users until valid input is provided, making the program user-friendly and robust.

### Interactive Pricing

The `run_interactive_pricing` function is the main entry point:
- Prompts for the number of assets, their parameters, correlations, and Monte Carlo settings.
- Displays a model summary.
- Lets users choose an option type to price or perform a correlation sensitivity analysis.
- Handles specific inputs (e.g., weights for basket options) and normalizes weights if needed.

This makes the program accessible to users without coding expertise.

---

## Optimization Techniques

The code includes several optimizations to improve performance and usability.

### Efficient Random Number Generation

- **Mersenne Twister (`std::mt19937`)**: A high-quality random number generator that produces reliable, repeatable random numbers.
- **Single Instance**: The `TwoFactorModel` creates one `std::mt19937` instance, reused across simulations, avoiding the overhead of repeated initialization.
- **Normal Distribution**: Uses `std::normal_distribution` for efficient generation of normally distributed random numbers, critical for GBM.

### Simplified Cholesky Decomposition

For two assets, the code uses a simplified Cholesky decomposition formula, avoiding the need for a full matrix decomposition. This is faster and sufficient for the common case of two assets:

\[ Z_1 = W_1 \]
\[ Z_2 = \rho W_1 + \sqrt{1 - \rho^2} W_2 \]

For more than two assets, it falls back to independent randoms, which is less computationally intensive but less accurate.

### Monte Carlo Efficiency

- **Single-Loop Simulation**: The `price_option` method uses a single loop for simulations, minimizing overhead.
- **Reduced Simulations for Sensitivity Analysis**: The correlation sensitivity analysis uses half the number of simulations (`num_simulations/2`) to speed up computation while still providing meaningful results.
- **Vector Operations**: Uses `std::vector` for efficient storage and iteration over asset prices and payoffs.

---

## Usage Examples

Here’s how to use the program, with examples.

### Pricing a Basket Option

1. Run the program.
2. Enter the number of assets (e.g., 2).
3. For each asset, input:
   - Initial price (e.g., $100, $50).
   - Expected return (e.g., 0.05, 0.03).
   - Volatility (e.g., 0.2, 0.15).
4. Enter the correlation (e.g., 0.5).
5. Specify option parameters:
   - Time to maturity (e.g., 1 year).
   - Risk-free rate (e.g., 0.02).
   - Number of simulations (e.g., 100,000).
6. Choose option type 1 (Basket Option).
7. Enter strike price (e.g., $75) and call/put (e.g., ‘C’ for call).
8. Enter weights (e.g., 0.6, 0.4). If they don’t sum to 1, choose to normalize.
9. View the price and confidence interval (e.g., “Price: $10.25 ± $0.15”).

### Correlation Sensitivity Analysis

1. Follow steps 1–5 above, ensuring exactly 2 assets.
2. Choose option type 7 (Correlation Sensitivity Analysis).
3. The program tests correlations from -0.8 to 0.8 and shows how the basket call option price changes (e.g., higher correlations may increase the price due to synchronized asset movements).

---

## Limitations and Assumptions

### Model Assumptions

- **Geometric Brownian Motion**: Assumes asset prices follow GBM, which may not capture real-world complexities like jumps or fat-tailed distributions.
- **Constant Parameters**: Assumes constant drift, volatility, and correlation over time, which isn’t always realistic.
- **Risk-Neutral Pricing**: Uses the risk-free rate for discounting, assuming a risk-neutral world (standard in option pricing but not always reflective of market dynamics).

### Implementation Limitations

- **Two-Asset Focus**: The Cholesky decomposition is optimized for two assets; for more assets, it uses independent randoms, ignoring correlations.
- **Incomplete Greeks**: The delta calculation is a placeholder and not fully implemented.
- **No Path-Dependent Options**: The `simulate_paths` function exists but isn’t used, limiting the program to European-style options (exercised only at maturity).
- **No Parallelization**: Monte Carlo simulations are single-threaded, which can be slow for large numbers of simulations.

---

## Compilation and Execution

### Dependencies

- A C++ compiler supporting C++11 or later (e.g., g++, clang++).
- Standard C++ library (included in most compilers).
- No external libraries required.

### How to Compile

Save the code as `multi_asset_option.cpp` and compile with:

```bash
g++ -std=c++11 multi_asset_option.cpp -o multi_asset_option
```

For optimization (faster execution):

```bash
g++ -std=c++11 -O3 multi_asset_option.cpp -o multi_asset_option
```

### Running the Program

Run the executable:

```bash
./multi_asset_option
```

Follow the interactive prompts to input parameters and select an option type.

---

## Testing and Validation

### Unit Testing

The code doesn’t include explicit unit tests, but you can validate it by:
- **Known Cases**: Compare results with analytical solutions for simple cases (e.g., a basket option with one asset should match Black-Scholes).
- **Edge Cases**: Test extreme correlations (-1, 1) or zero volatility to ensure payoffs are correct.

### Monte Carlo Convergence

To verify accuracy:
- Increase `num_simulations` (e.g., from 10,000 to 1,000,000) and check if the price converges (changes less).
- The confidence interval (`±` value) should shrink as the number of simulations increases, indicating a more precise estimate.

=== Interactive Multi-Asset Option Pricing ===

Enter number of underlying assets (2 recommended): 2

--- Asset Parameters ---
Asset 1:
  Initial price (S0): $100
  Expected return (mu): 0.05
  Volatility (sigma): 0.20

Asset 2:
  Initial price (S0): $110  
  Expected return (mu): 0.06
  Volatility (sigma): 0.25

--- Correlation Matrix ---
Correlation between Asset 1 and Asset 2: 0.3

--- Option Parameters ---
Time to maturity (years): 1.0
Risk-free rate: 0.03

--- Simulation Parameters ---
Number of Monte Carlo simulations: 100000

Select option type to price:
1. Basket Option
2. Rainbow Option (Best-of)  
3. Rainbow Option (Worst-of)
4. Exchange Option
5. Spread Option
6. Price All Options
7. Correlation Sensitivity Analysis
Enter your choice (1-7): 1

--- BASKET OPTION ---

Strike price: $105

Call or Put option? (C/P): C

Enter weights for basket (should sum to 1.0):

Weight for Asset 1: 0.6

Weight for Asset 2: 0.4

Basket Call Results:

Price: $12.45 ± $0.23

P.S: This implementation is for educational and research purposes. Financial decisions should not be made solely based on this model without considering its limitations and consulting with qualified financial professionals.
