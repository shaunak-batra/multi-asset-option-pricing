#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <memory>
#include <functional>
#include <iomanip>
#include <cctype>

// Base class for multi-asset options
class MultiAssetOption {
public:
    double T;           // Time to maturity
    double r;           // Risk-free rate
    std::vector<double> K;  // Strike prices (can be multiple for different options)
    
    MultiAssetOption(double time_to_maturity, double risk_free_rate, std::vector<double> strikes)
        : T(time_to_maturity), r(risk_free_rate), K(strikes) {}
    
    virtual ~MultiAssetOption() = default;
    virtual double payoff(const std::vector<double>& spot_prices) const = 0;
    virtual std::string get_type() const = 0;
};

// Basket Option - payoff based on weighted sum of assets
class BasketOption : public MultiAssetOption {
private:
    std::vector<double> weights;
    bool is_call;
    
public:
    BasketOption(double T, double r, double strike, std::vector<double> w, bool call = true)
        : MultiAssetOption(T, r, {strike}), weights(w), is_call(call) {}
    
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
    
    std::string get_type() const override {
        return is_call ? "Basket Call" : "Basket Put";
    }
};

// Rainbow Option - best/worst of multiple assets
class RainbowOption : public MultiAssetOption {
private:
    bool is_call;
    bool is_best_of;  // true for best-of, false for worst-of
    
public:
    RainbowOption(double T, double r, double strike, bool call = true, bool best = true)
        : MultiAssetOption(T, r, {strike}), is_call(call), is_best_of(best) {}
    
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
    
    std::string get_type() const override {
        std::string type = is_call ? "Call" : "Put";
        std::string variant = is_best_of ? "Best-of" : "Worst-of";
        return variant + " Rainbow " + type;
    }
};

// Exchange Option - option to exchange one asset for another
class ExchangeOption : public MultiAssetOption {
public:
    ExchangeOption(double T, double r) : MultiAssetOption(T, r, {}) {}
    
    double payoff(const std::vector<double>& spot_prices) const override {
        if (spot_prices.size() < 2) return 0.0;
        return std::max(spot_prices[0] - spot_prices[1], 0.0);
    }
    
    std::string get_type() const override {
        return "Exchange Option";
    }
};

// Spread Option - payoff based on spread between two assets
class SpreadOption : public MultiAssetOption {
private:
    bool is_call;
    
public:
    SpreadOption(double T, double r, double strike, bool call = true)
        : MultiAssetOption(T, r, {strike}), is_call(call) {}
    
    double payoff(const std::vector<double>& spot_prices) const override {
        if (spot_prices.size() < 2) return 0.0;
        double spread = spot_prices[0] - spot_prices[1];
        
        if (is_call) {
            return std::max(spread - K[0], 0.0);
        } else {
            return std::max(K[0] - spread, 0.0);
        }
    }
    
    std::string get_type() const override {
        return is_call ? "Spread Call" : "Spread Put";
    }
};

// Two-Factor Model for correlated asset price evolution
class TwoFactorModel {
private:
    std::vector<double> S0;     // Initial spot prices
    std::vector<double> mu;     // Drift rates
    std::vector<double> sigma;  // Volatilities
    std::vector<std::vector<double>> correlation; // Correlation matrix
    std::mt19937 rng;
    std::normal_distribution<double> norm_dist;
    
public:
    TwoFactorModel(std::vector<double> initial_prices, 
                   std::vector<double> drift_rates,
                   std::vector<double> volatilities,
                   std::vector<std::vector<double>> corr_matrix)
        : S0(initial_prices), mu(drift_rates), sigma(volatilities), 
          correlation(corr_matrix), rng(std::random_device{}()), norm_dist(0.0, 1.0) {}
    
    // Generate correlated random numbers using Cholesky decomposition
    std::vector<double> generate_correlated_randoms() {
        size_t n = S0.size();
        std::vector<double> independent_randoms(n);
        std::vector<double> correlated_randoms(n, 0.0);
        
        // Generate independent random numbers
        for (size_t i = 0; i < n; ++i) {
            independent_randoms[i] = norm_dist(rng);
        }
        
        // Simple Cholesky decomposition for 2x2 case
        if (n == 2) {
            double rho = correlation[0][1];
            correlated_randoms[0] = independent_randoms[0];
            correlated_randoms[1] = rho * independent_randoms[0] + 
                                   std::sqrt(1 - rho * rho) * independent_randoms[1];
        } else {
            // For larger matrices, use full Cholesky (simplified here)
            correlated_randoms = independent_randoms;
        }
        
        return correlated_randoms;
    }
    
    // Simulate asset prices at maturity using geometric Brownian motion
    std::vector<double> simulate_final_prices(double T) {
        std::vector<double> Z = generate_correlated_randoms();
        std::vector<double> final_prices(S0.size());
        
        for (size_t i = 0; i < S0.size(); ++i) {
            double drift = (mu[i] - 0.5 * sigma[i] * sigma[i]) * T;
            double diffusion = sigma[i] * std::sqrt(T) * Z[i];
            final_prices[i] = S0[i] * std::exp(drift + diffusion);
        }
        
        return final_prices;
    }
    
    // Monte Carlo path simulation for path-dependent options
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
};

// Monte Carlo Pricer
class MonteCarloMultiAssetPricer {
private:
    TwoFactorModel model;
    int num_simulations;
    
public:
    MonteCarloMultiAssetPricer(TwoFactorModel m, int sims) 
        : model(m), num_simulations(sims) {}
    
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
    
    // Price with confidence interval
    std::pair<double, double> price_with_confidence(const MultiAssetOption& option, double confidence_level = 0.95) {
        std::vector<double> payoffs(num_simulations);
        
        for (int i = 0; i < num_simulations; ++i) {
            std::vector<double> final_prices = model.simulate_final_prices(option.T);
            payoffs[i] = option.payoff(final_prices);
        }
        
        double mean_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / num_simulations;
        
        // Calculate standard error
        double variance = 0.0;
        for (double payoff : payoffs) {
            variance += (payoff - mean_payoff) * (payoff - mean_payoff);
        }
        variance /= (num_simulations - 1);
        double std_error = std::sqrt(variance / num_simulations);
        
        double price = std::exp(-option.r * option.T) * mean_payoff;
        double price_std_error = std::exp(-option.r * option.T) * std_error;
        
        // 95% confidence interval (assuming normal distribution)
        double z_score = 1.96;  // for 95% confidence
        double margin_error = z_score * price_std_error;
        
        return {price, margin_error};
    }
};

// Utility functions for option Greeks (Delta approximation)
class GreeksCalculator {
private:
    TwoFactorModel& model;
    MonteCarloMultiAssetPricer& pricer;
    double bump_size;
    
public:
    GreeksCalculator(TwoFactorModel& m, MonteCarloMultiAssetPricer& p, double bump = 0.01)
        : model(m), pricer(p), bump_size(bump) {}
    
    std::vector<double> calculate_delta(const MultiAssetOption& option) {
        std::vector<double> deltas;
        double base_price = pricer.price_option(option);
        
        // Finite difference approximation for each underlying
        // Note: This is a simplified approach - in practice, you'd need to modify the model
        std::cout << "Delta calculation requires model modification for proper implementation.\n";
        std::cout << "Base price: " << base_price << std::endl;
        
        return deltas;
    }
};

// Input validation functions
double get_positive_double(const std::string& prompt) {
    double value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value <= 0) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter a positive number.\n";
        } else {
            std::cin.ignore(10000, '\n');
            return value;
        }
    }
}

double get_double(const std::string& prompt) {
    double value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter a valid number.\n";
        } else {
            std::cin.ignore(10000, '\n');
            return value;
        }
    }
}

double get_correlation(const std::string& prompt) {
    double value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value < -1.0 || value > 1.0) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter a correlation between -1.0 and 1.0.\n";
        } else {
            std::cin.ignore(10000, '\n');
            return value;
        }
    }
}

int get_positive_int(const std::string& prompt) {
    int value;
    while (true) {
        std::cout << prompt;
        std::cin >> value;
        if (std::cin.fail() || value <= 0) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter a positive integer.\n";
        } else {
            std::cin.ignore(10000, '\n');
            return value;
        }
    }
}

int get_option_choice() {
    int choice;
    while (true) {
        std::cout << "\nSelect option type to price:\n";
        std::cout << "1. Basket Option\n";
        std::cout << "2. Rainbow Option (Best-of)\n";
        std::cout << "3. Rainbow Option (Worst-of)\n";
        std::cout << "4. Exchange Option\n";
        std::cout << "5. Spread Option\n";
        std::cout << "6. Price All Options\n";
        std::cout << "7. Correlation Sensitivity Analysis\n";
        std::cout << "Enter your choice (1-7): ";
        
        std::cin >> choice;
        if (std::cin.fail() || choice < 1 || choice > 7) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter a number between 1 and 7.\n";
        } else {
            std::cin.ignore(10000, '\n');
            return choice;
        }
    }
}

bool get_call_put_choice() {
    char choice;
    while (true) {
        std::cout << "Call or Put option? (C/P): ";
        std::cin >> choice;
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "Please enter 'C' for Call or 'P' for Put.\n";
        } else {
            std::cin.ignore(10000, '\n');
            choice = std::toupper(choice);
            if (choice == 'C') return true;
            else if (choice == 'P') return false;
            else std::cout << "Please enter 'C' for Call or 'P' for Put.\n";
        }
    }
}

// Interactive user input function
void run_interactive_pricing() {
    std::cout << "=== Interactive Multi-Asset Option Pricing ===\n\n";
    
    // Get number of assets
    int num_assets;
    while (true) {
        num_assets = get_positive_int("Enter number of underlying assets (2 recommended): ");
        if (num_assets >= 2) break;
        std::cout << "Multi-asset options require at least 2 assets.\n";
    }
    
    // Get asset parameters
    std::vector<double> S0(num_assets), mu(num_assets), sigma(num_assets);
    
    std::cout << "\n--- Asset Parameters ---\n";
    for (int i = 0; i < num_assets; ++i) {
        std::cout << "\nAsset " << (i+1) << ":\n";
        S0[i] = get_positive_double("  Initial price (S0): $");
        mu[i] = get_double("  Expected return (mu): ");
        sigma[i] = get_positive_double("  Volatility (sigma): ");
    }
    
    // Get correlation matrix
    std::cout << "\n--- Correlation Matrix ---\n";
    std::vector<std::vector<double>> correlation(num_assets, std::vector<double>(num_assets));
    
    // Initialize diagonal
    for (int i = 0; i < num_assets; ++i) {
        correlation[i][i] = 1.0;
    }
    
    // Get off-diagonal elements
    for (int i = 0; i < num_assets; ++i) {
        for (int j = i + 1; j < num_assets; ++j) {
            std::string prompt = "Correlation between Asset " + std::to_string(i+1) + 
                               " and Asset " + std::to_string(j+1) + ": ";
            double corr = get_correlation(prompt);
            correlation[i][j] = corr;
            correlation[j][i] = corr;
        }
    }
    
    // Get general option parameters
    std::cout << "\n--- Option Parameters ---\n";
    double T = get_positive_double("Time to maturity (years): ");
    double r = get_double("Risk-free rate: ");
    
    // Get Monte Carlo parameters
    std::cout << "\n--- Simulation Parameters ---\n";
    int num_simulations = get_positive_int("Number of Monte Carlo simulations (recommended: 100000): ");
    
    // Create model and pricer
    TwoFactorModel model(S0, mu, sigma, correlation);
    MonteCarloMultiAssetPricer pricer(model, num_simulations);
    
    // Display model summary
    std::cout << "\n=== MODEL SUMMARY ===\n";
    std::cout << "Initial prices: [";
    for (size_t i = 0; i < S0.size(); ++i) {
        std::cout << S0[i] << (i < S0.size()-1 ? ", " : "");
    }
    std::cout << "]\n";
    
    std::cout << "Expected returns: [";
    for (size_t i = 0; i < mu.size(); ++i) {
        std::cout << mu[i] << (i < mu.size()-1 ? ", " : "");
    }
    std::cout << "]\n";
    
    std::cout << "Volatilities: [";
    for (size_t i = 0; i < sigma.size(); ++i) {
        std::cout << sigma[i] << (i < sigma.size()-1 ? ", " : "");
    }
    std::cout << "]\n";
    
    std::cout << "Correlation Matrix:\n";
    for (size_t i = 0; i < correlation.size(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < correlation[i].size(); ++j) {
            std::cout << correlation[i][j] << (j < correlation[i].size()-1 ? ", " : "");
        }
        std::cout << "]\n";
    }
    
    std::cout << "Risk-free rate: " << r << "\n";
    std::cout << "Time to maturity: " << T << " years\n";
    std::cout << "Monte Carlo simulations: " << num_simulations << "\n\n";
    
    // Get user's choice and price accordingly
    int choice = get_option_choice();
    
    switch (choice) {
        case 1: {
            // Basket Option
            std::cout << "\n--- BASKET OPTION ---\n";
            double strike = get_positive_double("Strike price: $");
            bool is_call = get_call_put_choice();
            
            std::vector<double> weights(num_assets);
            std::cout << "Enter weights for basket (should sum to 1.0):\n";
            double weight_sum = 0.0;
            for (int i = 0; i < num_assets; ++i) {
                weights[i] = get_positive_double("Weight for Asset " + std::to_string(i+1) + ": ");
                weight_sum += weights[i];
            }
            
            if (std::abs(weight_sum - 1.0) > 0.01) {
                std::cout << "Warning: Weights sum to " << weight_sum << " (not 1.0)\n";
                char normalize;
                std::cout << "Normalize weights? (Y/N): ";
                std::cin >> normalize;
                std::cin.ignore(10000, '\n');
                if (std::toupper(normalize) == 'Y') {
                    for (auto& w : weights) w /= weight_sum;
                }
            }
            
            BasketOption basket(T, r, strike, weights, is_call);
            auto [price, error] = pricer.price_with_confidence(basket);
            
            std::cout << "\n" << basket.get_type() << " Results:\n";
            std::cout << "Price: $" << price << " ± $" << error << "\n";
            break;
        }
        
        case 2: case 3: {
            // Rainbow Option
            bool is_best = (choice == 2);
            std::cout << "\n--- " << (is_best ? "BEST-OF" : "WORST-OF") << " RAINBOW OPTION ---\n";
            double strike = get_positive_double("Strike price: $");
            bool is_call = get_call_put_choice();
            
            RainbowOption rainbow(T, r, strike, is_call, is_best);
            auto [price, error] = pricer.price_with_confidence(rainbow);
            
            std::cout << "\n" << rainbow.get_type() << " Results:\n";
            std::cout << "Price: $" << price << " ± $" << error << "\n";
            break;
        }
        
        case 4: {
            // Exchange Option
            if (num_assets < 2) {
                std::cout << "Exchange option requires at least 2 assets.\n";
                break;
            }
            std::cout << "\n--- EXCHANGE OPTION ---\n";
            std::cout << "Option to exchange Asset 1 for Asset 2\n";
            
            ExchangeOption exchange(T, r);
            auto [price, error] = pricer.price_with_confidence(exchange);
            
            std::cout << "\n" << exchange.get_type() << " Results:\n";
            std::cout << "Price: $" << price << " ± $" << error << "\n";
            break;
        }
        
        case 5: {
            // Spread Option
            if (num_assets < 2) {
                std::cout << "Spread option requires at least 2 assets.\n";
                break;
            }
            std::cout << "\n--- SPREAD OPTION ---\n";
            std::cout << "Spread = Asset 1 - Asset 2\n";
            double strike = get_double("Strike price: $");
            bool is_call = get_call_put_choice();
            
            SpreadOption spread(T, r, strike, is_call);
            auto [price, error] = pricer.price_with_confidence(spread);
            
            std::cout << "\n" << spread.get_type() << " Results:\n";
            std::cout << "Price: $" << price << " ± $" << error << "\n";
            break;
        }
        
        case 6: {
            // Price All Options
            std::cout << "\n=== PRICING ALL OPTION TYPES ===\n";
            
            // Use default parameters for comprehensive analysis
            double default_strike = 0.0;
            for (double s : S0) default_strike += s;
            default_strike /= S0.size(); // Average of spot prices
            
            std::vector<double> equal_weights(num_assets, 1.0/num_assets);
            
            // Basket Call
            BasketOption basket_call(T, r, default_strike, equal_weights, true);
            auto [basket_price, basket_error] = pricer.price_with_confidence(basket_call);
            std::cout << "\n1. Basket Call (Equal weights, Strike=" << default_strike << "):\n";
            std::cout << "   Price: $" << basket_price << " ± $" << basket_error << "\n";
            
            // Rainbow Options
            RainbowOption best_call(T, r, default_strike, true, true);
            RainbowOption worst_call(T, r, default_strike, true, false);
            auto [best_price, best_error] = pricer.price_with_confidence(best_call);
            auto [worst_price, worst_error] = pricer.price_with_confidence(worst_call);
            std::cout << "\n2. Best-of Rainbow Call (Strike=" << default_strike << "):\n";
            std::cout << "   Price: $" << best_price << " ± $" << best_error << "\n";
            std::cout << "\n3. Worst-of Rainbow Call (Strike=" << default_strike << "):\n";
            std::cout << "   Price: $" << worst_price << " ± $" << worst_error << "\n";
            
            if (num_assets >= 2) {
                // Exchange Option
                ExchangeOption exchange(T, r);
                auto [exchange_price, exchange_error] = pricer.price_with_confidence(exchange);
                std::cout << "\n4. Exchange Option (Asset 1 for Asset 2):\n";
                std::cout << "   Price: $" << exchange_price << " ± $" << exchange_error << "\n";
                
                // Spread Option
                SpreadOption spread_call(T, r, 0.0, true);
                auto [spread_price, spread_error] = pricer.price_with_confidence(spread_call);
                std::cout << "\n5. Spread Call (Strike=0):\n";
                std::cout << "   Price: $" << spread_price << " ± $" << spread_error << "\n";
            }
            break;
        }
        
        case 7: {
            // Correlation Sensitivity Analysis
            if (num_assets != 2) {
                std::cout << "\nCorrelation sensitivity analysis is currently implemented for 2 assets only.\n";
                break;
            }
            
            std::cout << "\n=== CORRELATION SENSITIVITY ANALYSIS ===\n";
            double analysis_strike = (S0[0] + S0[1]) / 2.0;
            std::vector<double> equal_weights = {0.5, 0.5};
            
            std::vector<double> test_correlations = {-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8};
            std::cout << "\nBasket Call Price vs Correlation (Strike=" << analysis_strike << "):\n";
            std::cout << "Correlation\tPrice\n";
            std::cout << "-----------\t-----\n";
            
            for (double rho : test_correlations) {
                std::vector<std::vector<double>> corr_matrix = {{1.0, rho}, {rho, 1.0}};
                TwoFactorModel temp_model(S0, mu, sigma, corr_matrix);
                MonteCarloMultiAssetPricer temp_pricer(temp_model, num_simulations/2); // Use fewer sims for speed
                
                BasketOption basket_test(T, r, analysis_strike, equal_weights, true);
                double price = temp_pricer.price_option(basket_test);
                std::cout << std::fixed << std::setprecision(2) << rho << "\t\t$" << std::setprecision(4) << price << std::endl;
            }
            break;
        }
    }
    
    std::cout << "\n=== PRICING COMPLETE ===\n";
}

int main() {
    try {
        run_interactive_pricing();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}