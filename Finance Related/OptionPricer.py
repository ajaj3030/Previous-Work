import numpy as np
from scipy.stats import norm
import argparse
class OptionPricer:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.option_type = option_type.lower()

    def black_scholes_merton(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        if self.option_type == 'call':
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return price

    def monte_carlo(self, num_simulations=2000000):
        # Generate random normally distributed numbers
        Z = np.random.standard_normal(num_simulations)
        
        # Calculate stock price at maturity
        S_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)
        
        # Calculate payoff
        if self.option_type == 'call':
            payoff = np.maximum(S_T - self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K - S_T, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'")
        
        # Calculate option price (discounted expected payoff)
        option_price = np.exp(-self.r * self.T) * np.mean(payoff)
        
        return option_price

def compare_methods(S, K, T, r, sigma, option_type, num_runs=100, num_simulations=100000):
    pricer = OptionPricer(S, K, T, r, sigma, option_type)
    bsm_price = pricer.black_scholes_merton()
    mc_prices = [pricer.monte_carlo(num_simulations) for _ in range(num_runs)]
    
    mc_mean = np.mean(mc_prices)
    mc_std = np.std(mc_prices)
    
    print(f"Black-Scholes-Merton price: {bsm_price:.4f}")
    print(f"Monte Carlo mean price: {mc_mean:.4f}")
    print(f"Monte Carlo std dev: {mc_std:.4f}")
    print(f"Difference (BSM - MC mean): {bsm_price - mc_mean:.4f}")

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Option Pricer")
    parser.add_argument("--S", type=float, default=100, help="Current stock price")
    parser.add_argument("--K", type=float, default=100, help="Strike price")
    parser.add_argument("--T", type=float, default=1, help="Time to maturity (in years)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free interest rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--option_type", type=str, choices=['call', 'put'], default='call', help="Option type (call or put)")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of Monte Carlo runs")
    parser.add_argument("--num_simulations", type=int, default=100000, help="Number of Monte Carlo simulations per run")

    args = parser.parse_args()

    print(f"{args.option_type.capitalize()} Option:")
    compare_methods(args.S, args.K, args.T, args.r, args.sigma, args.option_type, args.num_runs, args.num_simulations)