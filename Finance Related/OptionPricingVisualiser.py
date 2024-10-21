import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

class OptionPricingVisualizer:
    def __init__(self, option_pricer):
        self.pricer = option_pricer

    def plot_option_payoff(self):
        S_range = np.linspace(self.pricer.K * 0.5, self.pricer.K * 1.5, 100)
        payoffs = np.maximum(S_range - self.pricer.K, 0) if self.pricer.option_type == 'call' else np.maximum(self.pricer.K - S_range, 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(S_range, payoffs)
        plt.title(f"{self.pricer.option_type.capitalize()} Option Payoff at Expiration")
        plt.xlabel("Stock Price")
        plt.ylabel("Payoff")
        plt.axvline(x=self.pricer.K, color='r', linestyle='--', label='Strike Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_price_vs_stock_price(self):
        S_range = np.linspace(self.pricer.K * 0.5, self.pricer.K * 1.5, 100)
        prices = [self.pricer.black_scholes_merton() for self.pricer.S in S_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(S_range, prices)
        plt.title(f"{self.pricer.option_type.capitalize()} Option Price vs Stock Price")
        plt.xlabel("Stock Price")
        plt.ylabel("Option Price")
        plt.axvline(x=self.pricer.K, color='r', linestyle='--', label='Strike Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_implied_volatility_smile(self):
        K_range = np.linspace(self.pricer.S * 0.8, self.pricer.S * 1.2, 20)
        implied_vols = []
        
        original_K = self.pricer.K
        for K in K_range:
            self.pricer.K = K
            market_price = self.pricer.black_scholes_merton()  # Using BSM as a proxy for market price
            implied_vol = self.calculate_implied_volatility(market_price)
            implied_vols.append(implied_vol)
        self.pricer.K = original_K  # Reset K to original value
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_range / self.pricer.S, implied_vols)
        plt.title("Implied Volatility Smile")
        plt.xlabel("Moneyness (K/S)")
        plt.ylabel("Implied Volatility")
        plt.grid(True)
        plt.show()

    def plot_greeks(self):
        S_range = np.linspace(self.pricer.K * 0.5, self.pricer.K * 1.5, 100)
        deltas, gammas, vegas, thetas = [], [], [], []
        
        for S in S_range:
            self.pricer.S = S
            deltas.append(self.calculate_delta())
            gammas.append(self.calculate_gamma())
            vegas.append(self.calculate_vega())
            thetas.append(self.calculate_theta())
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(S_range, deltas)
        plt.title("Delta")
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(S_range, gammas)
        plt.title("Gamma")
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(S_range, vegas)
        plt.title("Vega")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(S_range, thetas)
        plt.title("Theta")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_monte_carlo_convergence(self, max_simulations=100000, steps=20):
        sim_numbers = np.linspace(1000, max_simulations, steps).astype(int)
        mc_prices = []
        bsm_price = self.pricer.black_scholes_merton()
        
        for n in sim_numbers:
            mc_price = self.pricer.monte_carlo(n)
            mc_prices.append(mc_price)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sim_numbers, mc_prices, label='Monte Carlo')
        plt.axhline(y=bsm_price, color='r', linestyle='--', label='BSM Price')
        plt.title("Monte Carlo Price Convergence")
        plt.xlabel("Number of Simulations")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_implied_volatility(self, market_price, tolerance=1e-5, max_iterations=100):
        low_vol = 1e-5
        high_vol = 10  # Assuming volatility won't be higher than 1000%
        
        for _ in range(max_iterations):
            mid_vol = (low_vol + high_vol) / 2
            original_sigma = self.pricer.sigma
            self.pricer.sigma = mid_vol
            price = self.pricer.black_scholes_merton()
            self.pricer.sigma = original_sigma  # Reset sigma to original value
            
            if abs(price - market_price) < tolerance:
                return mid_vol
            
            if price > market_price:
                high_vol = mid_vol
            else:
                low_vol = mid_vol
        
        raise ValueError("Implied volatility calculation did not converge")

    def calculate_delta(self):
        d1 = (np.log(self.pricer.S / self.pricer.K) + 
              (self.pricer.r + 0.5 * self.pricer.sigma ** 2) * self.pricer.T) / \
             (self.pricer.sigma * np.sqrt(self.pricer.T))
        
        if self.pricer.option_type == 'call':
            return norm.cdf(d1)
        elif self.pricer.option_type == 'put':
            return norm.cdf(d1) - 1
        else:
            raise ValueError("Option type must be 'call' or 'put'")

    def calculate_gamma(self):
        # Placeholder for gamma calculation
        return 0

    def calculate_vega(self):
        # Placeholder for vega calculation
        return 0

    def calculate_theta(self):
        # Placeholder for theta calculation
        return 0

# Example usage
if __name__ == "__main__":
    from OptionPricer import OptionPricer  # Assuming this is your existing OptionPricer class

    S = 100  # Current stock price
    K = 100  # Strike price
    T = 1    # Time to maturity (in years)
    r = 0.05 # Risk-free interest rate
    sigma = 0.2 # Volatility

    pricer = OptionPricer(S, K, T, r, sigma, 'call')
    visualizer = OptionPricingVisualizer(pricer)

    visualizer.plot_option_payoff()
    visualizer.plot_price_vs_stock_price()
    visualizer.plot_implied_volatility_smile()
    #visualizer.plot_greeks()
    visualizer.plot_monte_carlo_convergence()