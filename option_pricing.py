import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Tuple, Optional, Dict, Any
from simulate_paths import GBMSimulator


class BlackScholes:
    """Black-Scholes analytical option pricing."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @classmethod
    def greeks(cls, S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, Tuple[float, float]]:
        """Calculate option Greeks for call and put options."""
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(S, K, T, r, sigma)
        
        sqrt_T = np.sqrt(T)
        exp_neg_rT = np.exp(-r * T)
        
        # Delta
        call_delta = norm.cdf(d1)
        put_delta = call_delta - 1
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
        
        # Theta
        common_theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_T)
        call_theta = common_theta - r * K * exp_neg_rT * norm.cdf(d2)
        put_theta = common_theta + r * K * exp_neg_rT * norm.cdf(-d2)
        
        # Vega (same for call and put)
        vega = S * norm.pdf(d1) * sqrt_T
        
        # Rho
        call_rho = K * T * exp_neg_rT * norm.cdf(d2)
        put_rho = -K * T * exp_neg_rT * norm.cdf(-d2)
        
        return {
            'delta': (call_delta, put_delta),
            'gamma': (gamma, gamma),
            'theta': (call_theta / 365, put_theta / 365),  # Daily theta
            'vega': (vega / 100, vega / 100),  # Vega per 1% vol change
            'rho': (call_rho / 100, put_rho / 100)  # Rho per 1% rate change
        }


class MonteCarloOptionPricer:
    """Monte Carlo option pricing engine."""
    
    def __init__(self, S0: float, r: float, sigma: float, T: float):
        """
        Initialize option pricing parameters.
        
        Args:
            S0: Current stock price
            r: Risk-free rate
            sigma: Volatility
            T: Time to expiration
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        
        # Use risk-neutral drift for option pricing
        self.simulator = GBMSimulator(S0, r, sigma, T, T/100)
    
    def price_european_option(self, K: float, option_type: str, n_simulations: int = 100000,
                            random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Price European call/put option using Monte Carlo.
        
        Args:
            K: Strike price
            option_type: 'call' or 'put'
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing price, standard error, and confidence interval
        """
        # Simulate terminal stock prices
        ST = self.simulator.simulate_terminal_prices(n_simulations, random_state)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        z_score = 1.96  # 95% confidence
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': (ci_lower, ci_upper),
            'payoffs': payoffs,
            'terminal_prices': ST
        }
    
    def price_asian_option(self, K: float, option_type: str, n_simulations: int = 100000,
                          n_steps: int = 100, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Price Asian (arithmetic average) option using Monte Carlo.
        
        Args:
            K: Strike price
            option_type: 'call' or 'put'
            n_simulations: Number of Monte Carlo simulations
            n_steps: Number of time steps for averaging
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing price and statistics
        """
        dt = self.T / n_steps
        gbm = GBMSimulator(self.S0, self.r, self.sigma, self.T, dt)
        
        # Simulate paths
        paths = gbm.simulate_path(n_simulations, random_state)
        
        # Calculate arithmetic average for each path
        avg_prices = np.mean(paths, axis=1)
        
        # Calculate payoffs based on average price
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - avg_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': payoffs,
            'average_prices': avg_prices,
            'paths': paths
        }
    
    def price_barrier_option(self, K: float, barrier: float, option_type: str, 
                           barrier_type: str, n_simulations: int = 100000,
                           n_steps: int = 100, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Price barrier option using Monte Carlo.
        
        Args:
            K: Strike price
            barrier: Barrier level
            option_type: 'call' or 'put'
            barrier_type: 'knock_in', 'knock_out'
            n_simulations: Number of simulations
            n_steps: Number of time steps to check barrier
            random_state: Random seed
            
        Returns:
            Dictionary containing price and statistics
        """
        dt = self.T / n_steps
        gbm = GBMSimulator(self.S0, self.r, self.sigma, self.T, dt)
        
        # Simulate paths
        paths = gbm.simulate_path(n_simulations, random_state)
        
        # Check barrier conditions
        if barrier_type == 'knock_out':
            # Option becomes worthless if barrier is crossed
            barrier_crossed = np.any(paths >= barrier, axis=1) if barrier > self.S0 else np.any(paths <= barrier, axis=1)
            active_options = ~barrier_crossed
        elif barrier_type == 'knock_in':
            # Option becomes active only if barrier is crossed
            barrier_crossed = np.any(paths >= barrier, axis=1) if barrier > self.S0 else np.any(paths <= barrier, axis=1)
            active_options = barrier_crossed
        else:
            raise ValueError("barrier_type must be 'knock_in' or 'knock_out'")
        
        # Calculate payoffs for active options
        terminal_prices = paths[:, -1]
        if option_type.lower() == 'call':
            payoffs = np.maximum(terminal_prices - K, 0) * active_options
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - terminal_prices, 0) * active_options
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': payoffs,
            'active_fraction': np.mean(active_options),
            'barrier_crossed': barrier_crossed,
            'paths': paths
        }


def compare_mc_bs_pricing(S0: float, K: float, T: float, r: float, sigma: float,
                         n_simulations: int = 100000) -> pd.DataFrame:
    """
    Compare Monte Carlo and Black-Scholes option prices.
    
    Args:
        S0: Current stock price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        n_simulations: Number of MC simulations
        
    Returns:
        DataFrame comparing prices and differences
    """
    # Black-Scholes prices
    bs_call = BlackScholes.call_price(S0, K, T, r, sigma)
    bs_put = BlackScholes.put_price(S0, K, T, r, sigma)
    
    # Monte Carlo prices
    mc_pricer = MonteCarloOptionPricer(S0, r, sigma, T)
    mc_call = mc_pricer.price_european_option(K, 'call', n_simulations, random_state=42)
    mc_put = mc_pricer.price_european_option(K, 'put', n_simulations, random_state=42)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Option Type': ['Call', 'Put'],
        'Black-Scholes': [bs_call, bs_put],
        'Monte Carlo': [mc_call['price'], mc_put['price']],
        'MC Std Error': [mc_call['std_error'], mc_put['std_error']],
        'Absolute Difference': [abs(bs_call - mc_call['price']), abs(bs_put - mc_put['price'])],
        'Relative Difference (%)': [
            abs(bs_call - mc_call['price']) / bs_call * 100,
            abs(bs_put - mc_put['price']) / bs_put * 100
        ]
    })
    
    return comparison


def convergence_analysis(S0: float, K: float, T: float, r: float, sigma: float,
                        max_simulations: int = 1000000, step_size: int = 10000) -> Dict[str, Any]:
    """
    Analyze Monte Carlo convergence for option pricing.
    
    Args:
        S0: Current stock price
        K: Strike price  
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        max_simulations: Maximum number of simulations
        step_size: Step size for convergence analysis
        
    Returns:
        Dictionary with convergence data
    """
    mc_pricer = MonteCarloOptionPricer(S0, r, sigma, T)
    bs_call = BlackScholes.call_price(S0, K, T, r, sigma)
    
    n_sims = np.arange(step_size, max_simulations + 1, step_size)
    call_prices = []
    std_errors = []
    
    # Use same random seed for consistency
    np.random.seed(42)
    ST = mc_pricer.simulator.simulate_terminal_prices(max_simulations)
    
    for n in n_sims:
        # Use first n simulations
        payoffs = np.maximum(ST[:n] - K, 0)
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n)
        
        call_prices.append(price)
        std_errors.append(std_error)
    
    return {
        'n_simulations': n_sims,
        'mc_prices': np.array(call_prices),
        'std_errors': np.array(std_errors),
        'bs_price': bs_call,
        'final_error': abs(call_prices[-1] - bs_call)
    }


if __name__ == "__main__":
    # Example usage and validation
    S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
    
    print("=== Black-Scholes vs Monte Carlo Comparison ===")
    comparison = compare_mc_bs_pricing(S0, K, T, r, sigma, 100000)
    print(comparison.round(6))
    
    print("\n=== Option Greeks ===")
    greeks = BlackScholes.greeks(S0, K, T, r, sigma)
    for greek, (call_val, put_val) in greeks.items():
        print(f"{greek.capitalize()}: Call = {call_val:.4f}, Put = {put_val:.4f}")
    
    print("\n=== Convergence Analysis ===")
    conv_data = convergence_analysis(S0, K, T, r, sigma, 100000, 5000)
    print(f"Final MC Price: {conv_data['mc_prices'][-1]:.6f}")
    print(f"BS Price: {conv_data['bs_price']:.6f}")
    print(f"Final Error: {conv_data['final_error']:.6f}")
    print(f"Final Std Error: {conv_data['std_errors'][-1]:.6f}")