import numpy as np
import pandas as pd
from typing import Tuple, Optional


class GBMSimulator:
    """Geometric Brownian Motion simulator for stock price paths."""
    
    def __init__(self, S0: float, mu: float, sigma: float, T: float, dt: float):
        """
        Initialize GBM parameters.
        
        Args:
            S0: Initial stock price
            mu: Drift rate (expected return)
            sigma: Volatility
            T: Time to maturity
            dt: Time step
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)
        self.times = np.linspace(0, T, self.n_steps + 1)
    
    def simulate_path(self, n_paths: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate stock price paths using GBM.
        
        Args:
            n_paths: Number of simulation paths
            random_state: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps+1) containing simulated paths
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(self.dt), (n_paths, self.n_steps))
        
        # Initialize price matrix
        S = np.zeros((n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        
        # Simulate paths using exact solution of GBM SDE
        for i in range(self.n_steps):
            S[:, i + 1] = S[:, i] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW[:, i]
            )
        
        return S
    
    def simulate_terminal_prices(self, n_simulations: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Efficiently simulate only terminal stock prices (for option pricing).
        
        Args:
            n_simulations: Number of price simulations
            random_state: Random seed for reproducibility
            
        Returns:
            Array of terminal stock prices
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Use exact GBM formula for terminal price
        Z = np.random.normal(0, 1, n_simulations)
        ST = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z
        )
        
        return ST


class MultiAssetGBM:
    """Multi-asset GBM simulator for portfolio analysis."""
    
    def __init__(self, S0: np.ndarray, mu: np.ndarray, sigma: np.ndarray, 
                 correlation_matrix: np.ndarray, T: float, dt: float):
        """
        Initialize multi-asset GBM parameters.
        
        Args:
            S0: Initial prices array
            mu: Drift rates array
            sigma: Volatilities array
            correlation_matrix: Asset correlation matrix
            T: Time to maturity
            dt: Time step
        """
        self.S0 = np.array(S0)
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        self.correlation_matrix = np.array(correlation_matrix)
        self.T = T
        self.dt = dt
        self.n_assets = len(S0)
        self.n_steps = int(T / dt)
        self.times = np.linspace(0, T, self.n_steps + 1)
        
        # Compute Cholesky decomposition for correlated random numbers
        self.chol_matrix = np.linalg.cholesky(correlation_matrix)
    
    def simulate_paths(self, n_paths: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate correlated multi-asset paths.
        
        Args:
            n_paths: Number of simulation paths
            random_state: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_assets, n_steps+1) containing simulated paths
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate independent random numbers
        dW_indep = np.random.normal(0, np.sqrt(self.dt), 
                                   (n_paths, self.n_assets, self.n_steps))
        
        # Apply correlation structure
        dW_corr = np.zeros_like(dW_indep)
        for path in range(n_paths):
            for step in range(self.n_steps):
                dW_corr[path, :, step] = self.chol_matrix @ dW_indep[path, :, step]
        
        # Initialize price tensor
        S = np.zeros((n_paths, self.n_assets, self.n_steps + 1))
        S[:, :, 0] = self.S0[np.newaxis, :]
        
        # Simulate paths
        for i in range(self.n_steps):
            for asset in range(self.n_assets):
                S[:, asset, i + 1] = S[:, asset, i] * np.exp(
                    (self.mu[asset] - 0.5 * self.sigma[asset]**2) * self.dt + 
                    self.sigma[asset] * dW_corr[:, asset, i]
                )
        
        return S
    
    def simulate_portfolio_values(self, weights: np.ndarray, n_paths: int = 1000, 
                                 random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate portfolio value paths given asset weights.
        
        Args:
            weights: Portfolio weights array
            n_paths: Number of simulation paths
            random_state: Random seed for reproducibility
            
        Returns:
            Array of portfolio values over time
        """
        asset_paths = self.simulate_paths(n_paths, random_state)
        
        # Calculate portfolio values
        portfolio_values = np.zeros((n_paths, self.n_steps + 1))
        initial_portfolio_value = np.sum(weights * self.S0)
        
        for path in range(n_paths):
            for step in range(self.n_steps + 1):
                portfolio_values[path, step] = np.sum(weights * asset_paths[path, :, step])
        
        return portfolio_values


def generate_sample_data() -> dict:
    """Generate sample market data for testing."""
    return {
        'single_asset': {
            'S0': 100.0,      # Current stock price
            'mu': 0.05,       # Expected return (5% annually)
            'sigma': 0.2,     # Volatility (20% annually)
            'T': 1.0,         # 1 year
            'dt': 1/252       # Daily steps
        },
        'multi_asset': {
            'S0': [100, 120, 80],           # Initial prices
            'mu': [0.05, 0.07, 0.04],       # Expected returns
            'sigma': [0.2, 0.25, 0.18],     # Volatilities
            'correlation_matrix': [
                [1.0, 0.3, 0.1],
                [0.3, 1.0, 0.2],
                [0.1, 0.2, 1.0]
            ],
            'T': 1.0,
            'dt': 1/252
        }
    }


if __name__ == "__main__":
    # Example usage
    params = generate_sample_data()['single_asset']
    
    # Single asset simulation
    gbm = GBMSimulator(**params)
    paths = gbm.simulate_path(n_paths=1000, random_state=42)
    
    print(f"Simulated {paths.shape[0]} paths with {paths.shape[1]} time steps")
    print(f"Initial price: ${params['S0']:.2f}")
    print(f"Final price range: ${paths[:, -1].min():.2f} - ${paths[:, -1].max():.2f}")
    print(f"Average final price: ${paths[:, -1].mean():.2f}")