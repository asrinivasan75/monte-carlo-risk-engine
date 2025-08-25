import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional, Dict, Any, List
from simulate_paths import MultiAssetGBM


class RiskMetrics:
    """Calculate portfolio risk metrics using Monte Carlo simulation."""
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        return -np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def conditional_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Array of portfolio returns
            confidence_level: Confidence level
            
        Returns:
            CVaR value (average of losses beyond VaR)
        """
        var_threshold = -RiskMetrics.value_at_risk(returns, confidence_level)
        tail_losses = returns[returns <= var_threshold]
        return -np.mean(tail_losses) if len(tail_losses) > 0 else 0.0
    
    @staticmethod
    def expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Alias for conditional_var."""
        return RiskMetrics.conditional_var(returns, confidence_level)
    
    @staticmethod
    def parametric_var(mean_return: float, std_return: float, 
                      confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """
        Calculate parametric VaR assuming normal distribution.
        
        Args:
            mean_return: Expected return
            std_return: Standard deviation of returns
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            
        Returns:
            Parametric VaR
        """
        z_score = stats.norm.ppf(1 - confidence_level)
        horizon_mean = mean_return * time_horizon
        horizon_std = std_return * np.sqrt(time_horizon)
        return -(horizon_mean + z_score * horizon_std)
    
    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related statistics.
        
        Args:
            prices: Array of portfolio values/prices
            
        Returns:
            Dictionary with drawdown statistics
        """
        # Calculate cumulative maximum (peak)
        peak = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - peak) / peak
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find the peak before maximum drawdown
        peak_idx = np.argmax(peak[:max_dd_idx + 1])
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_pct': abs(max_dd) * 100,
            'peak_idx': peak_idx,
            'trough_idx': max_dd_idx,
            'drawdown_series': drawdown,
            'peak_value': prices[peak_idx],
            'trough_value': prices[max_dd_idx]
        }
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                    periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                     periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        if downside_std == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)


class PortfolioRiskAnalyzer:
    """Comprehensive portfolio risk analysis using Monte Carlo simulation."""
    
    def __init__(self, assets_data: Dict[str, Any], portfolio_weights: np.ndarray):
        """
        Initialize portfolio risk analyzer.
        
        Args:
            assets_data: Dictionary containing asset parameters
            portfolio_weights: Portfolio weights array
        """
        self.assets_data = assets_data
        self.weights = np.array(portfolio_weights)
        self.n_assets = len(portfolio_weights)
        
        # Validate weights
        if not np.isclose(np.sum(self.weights), 1.0):
            raise ValueError("Portfolio weights must sum to 1.0")
    
    def simulate_portfolio_scenarios(self, n_scenarios: int = 10000, 
                                   time_horizon: int = 252,
                                   random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate portfolio scenarios using Monte Carlo.
        
        Args:
            n_scenarios: Number of Monte Carlo scenarios
            time_horizon: Time horizon in days
            random_state: Random seed
            
        Returns:
            Dictionary with simulation results
        """
        # Extract parameters from assets_data
        S0 = np.array(self.assets_data['S0'])
        mu = np.array(self.assets_data['mu'])
        sigma = np.array(self.assets_data['sigma'])
        correlation_matrix = np.array(self.assets_data['correlation_matrix'])
        
        # Create multi-asset simulator
        dt = 1 / 252  # Daily time step
        T = time_horizon / 252  # Convert to years
        
        multi_gbm = MultiAssetGBM(S0, mu, sigma, correlation_matrix, T, dt)
        
        # Simulate portfolio values
        portfolio_values = multi_gbm.simulate_portfolio_values(
            self.weights, n_scenarios, random_state
        )
        
        # Calculate returns
        initial_value = portfolio_values[:, 0]
        final_value = portfolio_values[:, -1]
        total_returns = (final_value - initial_value) / initial_value
        
        # Calculate daily returns for the entire paths
        daily_returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
        
        return {
            'portfolio_values': portfolio_values,
            'total_returns': total_returns,
            'daily_returns': daily_returns,
            'initial_value': np.mean(initial_value),
            'final_values': final_value,
            'scenarios': n_scenarios,
            'time_horizon': time_horizon
        }
    
    def calculate_risk_metrics(self, simulation_results: Dict[str, Any],
                             confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics from simulation results.
        
        Args:
            simulation_results: Results from simulate_portfolio_scenarios
            confidence_levels: List of confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with all risk metrics
        """
        total_returns = simulation_results['total_returns']
        daily_returns = simulation_results['daily_returns'].flatten()
        portfolio_values = simulation_results['portfolio_values']
        
        risk_metrics = {}
        
        # Calculate VaR and CVaR for different confidence levels
        for conf_level in confidence_levels:
            var_key = f'VaR_{int(conf_level*100)}%'
            cvar_key = f'CVaR_{int(conf_level*100)}%'
            
            # For total returns
            risk_metrics[f'{var_key}_total'] = RiskMetrics.value_at_risk(total_returns, conf_level)
            risk_metrics[f'{cvar_key}_total'] = RiskMetrics.conditional_var(total_returns, conf_level)
            
            # For daily returns (scaled to initial portfolio value)
            initial_value = simulation_results['initial_value']
            daily_var = RiskMetrics.value_at_risk(daily_returns, conf_level) * initial_value
            daily_cvar = RiskMetrics.conditional_var(daily_returns, conf_level) * initial_value
            
            risk_metrics[f'{var_key}_daily'] = daily_var
            risk_metrics[f'{cvar_key}_daily'] = daily_cvar
        
        # Maximum Drawdown analysis
        max_dd_stats = []
        for scenario in range(portfolio_values.shape[0]):
            dd_stats = RiskMetrics.maximum_drawdown(portfolio_values[scenario, :])
            max_dd_stats.append(dd_stats['max_drawdown'])
        
        risk_metrics['avg_max_drawdown'] = np.mean(max_dd_stats)
        risk_metrics['worst_max_drawdown'] = np.max(max_dd_stats)
        risk_metrics['max_drawdown_distribution'] = np.array(max_dd_stats)
        
        # Performance ratios
        risk_metrics['sharpe_ratio'] = RiskMetrics.sharpe_ratio(daily_returns)
        risk_metrics['sortino_ratio'] = RiskMetrics.sortino_ratio(daily_returns)
        
        # Return statistics
        risk_metrics['expected_return'] = np.mean(total_returns)
        risk_metrics['return_volatility'] = np.std(total_returns)
        risk_metrics['daily_volatility'] = np.std(daily_returns)
        risk_metrics['annualized_volatility'] = np.std(daily_returns) * np.sqrt(252)
        
        # Percentile statistics
        risk_metrics['return_percentiles'] = {
            '5%': np.percentile(total_returns, 5),
            '25%': np.percentile(total_returns, 25),
            '50%': np.percentile(total_returns, 50),
            '75%': np.percentile(total_returns, 75),
            '95%': np.percentile(total_returns, 95)
        }
        
        return risk_metrics
    
    def stress_test(self, stress_scenarios: Dict[str, Dict[str, float]],
                   base_scenarios: int = 10000) -> Dict[str, Any]:
        """
        Perform stress testing under different market conditions.
        
        Args:
            stress_scenarios: Dictionary of stress test scenarios
            base_scenarios: Number of Monte Carlo scenarios per stress test
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        # Base case
        base_results = self.simulate_portfolio_scenarios(base_scenarios, random_state=42)
        base_metrics = self.calculate_risk_metrics(base_results)
        stress_results['base_case'] = {
            'simulation': base_results,
            'metrics': base_metrics
        }
        
        # Stress scenarios
        for scenario_name, adjustments in stress_scenarios.items():
            # Create modified assets data
            stressed_data = self.assets_data.copy()
            
            if 'volatility_multiplier' in adjustments:
                stressed_data['sigma'] = np.array(stressed_data['sigma']) * adjustments['volatility_multiplier']
            
            if 'return_adjustment' in adjustments:
                stressed_data['mu'] = np.array(stressed_data['mu']) + adjustments['return_adjustment']
            
            if 'correlation_multiplier' in adjustments:
                # Adjust off-diagonal correlations
                corr_matrix = np.array(stressed_data['correlation_matrix'])
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                corr_matrix[mask] *= adjustments['correlation_multiplier']
                # Ensure positive definite
                eigenvals = np.linalg.eigvals(corr_matrix)
                if np.min(eigenvals) < 0:
                    corr_matrix += np.eye(corr_matrix.shape[0]) * abs(np.min(eigenvals)) * 1.1
                stressed_data['correlation_matrix'] = corr_matrix
            
            # Create temporary analyzer for stressed scenario
            temp_analyzer = PortfolioRiskAnalyzer(stressed_data, self.weights)
            stressed_sim = temp_analyzer.simulate_portfolio_scenarios(base_scenarios, random_state=42)
            stressed_metrics = temp_analyzer.calculate_risk_metrics(stressed_sim)
            
            stress_results[scenario_name] = {
                'simulation': stressed_sim,
                'metrics': stressed_metrics,
                'adjustments': adjustments
            }
        
        return stress_results


def create_sample_portfolio() -> Tuple[Dict[str, Any], np.ndarray]:
    """Create sample portfolio for demonstration."""
    
    # Sample 3-asset portfolio: Large Cap Stock, Small Cap Stock, Bond
    assets_data = {
        'names': ['Large Cap', 'Small Cap', 'Bond'],
        'S0': [100, 50, 1000],  # Initial prices
        'mu': [0.08, 0.12, 0.03],  # Expected returns
        'sigma': [0.15, 0.25, 0.05],  # Volatilities
        'correlation_matrix': [
            [1.0, 0.7, -0.1],
            [0.7, 1.0, -0.05],
            [-0.1, -0.05, 1.0]
        ]
    }
    
    # Portfolio weights: 50% large cap, 30% small cap, 20% bonds
    weights = np.array([0.5, 0.3, 0.2])
    
    return assets_data, weights


if __name__ == "__main__":
    # Example usage
    print("=== Portfolio Risk Analysis Example ===\n")
    
    # Create sample portfolio
    assets_data, weights = create_sample_portfolio()
    
    print("Portfolio Composition:")
    for i, (name, weight) in enumerate(zip(assets_data['names'], weights)):
        print(f"  {name}: {weight:.1%}")
    print()
    
    # Initialize analyzer
    analyzer = PortfolioRiskAnalyzer(assets_data, weights)
    
    # Run simulation
    print("Running Monte Carlo simulation...")
    simulation = analyzer.simulate_portfolio_scenarios(n_scenarios=50000, time_horizon=252, random_state=42)
    
    # Calculate risk metrics
    metrics = analyzer.calculate_risk_metrics(simulation)
    
    print("\n=== Risk Metrics (1-Year Horizon) ===")
    print(f"Expected Return: {metrics['expected_return']:.2%}")
    print(f"Return Volatility: {metrics['return_volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    
    print(f"\nValue at Risk (95%): {metrics['VaR_95%_total']:.2%}")
    print(f"Conditional VaR (95%): {metrics['CVaR_95%_total']:.2%}")
    print(f"Value at Risk (99%): {metrics['VaR_99%_total']:.2%}")
    print(f"Conditional VaR (99%): {metrics['CVaR_99%_total']:.2%}")
    
    print(f"\nAverage Max Drawdown: {metrics['avg_max_drawdown']:.2%}")
    print(f"Worst Max Drawdown: {metrics['worst_max_drawdown']:.2%}")
    
    # Stress testing
    print("\n=== Stress Testing ===")
    stress_scenarios = {
        'market_crash': {
            'return_adjustment': -0.10,  # 10% lower returns
            'volatility_multiplier': 2.0,  # Double volatility
            'correlation_multiplier': 1.5   # Higher correlations
        },
        'low_vol_environment': {
            'volatility_multiplier': 0.5,   # Half volatility
            'correlation_multiplier': 0.8   # Lower correlations
        }
    }
    
    stress_results = analyzer.stress_test(stress_scenarios, base_scenarios=10000)
    
    for scenario_name, results in stress_results.items():
        if scenario_name != 'base_case':
            print(f"\n{scenario_name.replace('_', ' ').title()}:")
            metrics = results['metrics']
            print(f"  Expected Return: {metrics['expected_return']:.2%}")
            print(f"  VaR (95%): {metrics['VaR_95%_total']:.2%}")
            print(f"  CVaR (95%): {metrics['CVaR_95%_total']:.2%}")