import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional, Tuple
from simulate_paths import GBMSimulator, MultiAssetGBM
from option_pricing import MonteCarloOptionPricer, BlackScholes, convergence_analysis
from risk_metrics import PortfolioRiskAnalyzer, RiskMetrics


class FinancialVisualizer:
    """Create comprehensive financial visualizations for Monte Carlo analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer with specified style."""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 10)
    
    def plot_simulated_paths(self, paths: np.ndarray, times: np.ndarray, 
                           n_paths_display: int = 100, title: str = "Simulated Stock Price Paths",
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot simulated stock price paths.
        
        Args:
            paths: Array of simulated paths (n_paths, n_steps)
            times: Time array
            n_paths_display: Number of paths to display
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot subset of paths
        n_display = min(n_paths_display, paths.shape[0])
        indices = np.random.choice(paths.shape[0], n_display, replace=False)
        
        for i in indices:
            ax.plot(times, paths[i, :], alpha=0.3, linewidth=0.5, color=self.colors[0])
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        ax.plot(times, mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot percentiles
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        ax.fill_between(times, p5, p95, alpha=0.2, color='gray', label='5th-95th Percentile')
        
        ax.set_xlabel('Time (Years)')
        ax.set_ylabel('Stock Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_payoff_distribution(self, payoffs: np.ndarray, option_type: str,
                               strike: float, spot: float, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot option payoff distribution.
        
        Args:
            payoffs: Array of option payoffs
            option_type: 'call' or 'put'
            strike: Strike price
            spot: Current spot price
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax1.hist(payoffs, bins=50, alpha=0.7, density=True, color=self.colors[1])
        ax1.axvline(np.mean(payoffs), color='red', linestyle='--', 
                   label=f'Mean Payoff: ${np.mean(payoffs):.2f}')
        ax1.set_xlabel('Payoff ($)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{option_type.capitalize()} Option Payoff Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(payoffs, vert=True)
        ax2.set_ylabel('Payoff ($)')
        ax2.set_title('Payoff Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: ${np.mean(payoffs):.2f}
Std: ${np.std(payoffs):.2f}
Max: ${np.max(payoffs):.2f}
Min: ${np.min(payoffs):.2f}
Prob(ITM): {np.mean(payoffs > 0):.2%}"""
        
        ax2.text(1.1, np.median(payoffs), stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, convergence_data: Dict[str, Any], 
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot Monte Carlo convergence analysis.
        
        Args:
            convergence_data: Data from convergence_analysis function
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        n_sims = convergence_data['n_simulations']
        mc_prices = convergence_data['mc_prices']
        std_errors = convergence_data['std_errors']
        bs_price = convergence_data['bs_price']
        
        # Price convergence
        ax1.plot(n_sims, mc_prices, label='Monte Carlo Price', color=self.colors[0])
        ax1.axhline(bs_price, color='red', linestyle='--', label='Black-Scholes Price')
        ax1.fill_between(n_sims, mc_prices - 1.96*std_errors, mc_prices + 1.96*std_errors,
                        alpha=0.2, label='95% Confidence Interval')
        ax1.set_xlabel('Number of Simulations')
        ax1.set_ylabel('Option Price')
        ax1.set_title('Price Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Standard error
        ax2.loglog(n_sims, std_errors, label='Standard Error', color=self.colors[2])
        ax2.loglog(n_sims, 1/np.sqrt(n_sims) * std_errors[0] * np.sqrt(n_sims[0]),
                  '--', label=r'$1/\sqrt{n}$ Reference', color='red')
        ax2.set_xlabel('Number of Simulations')
        ax2.set_ylabel('Standard Error')
        ax2.set_title('Standard Error Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Absolute error
        abs_errors = np.abs(mc_prices - bs_price)
        ax3.loglog(n_sims, abs_errors, label='Absolute Error', color=self.colors[3])
        ax3.set_xlabel('Number of Simulations')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Absolute Error vs Black-Scholes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Relative error
        rel_errors = abs_errors / bs_price * 100
        ax4.semilogx(n_sims, rel_errors, label='Relative Error (%)', color=self.colors[4])
        ax4.set_xlabel('Number of Simulations')
        ax4.set_ylabel('Relative Error (%)')
        ax4.set_title('Relative Error vs Black-Scholes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics(self, risk_metrics: Dict[str, Any], 
                         portfolio_returns: np.ndarray,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot comprehensive risk metrics visualization.
        
        Args:
            risk_metrics: Dictionary of calculated risk metrics
            portfolio_returns: Array of portfolio returns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])  # Return distribution
        ax2 = fig.add_subplot(gs[0, 2])   # VaR/CVaR
        ax3 = fig.add_subplot(gs[1, :])   # Risk metrics comparison
        ax4 = fig.add_subplot(gs[2, 0])   # Drawdown distribution
        ax5 = fig.add_subplot(gs[2, 1])   # Performance ratios
        ax6 = fig.add_subplot(gs[2, 2])   # Return percentiles
        
        # 1. Return distribution with VaR markers
        ax1.hist(portfolio_returns, bins=50, alpha=0.7, density=True, color=self.colors[0])
        
        var_95 = risk_metrics.get('VaR_95%_total', 0)
        var_99 = risk_metrics.get('VaR_99%_total', 0)
        
        ax1.axvline(-var_95, color='orange', linestyle='--', label=f'VaR 95%: {var_95:.2%}')
        ax1.axvline(-var_99, color='red', linestyle='--', label=f'VaR 99%: {var_99:.2%}')
        ax1.axvline(np.mean(portfolio_returns), color='green', linestyle='--', 
                   label=f'Mean: {np.mean(portfolio_returns):.2%}')
        
        ax1.set_xlabel('Portfolio Returns')
        ax1.set_ylabel('Density')
        ax1.set_title('Portfolio Return Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. VaR/CVaR comparison
        var_cvar_data = {
            'VaR 95%': var_95,
            'CVaR 95%': risk_metrics.get('CVaR_95%_total', 0),
            'VaR 99%': var_99,
            'CVaR 99%': risk_metrics.get('CVaR_99%_total', 0)
        }
        
        bars = ax2.bar(range(len(var_cvar_data)), list(var_cvar_data.values()),
                      color=[self.colors[i] for i in range(4)])
        ax2.set_xticks(range(len(var_cvar_data)))
        ax2.set_xticklabels(list(var_cvar_data.keys()), rotation=45)
        ax2.set_ylabel('Risk Measure')
        ax2.set_title('VaR vs CVaR')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, var_cvar_data.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.2%}', ha='center', va='bottom', fontsize=9)
        
        # 3. Risk metrics comparison table
        ax3.axis('off')
        metrics_table = [
            ['Expected Return', f"{risk_metrics.get('expected_return', 0):.2%}"],
            ['Volatility', f"{risk_metrics.get('return_volatility', 0):.2%}"],
            ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.3f}"],
            ['Sortino Ratio', f"{risk_metrics.get('sortino_ratio', 0):.3f}"],
            ['Avg Max Drawdown', f"{risk_metrics.get('avg_max_drawdown', 0):.2%}"],
            ['Worst Max Drawdown', f"{risk_metrics.get('worst_max_drawdown', 0):.2%}"]
        ]
        
        table = ax3.table(cellText=metrics_table,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax3.set_title('Portfolio Risk Summary')
        
        # 4. Maximum Drawdown distribution
        if 'max_drawdown_distribution' in risk_metrics:
            dd_dist = risk_metrics['max_drawdown_distribution']
            ax4.hist(dd_dist, bins=30, alpha=0.7, color=self.colors[5])
            ax4.axvline(np.mean(dd_dist), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(dd_dist):.2%}')
            ax4.set_xlabel('Maximum Drawdown')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Max Drawdown Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Performance ratios
        ratios = {
            'Sharpe': risk_metrics.get('sharpe_ratio', 0),
            'Sortino': risk_metrics.get('sortino_ratio', 0)
        }
        
        bars = ax5.bar(range(len(ratios)), list(ratios.values()),
                      color=self.colors[6:8])
        ax5.set_xticks(range(len(ratios)))
        ax5.set_xticklabels(list(ratios.keys()))
        ax5.set_ylabel('Ratio')
        ax5.set_title('Performance Ratios')
        ax5.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, ratios.values()):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 6. Return percentiles
        if 'return_percentiles' in risk_metrics:
            percentiles = risk_metrics['return_percentiles']
            ax6.bar(range(len(percentiles)), list(percentiles.values()),
                   color=self.colors[8:])
            ax6.set_xticks(range(len(percentiles)))
            ax6.set_xticklabels(list(percentiles.keys()))
            ax6.set_ylabel('Return')
            ax6.set_title('Return Percentiles')
            ax6.grid(True, alpha=0.3)
            
            # Format y-axis as percentage
            ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix: np.ndarray, 
                               asset_names: List[str],
                               figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_matrix: Asset correlation matrix
            asset_names: List of asset names
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(correlation_matrix, 
                   xticklabels=asset_names,
                   yticklabels=asset_names,
                   annot=True, 
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Asset Correlation Matrix')
        return fig
    
    def create_interactive_paths_plot(self, paths: np.ndarray, times: np.ndarray,
                                    title: str = "Interactive Stock Price Paths") -> go.Figure:
        """
        Create interactive stock price paths plot using Plotly.
        
        Args:
            paths: Array of simulated paths
            times: Time array
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add sample paths
        n_display = min(100, paths.shape[0])
        indices = np.random.choice(paths.shape[0], n_display, replace=False)
        
        for i, idx in enumerate(indices):
            fig.add_trace(go.Scatter(
                x=times,
                y=paths[idx, :],
                mode='lines',
                line=dict(width=1, color=f'rgba(100,100,100,0.1)'),
                showlegend=False,
                hovertemplate='Time: %{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(
            x=times,
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Mean Path',
            hovertemplate='Time: %{x:.2f}<br>Mean Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add percentile bands
        p5 = np.percentile(paths, 5, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill='toself',
            fillcolor='rgba(128,128,128,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='5th-95th Percentile',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (Years)',
            yaxis_title='Stock Price ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_interactive_risk_dashboard(self, risk_metrics: Dict[str, Any],
                                        portfolio_returns: np.ndarray) -> go.Figure:
        """
        Create interactive risk metrics dashboard.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            portfolio_returns: Array of portfolio returns
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Return Distribution', 'VaR/CVaR Comparison', 'Performance Metrics',
                          'Drawdown Distribution', 'Return Percentiles', 'Risk Summary'),
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}, {"type": "table"}]]
        )
        
        # 1. Return distribution
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            nbinsx=50,
            name='Returns',
            opacity=0.7
        ), row=1, col=1)
        
        # Add VaR lines
        var_95 = risk_metrics.get('VaR_95%_total', 0)
        var_99 = risk_metrics.get('VaR_99%_total', 0)
        
        fig.add_vline(x=-var_95, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_vline(x=-var_99, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. VaR/CVaR comparison
        var_cvar_labels = ['VaR 95%', 'CVaR 95%', 'VaR 99%', 'CVaR 99%']
        var_cvar_values = [
            var_95,
            risk_metrics.get('CVaR_95%_total', 0),
            var_99,
            risk_metrics.get('CVaR_99%_total', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=var_cvar_labels,
            y=var_cvar_values,
            name='Risk Measures'
        ), row=1, col=2)
        
        # 3. Performance metrics
        perf_metrics = ['Sharpe Ratio', 'Sortino Ratio']
        perf_values = [
            risk_metrics.get('sharpe_ratio', 0),
            risk_metrics.get('sortino_ratio', 0)
        ]
        
        fig.add_trace(go.Bar(
            x=perf_metrics,
            y=perf_values,
            name='Performance'
        ), row=1, col=3)
        
        # 4. Drawdown distribution
        if 'max_drawdown_distribution' in risk_metrics:
            fig.add_trace(go.Histogram(
                x=risk_metrics['max_drawdown_distribution'],
                nbinsx=30,
                name='Max Drawdown'
            ), row=2, col=1)
        
        # 5. Return percentiles
        if 'return_percentiles' in risk_metrics:
            percentiles = risk_metrics['return_percentiles']
            fig.add_trace(go.Bar(
                x=list(percentiles.keys()),
                y=list(percentiles.values()),
                name='Percentiles'
            ), row=2, col=2)
        
        # 6. Risk summary table
        summary_data = [
            ['Expected Return', f"{risk_metrics.get('expected_return', 0):.2%}"],
            ['Volatility', f"{risk_metrics.get('return_volatility', 0):.2%}"],
            ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.3f}"],
            ['VaR 95%', f"{var_95:.2%}"],
            ['CVaR 95%', f"{risk_metrics.get('CVaR_95%_total', 0):.2%}"],
            ['Max Drawdown', f"{risk_metrics.get('avg_max_drawdown', 0):.2%}"]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'Value']),
            cells=dict(values=list(zip(*summary_data)))
        ), row=2, col=3)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Portfolio Risk Dashboard"
        )
        
        return fig


if __name__ == "__main__":
    # Example usage
    print("Creating example visualizations...")
    
    # Initialize visualizer
    viz = FinancialVisualizer()
    
    # 1. Simulate some paths for demonstration
    from simulate_paths import generate_sample_data
    params = generate_sample_data()['single_asset']
    gbm = GBMSimulator(**params)
    paths = gbm.simulate_path(n_paths=1000, random_state=42)
    
    # Plot simulated paths
    fig1 = viz.plot_simulated_paths(paths, gbm.times, title="Example GBM Paths")
    plt.savefig('/Users/aadithyasrinivasan/Projects/monte-carlo-risk-engine/example_paths.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Option payoff visualization
    from option_pricing import MonteCarloOptionPricer
    pricer = MonteCarloOptionPricer(100, 0.05, 0.2, 1.0)
    call_result = pricer.price_european_option(105, 'call', 10000, 42)
    
    fig2 = viz.plot_payoff_distribution(call_result['payoffs'], 'call', 105, 100)
    plt.savefig('/Users/aadithyasrinivasan/Projects/monte-carlo-risk-engine/example_payoffs.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Example visualizations saved as PNG files.")
    print("Visualization module ready for use with Streamlit dashboard.")