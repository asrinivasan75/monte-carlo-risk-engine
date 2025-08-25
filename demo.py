#!/usr/bin/env python3
"""
Monte Carlo Financial Risk Engine - Comprehensive Demo

This script demonstrates all key features of the Monte Carlo simulation engine.
Run this script to see examples of stock simulation, option pricing, and portfolio risk analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from simulate_paths import GBMSimulator, MultiAssetGBM, generate_sample_data
from option_pricing import MonteCarloOptionPricer, BlackScholes, compare_mc_bs_pricing, convergence_analysis
from risk_metrics import PortfolioRiskAnalyzer, RiskMetrics, create_sample_portfolio
from visualizations import FinancialVisualizer

def demo_stock_simulation():
    """Demonstrate stock price simulation using GBM."""
    print("="*60)
    print("1. STOCK PRICE SIMULATION DEMO")
    print("="*60)
    
    # Parameters
    S0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    
    # Create simulator
    gbm = GBMSimulator(S0, mu, sigma, T, 1/252)
    
    # Simulate paths
    paths = gbm.simulate_path(1000, random_state=42)
    
    print(f"Simulated 1,000 stock price paths")
    print(f"Parameters: S0=${S0}, μ={mu:.1%}, σ={sigma:.1%}, T={T}yr")
    print(f"\nFinal Price Statistics:")
    print(f"  Mean: ${paths[:, -1].mean():.2f}")
    print(f"  Std:  ${paths[:, -1].std():.2f}")
    print(f"  Min:  ${paths[:, -1].min():.2f}")
    print(f"  Max:  ${paths[:, -1].max():.2f}")
    
    # Create visualization
    viz = FinancialVisualizer()
    fig = viz.plot_simulated_paths(paths, gbm.times, n_paths_display=50)
    plt.savefig('stock_simulation_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Visualization saved as 'stock_simulation_demo.png'")

def demo_option_pricing():
    """Demonstrate option pricing with Monte Carlo vs Black-Scholes."""
    print("\n" + "="*60)
    print("2. OPTION PRICING DEMO")
    print("="*60)
    
    # Parameters
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    
    print(f"Option Parameters: S0=${S0}, K=${K}, T={T}yr, r={r:.1%}, σ={sigma:.1%}")
    
    # Compare MC vs BS
    comparison = compare_mc_bs_pricing(S0, K, T, r, sigma, 100000)
    print(f"\nMonte Carlo vs Black-Scholes Comparison:")
    print(comparison.round(4))
    
    # Price call option with MC
    pricer = MonteCarloOptionPricer(S0, r, sigma, T)
    call_result = pricer.price_european_option(K, 'call', 50000, 42)
    
    print(f"\nDetailed Call Option Results:")
    print(f"  MC Price: ${call_result['price']:.4f} ± ${call_result['std_error']:.4f}")
    print(f"  95% CI: [${call_result['confidence_interval'][0]:.4f}, ${call_result['confidence_interval'][1]:.4f}]")
    
    # Greeks
    greeks = BlackScholes.greeks(S0, K, T, r, sigma)
    print(f"\nOption Greeks (Call/Put):")
    for greek, (call_val, put_val) in greeks.items():
        print(f"  {greek.capitalize():6}: {call_val:8.4f} / {put_val:8.4f}")
    
    # Asian option
    asian_result = pricer.price_asian_option(K, 'call', 25000, 50, 42)
    print(f"\nAsian Call Option:")
    print(f"  Price: ${asian_result['price']:.4f}")
    print(f"  European/Asian Ratio: {call_result['price']/asian_result['price']:.3f}")
    
    # Barrier option
    barrier_result = pricer.price_barrier_option(K, 110, 'call', 'knock_out', 25000, 50, 42)
    print(f"\nBarrier Call Option (Knock-Out at $110):")
    print(f"  Price: ${barrier_result['price']:.4f}")
    print(f"  Active Fraction: {barrier_result['active_fraction']:.2%}")

def demo_convergence_analysis():
    """Demonstrate Monte Carlo convergence."""
    print("\n" + "="*60)
    print("3. CONVERGENCE ANALYSIS DEMO")
    print("="*60)
    
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    
    conv_data = convergence_analysis(S0, K, T, r, sigma, 50000, 2500)
    
    print(f"Convergence Analysis Results:")
    print(f"  Black-Scholes Price: ${conv_data['bs_price']:.6f}")
    print(f"  Final MC Price: ${conv_data['mc_prices'][-1]:.6f}")
    print(f"  Final Error: ${conv_data['final_error']:.6f}")
    print(f"  Final Std Error: ${conv_data['std_errors'][-1]:.6f}")
    
    # Create convergence plot
    viz = FinancialVisualizer()
    fig = viz.plot_convergence(conv_data)
    plt.savefig('convergence_demo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Convergence plot saved as 'convergence_demo.png'")

def demo_portfolio_risk():
    """Demonstrate portfolio risk analysis."""
    print("\n" + "="*60)
    print("4. PORTFOLIO RISK ANALYSIS DEMO")
    print("="*60)
    
    # Create sample portfolio
    assets_data, weights = create_sample_portfolio()
    
    print(f"Portfolio Composition:")
    for i, (name, weight) in enumerate(zip(assets_data['names'], weights)):
        print(f"  {name}: {weight:.1%} (μ={assets_data['mu'][i]:.1%}, σ={assets_data['sigma'][i]:.1%})")
    
    # Analyze portfolio risk
    analyzer = PortfolioRiskAnalyzer(assets_data, weights)
    simulation = analyzer.simulate_portfolio_scenarios(10000, 252, 42)
    metrics = analyzer.calculate_risk_metrics(simulation)
    
    print(f"\nRisk Metrics (1-Year Horizon):")
    print(f"  Expected Return: {metrics['expected_return']:8.2%}")
    print(f"  Volatility: {metrics['return_volatility']:13.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:11.3f}")
    print(f"  VaR (95%): {metrics['VaR_95%_total']:14.2%}")
    print(f"  CVaR (95%): {metrics['CVaR_95%_total']:13.2%}")
    print(f"  Max Drawdown: {metrics['avg_max_drawdown']:9.2%}")
    
    # Stress testing
    stress_scenarios = {
        'market_crash': {
            'return_adjustment': -0.10,
            'volatility_multiplier': 2.0,
            'correlation_multiplier': 1.5
        }
    }
    
    stress_results = analyzer.stress_test(stress_scenarios, 5000)
    
    print(f"\nStress Test Results:")
    for scenario_name, results in stress_results.items():
        metrics = results['metrics']
        name = scenario_name.replace('_', ' ').title()
        print(f"  {name}:")
        print(f"    Expected Return: {metrics['expected_return']:6.2%}")
        print(f"    VaR (95%): {metrics.get('VaR_95%_total', 0):12.2%}")

def demo_comprehensive():
    """Run comprehensive demo of all features."""
    print("🎲 MONTE CARLO FINANCIAL RISK ENGINE")
    print("   Comprehensive Feature Demonstration")
    
    # Run all demos
    demo_stock_simulation()
    demo_option_pricing()
    demo_convergence_analysis()
    demo_portfolio_risk()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE! 🎉")
    print("="*60)
    print("Key achievements:")
    print("  ✅ Stock price simulation with GBM")
    print("  ✅ Option pricing (European, Asian, Barrier)")
    print("  ✅ Monte Carlo vs Black-Scholes validation")
    print("  ✅ Portfolio risk analysis (VaR, CVaR, drawdown)")
    print("  ✅ Stress testing capabilities")
    print("  ✅ Visualization generation")
    
    print(f"\nNext steps:")
    print(f"  📊 Run interactive dashboard: streamlit run streamlit_app.py")
    print(f"  📈 Explore individual modules in detail")
    print(f"  🔧 Customize parameters for your specific use case")

if __name__ == "__main__":
    demo_comprehensive()