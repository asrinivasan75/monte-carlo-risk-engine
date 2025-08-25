import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from simulate_paths import GBMSimulator, MultiAssetGBM, generate_sample_data
from option_pricing import MonteCarloOptionPricer, BlackScholes, compare_mc_bs_pricing, convergence_analysis
from risk_metrics import PortfolioRiskAnalyzer, RiskMetrics, create_sample_portfolio
from visualizations import FinancialVisualizer

# Configure Streamlit page
st.set_page_config(
    page_title="Monte Carlo Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.markdown('<div class="main-header">🎲 Monte Carlo Financial Risk Engine</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Comprehensive Monte Carlo simulation engine for financial risk analysis and derivative pricing.**
    
    Features:
    - **Stock Price Simulation**: Geometric Brownian Motion (GBM) path generation
    - **Option Pricing**: Monte Carlo vs Black-Scholes comparison for European, Asian, and Barrier options
    - **Portfolio Risk**: VaR, CVaR, drawdown analysis with multi-asset correlations
    - **Interactive Visualizations**: Real-time parameter adjustment and analysis
    """)
    
    # Sidebar for navigation
    st.sidebar.title("🔧 Navigation")
    selected_section = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["🏠 Overview", "📊 Stock Simulation", "💰 Option Pricing", "📈 Portfolio Risk", "🎯 Stress Testing"]
    )
    
    # Initialize visualizer
    viz = FinancialVisualizer()
    
    if selected_section == "🏠 Overview":
        show_overview()
    elif selected_section == "📊 Stock Simulation":
        show_stock_simulation(viz)
    elif selected_section == "💰 Option Pricing":
        show_option_pricing(viz)
    elif selected_section == "📈 Portfolio Risk":
        show_portfolio_risk(viz)
    elif selected_section == "🎯 Stress Testing":
        show_stress_testing(viz)

def show_overview():
    """Display application overview and quick examples."""
    
    st.markdown('<div class="section-header">📋 Application Overview</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        
        **Monte Carlo Simulation Engine:**
        - Single and multi-asset GBM simulation
        - Correlated asset path generation
        - Efficient terminal price sampling
        
        **Option Pricing Tools:**
        - European Call/Put options
        - Asian (path-dependent) options  
        - Barrier options (knock-in/out)
        - Greeks calculation
        - Convergence analysis
        
        **Risk Management:**
        - Value-at-Risk (VaR) calculation
        - Conditional VaR (Expected Shortfall)
        - Maximum Drawdown analysis
        - Sharpe and Sortino ratios
        - Portfolio stress testing
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Quick Example
        
        Let's price a European call option using both Monte Carlo and Black-Scholes:
        """)
        
        # Quick example calculation
        S0, K, T, r, sigma = 100, 105, 1.0, 0.05, 0.2
        
        # Black-Scholes price
        bs_price = BlackScholes.call_price(S0, K, T, r, sigma)
        
        # Monte Carlo price (small sample for speed)
        mc_pricer = MonteCarloOptionPricer(S0, r, sigma, T)
        mc_result = mc_pricer.price_european_option(K, 'call', 10000, 42)
        
        # Display results
        st.info(f"""
        **Option Parameters:**
        - Stock Price: ${S0}
        - Strike Price: ${K}  
        - Time to Expiry: {T} year
        - Risk-free Rate: {r:.1%}
        - Volatility: {sigma:.1%}
        
        **Results:**
        - Black-Scholes Price: ${bs_price:.4f}
        - Monte Carlo Price: ${mc_result['price']:.4f}
        - Difference: ${abs(bs_price - mc_result['price']):.4f}
        - MC Standard Error: ${mc_result['std_error']:.4f}
        """)
    
    # Navigation help
    st.markdown("""
    ---
    ### 🧭 How to Navigate
    
    Use the sidebar to explore different sections:
    
    1. **📊 Stock Simulation**: Visualize GBM paths and analyze price distributions
    2. **💰 Option Pricing**: Compare Monte Carlo vs analytical pricing methods
    3. **📈 Portfolio Risk**: Analyze multi-asset portfolio risk metrics
    4. **🎯 Stress Testing**: Evaluate portfolio performance under adverse scenarios
    
    Each section includes interactive controls to adjust parameters in real-time!
    """)

def show_stock_simulation(viz):
    """Display stock price simulation interface."""
    
    st.markdown('<div class="section-header">📊 Stock Price Simulation</div>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("### 📊 Simulation Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        S0 = st.number_input("Initial Price ($)", value=100.0, min_value=1.0)
        mu = st.slider("Drift Rate (%)", -20.0, 50.0, 5.0) / 100
        sigma = st.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
    
    with col2:
        T = st.slider("Time Horizon (years)", 0.1, 5.0, 1.0)
        n_paths = st.selectbox("Number of Paths", [100, 500, 1000, 5000], index=2)
        n_display = st.slider("Paths to Display", 10, min(500, n_paths), 100)
    
    random_seed = st.sidebar.checkbox("Use Random Seed", value=True)
    seed_value = st.sidebar.number_input("Random Seed", value=42) if random_seed else None
    
    # Generate simulation
    if st.sidebar.button("🎲 Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            # Create simulator
            dt = 1/252  # Daily time steps
            gbm = GBMSimulator(S0, mu, sigma, T, dt)
            
            # Simulate paths
            paths = gbm.simulate_path(n_paths, seed_value)
            
            # Create interactive plot
            fig = viz.create_interactive_paths_plot(paths, gbm.times, 
                                                  f"Simulated Stock Price Paths (n={n_paths})")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            final_prices = paths[:, -1]
            returns = (final_prices - S0) / S0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Final Price", 
                    f"${np.mean(final_prices):.2f}",
                    f"{np.mean(returns):.2%}"
                )
            
            with col2:
                st.metric(
                    "Price Volatility",
                    f"${np.std(final_prices):.2f}",
                    f"{np.std(returns):.2%}"
                )
            
            with col3:
                st.metric(
                    "Min Price",
                    f"${np.min(final_prices):.2f}",
                    f"{(np.min(final_prices) - S0) / S0:.2%}"
                )
            
            with col4:
                st.metric(
                    "Max Price", 
                    f"${np.max(final_prices):.2f}",
                    f"{(np.max(final_prices) - S0) / S0:.2%}"
                )
            
            # Price distribution
            st.markdown("### 📈 Final Price Distribution")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                name='Final Prices',
                opacity=0.7
            ))
            
            fig_hist.add_vline(x=S0, line_dash="dash", line_color="red", 
                              annotation_text=f"Initial: ${S0}")
            fig_hist.add_vline(x=np.mean(final_prices), line_dash="dash", line_color="green",
                              annotation_text=f"Mean: ${np.mean(final_prices):.2f}")
            
            fig_hist.update_layout(
                title="Distribution of Final Stock Prices",
                xaxis_title="Final Price ($)",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Show some statistics
            st.markdown("### 📊 Detailed Statistics")
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', '5th Percentile', '95th Percentile'],
                'Final Price ($)': [
                    np.mean(final_prices),
                    np.median(final_prices), 
                    np.std(final_prices),
                    pd.Series(final_prices).skew(),
                    pd.Series(final_prices).kurtosis(),
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ],
                'Return (%)': [
                    np.mean(returns) * 100,
                    np.median(returns) * 100,
                    np.std(returns) * 100,
                    pd.Series(returns).skew(),
                    pd.Series(returns).kurtosis(), 
                    np.percentile(returns, 5) * 100,
                    np.percentile(returns, 95) * 100
                ]
            })
            
            st.dataframe(stats_df.round(4), use_container_width=True)

def show_option_pricing(viz):
    """Display option pricing interface."""
    
    st.markdown('<div class="section-header">💰 Option Pricing Engine</div>', 
                unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("### 💰 Option Parameters")
    
    # Market parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        S0 = st.number_input("Stock Price ($)", value=100.0, min_value=1.0, key="opt_S0")
        K = st.number_input("Strike Price ($)", value=105.0, min_value=1.0, key="opt_K")
        T = st.slider("Time to Expiry (years)", 0.01, 2.0, 1.0, key="opt_T")
    
    with col2:
        r = st.slider("Risk-free Rate (%)", 0.0, 20.0, 5.0, key="opt_r") / 100
        sigma = st.slider("Volatility (%)", 1.0, 100.0, 20.0, key="opt_sigma") / 100
        n_sims = st.selectbox("MC Simulations", [10000, 50000, 100000, 500000], index=1, key="opt_sims")
    
    # Option type
    option_type = st.sidebar.selectbox("Option Type", ["European", "Asian", "Barrier"])
    call_put = st.sidebar.radio("Call/Put", ["Call", "Put"])
    
    # Additional parameters for complex options
    if option_type == "Asian":
        n_steps = st.sidebar.slider("Averaging Steps", 50, 500, 100)
    elif option_type == "Barrier":
        barrier = st.sidebar.number_input("Barrier Level ($)", value=110.0, min_value=1.0)
        barrier_type = st.sidebar.selectbox("Barrier Type", ["knock_out", "knock_in"])
        n_steps = st.sidebar.slider("Monitoring Steps", 50, 500, 100)
    
    if st.sidebar.button("💰 Price Option", type="primary"):
        with st.spinner("Calculating option price..."):
            
            # Create pricer
            pricer = MonteCarloOptionPricer(S0, r, sigma, T)
            
            if option_type == "European":
                # Monte Carlo pricing
                mc_result = pricer.price_european_option(K, call_put.lower(), n_sims, 42)
                
                # Black-Scholes pricing
                if call_put.lower() == "call":
                    bs_price = BlackScholes.call_price(S0, K, T, r, sigma)
                else:
                    bs_price = BlackScholes.put_price(S0, K, T, r, sigma)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Monte Carlo Price", f"${mc_result['price']:.4f}",
                             f"±${mc_result['std_error']:.4f}")
                
                with col2:
                    st.metric("Black-Scholes Price", f"${bs_price:.4f}")
                
                with col3:
                    error = abs(mc_result['price'] - bs_price)
                    st.metric("Absolute Error", f"${error:.4f}",
                             f"{error/bs_price*100:.3f}%")
                
                # Payoff distribution
                st.markdown("### 💸 Payoff Distribution")
                
                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Histogram(
                    x=mc_result['payoffs'],
                    nbinsx=50,
                    name='Payoffs',
                    opacity=0.7
                ))
                
                fig_payoff.add_vline(x=np.mean(mc_result['payoffs']), line_dash="dash", 
                                    line_color="red", annotation_text=f"Mean: ${np.mean(mc_result['payoffs']):.2f}")
                
                fig_payoff.update_layout(
                    title=f"{call_put} Option Payoff Distribution (Strike: ${K})",
                    xaxis_title="Payoff ($)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig_payoff, use_container_width=True)
                
                # Greeks calculation
                if st.checkbox("Calculate Greeks"):
                    greeks = BlackScholes.greeks(S0, K, T, r, sigma)
                    
                    st.markdown("### 🔢 Option Greeks")
                    
                    greeks_df = pd.DataFrame({
                        'Greek': ['Delta', 'Gamma', 'Theta (daily)', 'Vega (per 1%)', 'Rho (per 1%)'],
                        'Call': [greeks['delta'][0], greeks['gamma'][0], greeks['theta'][0], 
                                greeks['vega'][0], greeks['rho'][0]],
                        'Put': [greeks['delta'][1], greeks['gamma'][1], greeks['theta'][1],
                               greeks['vega'][1], greeks['rho'][1]]
                    })
                    
                    st.dataframe(greeks_df.round(4), use_container_width=True)
            
            elif option_type == "Asian":
                # Asian option pricing
                asian_result = pricer.price_asian_option(K, call_put.lower(), n_sims, n_steps, 42)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Asian Option Price", f"${asian_result['price']:.4f}",
                             f"±${asian_result['std_error']:.4f}")
                
                with col2:
                    prob_itm = np.mean(asian_result['payoffs'] > 0)
                    st.metric("Prob(ITM)", f"{prob_itm:.2%}")
                
                # Average price distribution
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Histogram(
                    x=asian_result['average_prices'],
                    nbinsx=50,
                    name='Average Prices',
                    opacity=0.7
                ))
                
                fig_avg.add_vline(x=K, line_dash="dash", line_color="red",
                                 annotation_text=f"Strike: ${K}")
                
                fig_avg.update_layout(
                    title="Distribution of Average Stock Prices",
                    xaxis_title="Average Price ($)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
            
            elif option_type == "Barrier":
                # Barrier option pricing
                barrier_result = pricer.price_barrier_option(K, barrier, call_put.lower(), 
                                                           barrier_type, n_sims, n_steps, 42)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Barrier Option Price", f"${barrier_result['price']:.4f}",
                             f"±${barrier_result['std_error']:.4f}")
                
                with col2:
                    st.metric("Active Options", f"{barrier_result['active_fraction']:.2%}")
                
                with col3:
                    prob_barrier = np.mean(barrier_result['barrier_crossed'])
                    st.metric("Barrier Crossed", f"{prob_barrier:.2%}")
                
                # Show some sample paths with barrier
                st.markdown("### 🚧 Sample Paths with Barrier")
                
                sample_paths = barrier_result['paths'][:20]  # Show first 20 paths
                times = np.linspace(0, T, sample_paths.shape[1])
                
                fig_barrier = go.Figure()
                
                for i in range(sample_paths.shape[0]):
                    color = 'red' if barrier_result['barrier_crossed'][i] else 'blue'
                    fig_barrier.add_trace(go.Scatter(
                        x=times,
                        y=sample_paths[i, :],
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=f"Path {i+1}",
                        showlegend=False
                    ))
                
                fig_barrier.add_hline(y=barrier, line_dash="dash", line_color="orange",
                                     annotation_text=f"Barrier: ${barrier}")
                fig_barrier.add_hline(y=S0, line_dash="dot", line_color="green",
                                     annotation_text=f"Initial: ${S0}")
                
                fig_barrier.update_layout(
                    title=f"Sample Paths - {barrier_type.replace('_', ' ').title()} Barrier",
                    xaxis_title="Time (Years)",
                    yaxis_title="Stock Price ($)"
                )
                
                st.plotly_chart(fig_barrier, use_container_width=True)
    
    # Convergence analysis section
    st.markdown("---")
    if st.checkbox("🎯 Convergence Analysis"):
        st.markdown("### 📈 Monte Carlo Convergence Analysis")
        
        max_sims = st.slider("Max Simulations", 10000, 500000, 100000, step=10000)
        step_size = st.slider("Step Size", 1000, 20000, 5000, step=1000)
        
        if st.button("Run Convergence Analysis"):
            with st.spinner("Analyzing convergence..."):
                conv_data = convergence_analysis(S0, K, T, r, sigma, max_sims, step_size)
                
                # Create convergence plot
                fig_conv = go.Figure()
                
                # MC prices
                fig_conv.add_trace(go.Scatter(
                    x=conv_data['n_simulations'],
                    y=conv_data['mc_prices'],
                    mode='lines',
                    name='Monte Carlo Price',
                    line=dict(color='blue')
                ))
                
                # Confidence bands
                upper_bound = conv_data['mc_prices'] + 1.96 * conv_data['std_errors']
                lower_bound = conv_data['mc_prices'] - 1.96 * conv_data['std_errors']
                
                fig_conv.add_trace(go.Scatter(
                    x=np.concatenate([conv_data['n_simulations'], conv_data['n_simulations'][::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval'
                ))
                
                # Black-Scholes reference
                fig_conv.add_hline(y=conv_data['bs_price'], line_dash="dash", line_color="red",
                                  annotation_text=f"Black-Scholes: ${conv_data['bs_price']:.4f}")
                
                fig_conv.update_layout(
                    title="Monte Carlo Convergence to Black-Scholes Price",
                    xaxis_title="Number of Simulations",
                    yaxis_title="Option Price ($)"
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)
                
                st.success(f"Final convergence error: ${conv_data['final_error']:.6f}")

def show_portfolio_risk(viz):
    """Display portfolio risk analysis interface."""
    
    st.markdown('<div class="section-header">📈 Portfolio Risk Analysis</div>', 
                unsafe_allow_html=True)
    
    # Portfolio setup
    st.sidebar.markdown("### 📈 Portfolio Setup")
    
    portfolio_choice = st.sidebar.selectbox(
        "Portfolio Type",
        ["Sample 3-Asset Portfolio", "Custom Portfolio"]
    )
    
    if portfolio_choice == "Sample 3-Asset Portfolio":
        assets_data, weights = create_sample_portfolio()
        
        # Display portfolio composition
        st.markdown("### 📊 Portfolio Composition")
        
        comp_df = pd.DataFrame({
            'Asset': assets_data['names'],
            'Weight': [f"{w:.1%}" for w in weights],
            'Initial Price': [f"${p:.2f}" for p in assets_data['S0']],
            'Expected Return': [f"{r:.1%}" for r in assets_data['mu']],
            'Volatility': [f"{v:.1%}" for v in assets_data['sigma']]
        })
        
        st.dataframe(comp_df, use_container_width=True)
        
        # Correlation matrix
        corr_df = pd.DataFrame(assets_data['correlation_matrix'], 
                              index=assets_data['names'], 
                              columns=assets_data['names'])
        
        st.markdown("### 🔗 Asset Correlation Matrix")
        st.dataframe(corr_df.round(3), use_container_width=True)
        
    else:
        # Custom portfolio builder
        st.sidebar.markdown("#### Custom Assets")
        n_assets = st.sidebar.slider("Number of Assets", 2, 5, 3)
        
        assets_data = {'names': [], 'S0': [], 'mu': [], 'sigma': [], 'correlation_matrix': []}
        weights = []
        
        # Input for each asset
        for i in range(n_assets):
            with st.sidebar.expander(f"Asset {i+1}"):
                name = st.text_input("Name", f"Asset_{i+1}", key=f"name_{i}")
                price = st.number_input("Initial Price", 50.0, 500.0, 100.0, key=f"price_{i}")
                mu = st.slider("Expected Return (%)", -20.0, 30.0, 8.0, key=f"mu_{i}") / 100
                sigma = st.slider("Volatility (%)", 5.0, 50.0, 20.0, key=f"sigma_{i}") / 100
                weight = st.slider("Weight (%)", 0.0, 100.0, 100.0/n_assets, key=f"weight_{i}") / 100
                
                assets_data['names'].append(name)
                assets_data['S0'].append(price)
                assets_data['mu'].append(mu)
                assets_data['sigma'].append(sigma)
                weights.append(weight)
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w/weight_sum for w in weights]
        
        # Simple correlation matrix (can be enhanced)
        correlation = st.sidebar.slider("Average Correlation", -0.5, 0.9, 0.3)
        assets_data['correlation_matrix'] = [[1.0 if i==j else correlation 
                                            for j in range(n_assets)] 
                                           for i in range(n_assets)]
        
        weights = np.array(weights)
    
    # Simulation parameters
    st.sidebar.markdown("### ⚙️ Simulation Parameters")
    
    n_scenarios = st.sidebar.selectbox("Scenarios", [1000, 5000, 10000, 50000], index=2)
    time_horizon = st.sidebar.slider("Time Horizon (days)", 30, 500, 252)
    confidence_levels = st.sidebar.multiselect("Confidence Levels", [0.90, 0.95, 0.99], [0.95, 0.99])
    
    if st.sidebar.button("📊 Analyze Portfolio Risk", type="primary"):
        with st.spinner("Running portfolio risk analysis..."):
            
            # Create analyzer
            analyzer = PortfolioRiskAnalyzer(assets_data, weights)
            
            # Run simulation
            simulation = analyzer.simulate_portfolio_scenarios(
                n_scenarios, time_horizon, random_state=42
            )
            
            # Calculate metrics
            metrics = analyzer.calculate_risk_metrics(simulation, confidence_levels)
            
            # Display key metrics
            st.markdown("### 📊 Key Risk Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Expected Return", 
                    f"{metrics['expected_return']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Portfolio Volatility",
                    f"{metrics['return_volatility']:.2%}"
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Max Drawdown",
                    f"{metrics['avg_max_drawdown']:.2%}"
                )
            
            # VaR/CVaR metrics
            st.markdown("### ⚠️ Risk Measures")
            
            risk_cols = st.columns(len(confidence_levels))
            
            for i, conf_level in enumerate(confidence_levels):
                with risk_cols[i]:
                    var_key = f'VaR_{int(conf_level*100)}%_total'
                    cvar_key = f'CVaR_{int(conf_level*100)}%_total'
                    
                    st.markdown(f"**{conf_level:.0%} Confidence Level:**")
                    st.metric("Value at Risk", f"{metrics[var_key]:.2%}")
                    st.metric("Conditional VaR", f"{metrics[cvar_key]:.2%}")
            
            # Interactive dashboard
            st.markdown("### 📈 Interactive Risk Dashboard")
            
            portfolio_returns = simulation['total_returns']
            fig_dashboard = viz.create_interactive_risk_dashboard(metrics, portfolio_returns)
            
            st.plotly_chart(fig_dashboard, use_container_width=True)
            
            # Portfolio value evolution
            st.markdown("### 💰 Portfolio Value Evolution")
            
            portfolio_values = simulation['portfolio_values']
            times = np.linspace(0, time_horizon/252, portfolio_values.shape[1])
            
            # Show sample paths
            fig_portfolio = go.Figure()
            
            n_display = min(100, portfolio_values.shape[0])
            for i in range(n_display):
                fig_portfolio.add_trace(go.Scatter(
                    x=times,
                    y=portfolio_values[i, :],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(100,100,100,0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add mean path
            mean_path = np.mean(portfolio_values, axis=0)
            fig_portfolio.add_trace(go.Scatter(
                x=times,
                y=mean_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Mean Path'
            ))
            
            # Add percentile bands
            p5 = np.percentile(portfolio_values, 5, axis=0)
            p95 = np.percentile(portfolio_values, 95, axis=0)
            
            fig_portfolio.add_trace(go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([p95, p5[::-1]]),
                fill='toself',
                fillcolor='rgba(128,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='5th-95th Percentile'
            ))
            
            fig_portfolio.update_layout(
                title=f"Portfolio Value Evolution ({time_horizon} days)",
                xaxis_title="Time (Years)",
                yaxis_title="Portfolio Value ($)"
            )
            
            st.plotly_chart(fig_portfolio, use_container_width=True)
            
            # Detailed statistics table
            if st.checkbox("📋 Show Detailed Statistics"):
                st.markdown("### 📋 Detailed Risk Statistics")
                
                stats_data = []
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if 'ratio' in key.lower() or 'expected' in key.lower():
                            formatted_value = f"{value:.4f}"
                        elif '%' in key or 'var' in key.lower() or 'drawdown' in key.lower():
                            formatted_value = f"{value:.4%}"
                        else:
                            formatted_value = f"{value:.6f}"
                        
                        stats_data.append({
                            'Metric': key.replace('_', ' ').title(),
                            'Value': formatted_value
                        })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)

def show_stress_testing(viz):
    """Display stress testing interface."""
    
    st.markdown('<div class="section-header">🎯 Portfolio Stress Testing</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Stress testing evaluates portfolio performance under adverse market conditions.
    Adjust the scenarios below to see how different market environments affect your portfolio.
    """)
    
    # Use sample portfolio for stress testing
    assets_data, weights = create_sample_portfolio()
    
    # Display base portfolio
    st.markdown("### 📊 Base Portfolio")
    comp_df = pd.DataFrame({
        'Asset': assets_data['names'],
        'Weight': [f"{w:.1%}" for w in weights],
        'Expected Return': [f"{r:.1%}" for r in assets_data['mu']],
        'Volatility': [f"{v:.1%}" for v in assets_data['sigma']]
    })
    st.dataframe(comp_df, use_container_width=True)
    
    # Stress scenario configuration
    st.sidebar.markdown("### 🎯 Stress Scenarios")
    
    # Predefined scenarios
    run_predefined = st.sidebar.checkbox("Use Predefined Scenarios", value=True)
    
    if run_predefined:
        scenarios = {
            'market_crash': {
                'return_adjustment': -0.15,
                'volatility_multiplier': 2.5,
                'correlation_multiplier': 1.8,
                'description': 'Severe market downturn with high volatility and correlations'
            },
            'low_vol_regime': {
                'volatility_multiplier': 0.6,
                'correlation_multiplier': 0.7,
                'description': 'Low volatility environment with reduced correlations'
            },
            'high_correlation': {
                'correlation_multiplier': 2.0,
                'volatility_multiplier': 1.3,
                'description': 'Market stress with increased asset correlations'
            },
            'stagflation': {
                'return_adjustment': -0.08,
                'volatility_multiplier': 1.5,
                'description': 'Economic stagnation with moderate volatility increase'
            }
        }
    else:
        # Custom scenario builder
        st.sidebar.markdown("#### Custom Scenario")
        custom_scenarios = {}
        
        scenario_name = st.sidebar.text_input("Scenario Name", "custom_scenario")
        return_adj = st.sidebar.slider("Return Adjustment", -0.30, 0.10, -0.05)
        vol_mult = st.sidebar.slider("Volatility Multiplier", 0.3, 3.0, 1.0)
        corr_mult = st.sidebar.slider("Correlation Multiplier", 0.3, 3.0, 1.0)
        
        custom_scenarios[scenario_name] = {
            'return_adjustment': return_adj,
            'volatility_multiplier': vol_mult,
            'correlation_multiplier': corr_mult,
            'description': f'Custom scenario: {return_adj:.1%} return adj, {vol_mult:.1f}x vol, {corr_mult:.1f}x corr'
        }
        
        scenarios = custom_scenarios
    
    n_scenarios = st.sidebar.selectbox("Simulations per Scenario", [1000, 5000, 10000], index=1)
    
    if st.sidebar.button("🎯 Run Stress Tests", type="primary"):
        with st.spinner("Running stress tests..."):
            
            # Create analyzer
            analyzer = PortfolioRiskAnalyzer(assets_data, weights)
            
            # Remove descriptions from scenarios for analysis
            clean_scenarios = {k: {key: val for key, val in v.items() if key != 'description'} 
                             for k, v in scenarios.items()}
            
            # Run stress tests
            stress_results = analyzer.stress_test(clean_scenarios, n_scenarios)
            
            # Display results
            st.markdown("### 📊 Stress Test Results")
            
            # Create comparison table
            comparison_data = []
            
            for scenario_name, results in stress_results.items():
                metrics = results['metrics']
                
                if scenario_name == 'base_case':
                    display_name = "Base Case"
                else:
                    display_name = scenario_name.replace('_', ' ').title()
                
                comparison_data.append({
                    'Scenario': display_name,
                    'Expected Return': f"{metrics['expected_return']:.2%}",
                    'Volatility': f"{metrics['return_volatility']:.2%}",
                    'VaR 95%': f"{metrics.get('VaR_95%_total', 0):.2%}",
                    'CVaR 95%': f"{metrics.get('CVaR_95%_total', 0):.2%}",
                    'Max Drawdown': f"{metrics['avg_max_drawdown']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize stress test results
            st.markdown("### 📈 Stress Test Visualization")
            
            # Create comparison charts
            fig_stress = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Expected Returns', 'Value at Risk (95%)', 
                              'Volatility', 'Sharpe Ratios'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            scenario_names = list(stress_results.keys())
            
            # Expected Returns
            returns = [stress_results[s]['metrics']['expected_return'] for s in scenario_names]
            fig_stress.add_trace(go.Bar(
                x=[s.replace('_', ' ').title() for s in scenario_names],
                y=returns,
                name='Expected Return',
                marker_color='lightblue'
            ), row=1, col=1)
            
            # VaR 95%
            vars_95 = [stress_results[s]['metrics'].get('VaR_95%_total', 0) for s in scenario_names]
            fig_stress.add_trace(go.Bar(
                x=[s.replace('_', ' ').title() for s in scenario_names],
                y=vars_95,
                name='VaR 95%',
                marker_color='red'
            ), row=1, col=2)
            
            # Volatility
            vols = [stress_results[s]['metrics']['return_volatility'] for s in scenario_names]
            fig_stress.add_trace(go.Bar(
                x=[s.replace('_', ' ').title() for s in scenario_names],
                y=vols,
                name='Volatility',
                marker_color='orange'
            ), row=2, col=1)
            
            # Sharpe Ratios
            sharpes = [stress_results[s]['metrics']['sharpe_ratio'] for s in scenario_names]
            fig_stress.add_trace(go.Bar(
                x=[s.replace('_', ' ').title() for s in scenario_names],
                y=sharpes,
                name='Sharpe Ratio',
                marker_color='green'
            ), row=2, col=2)
            
            fig_stress.update_layout(
                height=600,
                showlegend=False,
                title_text="Stress Test Results Comparison"
            )
            
            st.plotly_chart(fig_stress, use_container_width=True)
            
            # Show scenario descriptions
            if run_predefined:
                st.markdown("### 📝 Scenario Descriptions")
                for scenario_name, scenario_data in scenarios.items():
                    if scenario_name != 'base_case':
                        st.info(f"**{scenario_name.replace('_', ' ').title()}:** {scenario_data['description']}")
            
            # Detailed analysis for selected scenario
            st.markdown("### 🔍 Detailed Scenario Analysis")
            
            selected_scenario = st.selectbox(
                "Select Scenario for Detailed Analysis:",
                [s.replace('_', ' ').title() for s in scenario_names]
            )
            
            selected_key = selected_scenario.lower().replace(' ', '_')
            
            if selected_key in stress_results:
                scenario_results = stress_results[selected_key]
                scenario_returns = scenario_results['simulation']['total_returns']
                
                # Return distribution comparison
                fig_dist = go.Figure()
                
                # Base case
                base_returns = stress_results['base_case']['simulation']['total_returns']
                fig_dist.add_trace(go.Histogram(
                    x=base_returns,
                    nbinsx=50,
                    name='Base Case',
                    opacity=0.7,
                    histnorm='probability'
                ))
                
                # Selected scenario
                fig_dist.add_trace(go.Histogram(
                    x=scenario_returns,
                    nbinsx=50,
                    name=selected_scenario,
                    opacity=0.7,
                    histnorm='probability'
                ))
                
                fig_dist.update_layout(
                    title=f"Return Distribution Comparison: Base Case vs {selected_scenario}",
                    xaxis_title="Portfolio Returns",
                    yaxis_title="Probability",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()