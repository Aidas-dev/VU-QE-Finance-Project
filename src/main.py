import json
import os
import pandas as pd
from data_processing.data_cleaning import clean_data, extract_risk_free_rate
from calculations.financial_metrics import (
    calculate_monthly_returns,
    calculate_sharpe_ratio,
    portfolio_sharpe_ratio
)
from optimization.portfolio_optimization import optimize_portfolio
from visualization.portfolio_plots import plot_portfolio_allocation

def load_config():
    """Load configuration from config.json"""
    with open('config.json') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Load input data
    df = pd.read_csv(config['input_data']['path'])
    
    # Data processing
    clean_df = clean_data(df)
    risk_free_rate = extract_risk_free_rate(clean_df)
    
    # Financial calculations
    monthly_returns = calculate_monthly_returns(clean_df)
    sharpe_ratios = calculate_sharpe_ratio(monthly_returns, risk_free_rate)
    
    # Portfolio optimization
    optimal_weights_no_shorting = optimize_portfolio(monthly_returns, risk_free_rate)
    optimal_weights_with_shorting = optimize_portfolio(
        monthly_returns, risk_free_rate, allow_shorting=True
    )
    
    # Portfolio performance
    sharpe_no_shorting = portfolio_sharpe_ratio(
        optimal_weights_no_shorting, monthly_returns, risk_free_rate
    )
    sharpe_with_shorting = portfolio_sharpe_ratio(
        optimal_weights_with_shorting, monthly_returns, risk_free_rate
    )
    
    print(f"\nOptimized Portfolio Sharpe Ratios:")
    print(f"No shorting: {sharpe_no_shorting:.4f}")
    print(f"With shorting: {sharpe_with_shorting:.4f}\n")
    
    # Visualization
    plot_portfolio_allocation(
        optimal_weights_no_shorting,
        'Optimal Portfolio Allocation (No Shorting)',
        'Optimal-portfolio-allocation-no-shorting'
    )
    plot_portfolio_allocation(
        optimal_weights_with_shorting,
        'Optimal Portfolio Allocation (With Shorting)',
        'Optimal-portfolio-allocation-with-shorting'
    )
    
    # Data export
    os.makedirs(config['output']['xlsx'], exist_ok=True)
    
    clean_df.to_excel(f"{config['output']['xlsx']}/Cleaned-Sharpe-ratio-picked-instruments-price-data.xlsx")
    risk_free_rate.to_excel(f"{config['output']['xlsx']}/Risk-free-rate-Sharpe-ratio-picked-instruments-data.xlsx")
    monthly_returns.to_excel(f"{config['output']['xlsx']}/Monthly-returns-Sharpe-ratio-picked-instruments-data.xlsx")
    sharpe_ratios.to_excel(f"{config['output']['xlsx']}/Sharpe-ratio-results.xlsx")
    optimal_weights_no_shorting.to_excel(f"{config['output']['xlsx']}/Optimal-portfolio-weights(no_shorting).xlsx")
    optimal_weights_with_shorting.to_excel(f"{config['output']['xlsx']}/Optimal-portfolio-weights(with_shorting).xlsx")

if __name__ == '__main__':
    main()
