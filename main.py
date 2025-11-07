import json
import os
import sys
import pandas as pd
from pypfopt import risk_models

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.data_cleaning import clean_data, extract_risk_free_rate
from src.calculations.financial_metrics import (
    calculate_monthly_returns,
    calculate_sharpe_ratio,
    portfolio_statistics
)
from src.optimization.portfolio_optimization import optimize_portfolio
from src.visualization.portfolio_plots import (
    plot_portfolio_allocation,
)

def load_config():
    """Load configuration from config.json"""
    with open('config.json') as f:
        return json.load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Load input data
    df = pd.read_excel(config['input_data']['path'])
    
    # Data processing
    clean_df = clean_data(df)
    risk_free_rate = extract_risk_free_rate(clean_df)
    
    # Financial calculations
    monthly_returns = calculate_monthly_returns(clean_df)
    
    sharpe_ratios = calculate_sharpe_ratio(monthly_returns, risk_free_rate)
    
    # Portfolio optimization
    optimal_weights_no_shorting, cov_matrix_no_shorting = optimize_portfolio(monthly_returns, risk_free_rate)
    optimal_weights_with_shorting, cov_matrix_with_shorting = optimize_portfolio(
        monthly_returns, risk_free_rate, allow_shorting=True)

    # Calculate excess returns
    excess_returns = monthly_returns.sub(risk_free_rate, axis=0)

    # Portfolio performance
    stats_no_shorting = portfolio_statistics(
        optimal_weights_no_shorting, monthly_returns, risk_free_rate
    )
    stats_with_shorting = portfolio_statistics(
        optimal_weights_with_shorting, monthly_returns, risk_free_rate
    )
    
    print("\nOptimized Portfolio Statistics:")
    print("\nNo shorting:")
    print(stats_no_shorting)
    print("\nWith shorting:")
    print(stats_with_shorting)
    
    # Portfolio allocation visualization
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
    
    # Data export - combined workbook with auto-sized columns
    os.makedirs(config['output']['xlsx'], exist_ok=True)
    output_path = f"{config['output']['xlsx'] }/Sharpe-ratio-analysis-results.xlsx"
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Write all sheets with rounded values
        df.round(3).to_excel(writer, sheet_name='Original Price Data')
        clean_df.round(3).to_excel(writer, sheet_name='Cleaned Price')
        risk_free_rate.round(3).to_excel(writer, sheet_name='Risk Free Rate')
        monthly_returns.round(3).to_excel(writer, sheet_name='Monthly Returns')
        excess_returns.round(3).to_excel(writer, sheet_name='Excess Returns')
        pd.DataFrame(cov_matrix_no_shorting,
                     index=monthly_returns.columns,
                     columns=monthly_returns.columns).round(3).to_excel(writer, sheet_name='Covariance (No Shorting)')
        pd.DataFrame(cov_matrix_with_shorting,
                     index=monthly_returns.columns,
                     columns=monthly_returns.columns).round(3).to_excel(writer, sheet_name='Covariance (With Shorting)')
        sharpe_ratios.round(3).to_excel(writer, sheet_name='Sharpe Ratios')
        optimal_weights_no_shorting.round(3).to_excel(writer, sheet_name='Optimal Weights (No Shorting)')
        optimal_weights_with_shorting.round(3).to_excel(writer, sheet_name='Optimal Weights (With Shorting)')
        stats_no_shorting.round(3).to_excel(writer, sheet_name='Portfolio Stats (No Shorting)')
        stats_with_shorting.round(3).to_excel(writer, sheet_name='Portfolio Stats (With Shorting)')
        
        # Use autofit for all column sizing including index
        for worksheet in writer.sheets.values():
            worksheet.autofit()

if __name__ == '__main__':
    main()
