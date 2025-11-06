import numpy as np
import pandas as pd

def calculate_monthly_returns(price_df):
    """Calculate monthly returns with robust data cleaning"""
    try:
        # Calculate percentage change between months
        returns = price_df.pct_change()
        
        # Remove first row (NaN)
        returns = returns.iloc[1:]
        
        # Convert returns to percentage
        returns = returns * 100
        
        return returns
    except Exception as e:
        print(f"Error calculating monthly returns: {str(e)}")
        return None

def calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year=12):
    """
    Calculate annualized Sharpe ratio for each instrument

    Parameters:
    returns (DataFrame): DataFrame of monthly returns
    risk_free_rate (Series): Series of risk-free rates
    periods_per_year (int): Number of periods per year (default 12 for monthly)

    Returns:
    DataFrame: Sharpe ratios for each instrument
    """
    # Calculate excess returns
    excess_returns = returns.sub(risk_free_rate, axis=0)

    # Annualize mean returns and standard deviation
    annualized_mean = excess_returns.mean() * periods_per_year
    annualized_std = excess_returns.std() * np.sqrt(periods_per_year)

    # Calculate Sharpe ratio
    sharpe_ratios = annualized_mean / annualized_std

    return sharpe_ratios

def portfolio_sharpe_ratio(weights, returns, risk_free_rate, periods_per_year=12):
    """Calculate Sharpe ratio for a weighted portfolio"""
    portfolio_returns = (returns * weights).sum(axis=1)
    excess_returns = portfolio_returns - risk_free_rate
    annualized_mean = excess_returns.mean() * periods_per_year
    annualized_std = excess_returns.std() * np.sqrt(periods_per_year)
    return annualized_mean / annualized_std
