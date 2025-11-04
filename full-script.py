# Standard libraries:

import os
import sys
from datetime import datetime
import json

# Imported libraries:

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl
import matplotlib.pyplot as plt
from pipe import select, where, take, groupby, sort, dedup
import seaborn as sns
import cvxopt as cvx
import pypfopt as pfopt
import openpyxl 

# Importing the data:

df = pd.read_csv('Sharpe-ratio-picked-instruments-data.csv')

"""
# Functions: 
"""

# Cleaning the data:
def clean_data(df):

    # Making first column the row index, this also removes rows with more values than the first row.
    df = df.set_index (df.columns[0])

    # Renaming the index column to "Date".
    df.index.name = 'Date'

    # Removing rows 1 to 6.
    df = df.drop(df.index[0:6])

    # Removing unnamed columns.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Removing columns containing "Security." .
    df = df.loc[:, ~df.columns.str.contains('Security')]


    # Removing columns containing "PX_BID" or "returns" in any row.
    cols_to_drop = [col for col in df.columns
                   if df[col].astype(str).str.contains('PX_BID|returns').any()]
    df = df.drop(columns=cols_to_drop)

    # Removing columns with blank values in any row. Since we are calculating the Sharpe ratio of a 118 month portfolio,
    # we won't use those financial instruments that have missing data.
    df = df.dropna(axis=1, how='any')

    # Converting df columns to numeric.
    df = df.apply(pd.to_numeric, errors='coerce')

    # Converting the index to datetime.
    df.index = pd.to_datetime(df.index)

    # Sort by index (ascending = oldest first)
    df = df.sort_index(ascending=True)

    return df

# Extracting the risk-free rate (USGG10YR Index), already in percentage.
def extract_risk_free_rate(df):

    # Extracting the risk-free rate (USGG10YR Index) from the data.
    risk_free_rate = df['USGG10YR Index'].copy()

    # Removing the first row (NaN). 
    risk_free_rate = risk_free_rate.iloc[1:]

    return risk_free_rate

# Creating a new dataframe with the monthly returns of each instrument in the 118 month portfolio:
def calculate_monthly_returns(price_df):
    """Calculate monthly returns with robust data cleaning"""
    try:
        # Remove risk-free rate column
        price_df = price_df.drop(columns=['USGG10YR Index'], errors='ignore')
        
        # Convert to numeric and clean
        price_df = price_df.apply(pd.to_numeric, errors='coerce')
        price_df = price_df.replace([np.inf, -np.inf], np.nan)
        
        # Calculate returns
        returns_df = price_df.pct_change()
        
        # Additional cleaning
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
        returns_df = returns_df.dropna(axis=1, how='all')
        
        # Convert to percentage and remove first NaN row
        returns_df = returns_df * 100
        returns_df = returns_df.iloc[1:]
        
        # Final validation
        if returns_df.empty:
            raise ValueError("No valid returns data after cleaning")
            
        return returns_df
        
    except Exception as e:
        print(f"Error calculating returns: {str(e)}")
        raise

# Calculating Sharpe ratios
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
    # Calculate excess returns.
    excess_returns = returns.sub(risk_free_rate, axis=0)
    
    # Annualize mean returns and standard deviation.
    annualized_mean = excess_returns.mean() * periods_per_year
    annualized_std = excess_returns.std() * np.sqrt(periods_per_year)
    
    # Calculate Sharpe ratio.
    sharpe_ratios = annualized_mean / annualized_std
    
    return sharpe_ratios

# Optimizing portfolio weights.
def optimize_portfolio(returns, risk_free_rate, allow_shorting=False):
    """
    Optimize portfolio weights to maximize Sharpe ratio with optional shorting
    
    Parameters:
    returns (DataFrame): DataFrame of asset returns
    risk_free_rate (Series): Series of risk-free rates
    allow_shorting (bool): Whether to allow negative weights (short positions)
    
    Returns:
    Series: Optimal weights for each asset
    """
    # Number of assets
    n = len(returns.columns)
    
    # Calculate excess returns
    excess_returns = returns.sub(risk_free_rate, axis=0)
    
    # Mean and covariance of excess returns
    mu = np.array(excess_returns.mean())
    S = np.array(excess_returns.cov())
    
    # Convert to cvxopt matrices
    P = cvx.matrix(S)
    q = cvx.matrix(np.zeros(n))
    
    # Constraints: sum to 1
    A = cvx.matrix(1.0, (1, n))
    b = cvx.matrix(1.0)
    
    if not allow_shorting:
        # No shorting constraints
        G = cvx.matrix(-np.eye(n))
        h = cvx.matrix(np.zeros(n))
        solution = cvx.solvers.qp(P, q, G, h, A, b)
    else:
        solution = cvx.solvers.qp(P, q, A=A, b=b)
    
    weights = np.array(solution['x']).flatten()
    
    # Calculate portfolio statistics
    portfolio_return = np.dot(weights, mu)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(S, weights)))
    
    # Verify portfolio has positive excess return
    if portfolio_return <= 0:
        # If not, find asset with highest Sharpe ratio
        sharpe_ratios = mu / np.sqrt(np.diag(S))
        best_asset = np.argmax(sharpe_ratios)
        weights = np.zeros(n)
        weights[best_asset] = 1.0
    
    return pd.Series(weights, index=returns.columns)

# Calculate portfolio Sharpe ratios
def portfolio_sharpe_ratio(weights, returns, risk_free_rate, periods_per_year=12):
    """Calculate Sharpe ratio for a weighted portfolio"""
    portfolio_returns = (returns * weights).sum(axis=1)
    excess_returns = portfolio_returns - risk_free_rate
    annualized_mean = excess_returns.mean() * periods_per_year
    annualized_std = excess_returns.std() * np.sqrt(periods_per_year)
    return annualized_mean / annualized_std

"""
 Visualization functions:
"""

#Efficient frontier plotting function.
def plot_efficient_frontier(returns, risk_free_rate, filename):
    """Plot the efficient frontier with fallback to simple plot"""
    print(f"\nAttempting to plot efficient frontier: {filename}")
    
    try:
        # Clean data
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(axis=1, how='any')
        
        if len(returns.columns) < 2:
            print("Insufficient assets after cleaning - skipping")
            return

        # Try full optimization plot
        try:
            mu = pfopt.expected_returns.mean_historical_return(returns)
            S = pfopt.risk_models.sample_cov(returns)
            ef = pfopt.EfficientFrontier(mu, S, solver='ECOS')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ef_max_sharpe = ef.deepcopy()
            rf_rate = risk_free_rate.mean()/100 if risk_free_rate.mean() > 1 else risk_free_rate.mean()
            
            ef_max_sharpe.max_sharpe(risk_free_rate=rf_rate)
            ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
            
            ef.plot_efficient_frontier(ax=ax, show_assets=False, color='coolwarm')
            ax.scatter(std_tangent, ret_tangent, marker="*", s=300, c="red", 
                      label="Maximum Sharpe Ratio")
            ax.set_title("Efficient Frontier", fontsize=16)
            ax.set_xlabel("Volatility (Standard Deviation)", fontsize=12)
            ax.set_ylabel("Expected Return", fontsize=12)
            ax.legend(fontsize=10)
            sns.despine()
            
            plt.savefig(f'Visualization Graphs/{filename}', dpi=300)
            plt.close()
            print("Successfully generated optimized efficient frontier plot")
            
        except Exception as opt_error:
            print(f"Optimization failed, using simple plot: {str(opt_error)}")
            plt.close()
            
            # Fallback simple plot
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(returns.std(), returns.mean(), alpha=0.5)
            ax.set_title("Simple Returns Plot (Fallback)", fontsize=16)
            ax.set_xlabel("Volatility", fontsize=12)
            ax.set_ylabel("Return", fontsize=12)
            sns.despine()
            
            plt.savefig(f'Visualization Graphs/{filename}', dpi=300)
            plt.close()
            print("Generated fallback simple returns plot")
            
    except Exception as e:
        print(f"Failed to generate any plot: {str(e)}")
        plt.close()
            
    except Exception as e:
        print(f"Error creating efficient frontier plot: {str(e)}")
        plt.close()

# Plotting the portfolio allocation as a pie chart.
def plot_portfolio_allocation(weights, title, filename):
    """Create a pie chart of portfolio allocations"""
    # Setting seaborn theme.
    sns.set_theme(style='whitegrid')
    
    # Filtering out near-zero weights.
    significant_weights = weights[weights > 0.01]
    other = weights[weights <= 0.01].sum()

    if other > 0:
        significant_weights['Other'] = other

    # Creating pie chart.
    plt.figure(figsize=(12, 8))


    patches, texts, autotexts = plt.pie(
        significant_weights,
        labels=significant_weights.index,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.85,
        explode=[0.05] * len(significant_weights),
        colors=sns.color_palette("pastel"),
        wedgeprops={'linewidth': 1, 'edgecolor': 'darkblue'}
    )

    # Style adjustments.
    plt.title(title, fontsize=16, pad=20)
    plt.axis('equal')

    # Makeing labels more readable.
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_color('white')
        autotext.set_weight('bold')

    # Add legend
    plt.legend(
        loc='upper left',
        bbox_to_anchor=(1, 1),
        fontsize=10
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(
        f'Visualization Graphs/{filename}', 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()


"""
# Main code:
"""

# Cleaning the data.
clean_df = clean_data(df)
# Risk-free rate.
risk_free_rate = extract_risk_free_rate(clean_df)
# Calculating monthly returns.
monthly_returns = calculate_monthly_returns(clean_df)
# Calculating  Sharpe ratios of each individual instrument.
sharpe_ratios = calculate_sharpe_ratio(monthly_returns, risk_free_rate)
# Optimizing portfolio weights.
optimal_weights_no_shorting = optimize_portfolio(monthly_returns, risk_free_rate)
optimal_weights_with_shorting = optimize_portfolio(monthly_returns, risk_free_rate, allow_shorting=True)

# Calculate and print Sharpe ratios of optimal portfolios.
sharpe_no_shorting = portfolio_sharpe_ratio(optimal_weights_no_shorting, monthly_returns, risk_free_rate)
sharpe_with_shorting = portfolio_sharpe_ratio(optimal_weights_with_shorting, monthly_returns, risk_free_rate)

print(f"\nOptimized Portfolio Sharpe Ratios:")
print(f"No shorting: {sharpe_no_shorting:.4f}")
print(f"With shorting: {sharpe_with_shorting:.4f}\n")

# Generate allocation plots, also saving them to the Visualization Graphs directory.
plot_portfolio_allocation(optimal_weights_no_shorting, 'Optimal Portfolio Allocation (No Shorting)', 'Optimal-portfolio-allocation-no-shorting.png')
plot_portfolio_allocation(optimal_weights_with_shorting, 'Optimal Portfolio Allocation (With Shorting)', 'Optimal-portfolio-allocation-with-shorting.png')

# Generate efficient frontier plots, also saving them to the Visualization Graphs directory.
plot_efficient_frontier(monthly_returns, risk_free_rate, 'Efficient-frontier-no-shorting.png')
plot_efficient_frontier(monthly_returns, risk_free_rate, 'Efficient-frontier-with-shorting.png')

"""
# Exporting the data as xlsx files:
"""

# Creating the output directories if they don't exist
os.makedirs('XLSX files', exist_ok=True)

# Exporting the cleaned data to a xlsx file:
clean_df.to_excel('XLSX files/Cleaned-Sharpe-ratio-picked-instruments-price-data.xlsx')

# Exporting the risk-free rate to a xlsx file:
risk_free_rate.to_excel('XLSX files/Risk-free-rate-Sharpe-ratio-picked-instruments-data.xlsx')

# Exporting the monthly returns to a xlsx file:
monthly_returns.to_excel('XLSX files/Monthly-returns-Sharpe-ratio-picked-instruments-data.xlsx')

# Exporting Sharpe ratios to a xlsx file:
sharpe_ratios.to_excel('XLSX files/Sharpe-ratio-results.xlsx')

# Exporting optimal weights with no shorting to a xlsx file:
optimal_weights_no_shorting.to_excel('XLSX files/Optimal-portfolio-weights(no_shorting).xlsx')

# Exporting optimal weights with shorting to a xlsx file:
optimal_weights_with_shorting.to_excel('XLSX files/Optimal-portfolio-weights(with_shorting).xlsx')

