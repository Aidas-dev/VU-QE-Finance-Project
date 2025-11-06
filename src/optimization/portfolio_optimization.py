import cvxpy as cp
import pypfopt as pfopt
import pandas as pd

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
    # Calculate excess returns
    excess_returns = returns.sub(risk_free_rate, axis=0)

    # Create EfficientFrontier object
    ef = pfopt.EfficientFrontier(
        excess_returns.mean(),
        excess_returns.cov(),
        weight_bounds=(-1 if allow_shorting else 0, 1)
    )
    # Shorting leverage constraint, no leverage
    ef.add_constraint(lambda w: cp.sum(cp.abs(w)) <= 1.0)
    # Maximize Sharpe ratio
    weights = ef.max_sharpe()

    return pd.Series(weights, index=returns.columns)

