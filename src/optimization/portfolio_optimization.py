import cvxpy as cp
import pypfopt as pfopt
import pandas as pd

def optimize_portfolio(returns, risk_free_rate, allow_shorting=False, leverage_constraint=None):
    """
    Optimize portfolio weights to maximize Sharpe ratio with optional shorting

    Parameters:
    returns (DataFrame): DataFrame of asset returns
    risk_free_rate (Series): Series of risk-free rates
    allow_shorting (bool): Whether to allow negative weights (short positions)

    Returns:
    Series: Optimal weights for each asset
    """
    # Calculate excess returns, and monthly sharpe ratio
    excess_returns = returns.sub(risk_free_rate, axis=0)
    # Calculate expected returns and the covariance matrix
    mu = excess_returns.mean()
    S = excess_returns.cov()

    # Initialize the EfficientFrontier object with dynamic weight bounds
    weight_bounds = (-1, 1) if allow_shorting else (0, 1)
    ef = pfopt.EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Get the raw optimized weights, remove small weights.
    weights = ef.max_sharpe()
    weights = ef.clean_weights()


    # Leverage constraint, no constraint by default.
    if leverage_constraint is not None:
        ef.add_constraint(lambda w: cp.sum(w) <= leverage_constraint, verbose=True)


    return pd.Series(weights, index=returns.columns)

