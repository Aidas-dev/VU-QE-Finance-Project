import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
    # Convert inputs to numpy arrays
    returns_array = returns.values
    risk_free = risk_free_rate.mean()
    
    # Calculate expected returns and covariance matrix
    mu = np.mean(returns_array, axis=0)
    S = np.cov(returns_array, rowvar=False)
    
    n_assets = len(mu)
    
    # Define optimization constraints
    constraints = []
    
    # Weight bounds
    if allow_shorting:
        bounds = [(-1, 1) for _ in range(n_assets)]
    else:
        bounds = [(0, 1) for _ in range(n_assets)]
    
    # Sum of weights constraint
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Leverage constraint if specified
    if leverage_constraint is not None:
        constraints.append({'type': 'ineq', 'fun': lambda w: leverage_constraint - np.sum(np.abs(w))})
    
    # Initial guess (equal weights)
    w0 = np.ones(n_assets) / n_assets
    
    # Define negative Sharpe ratio to minimize
    def negative_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(S, w)))
        sharpe = (port_return - risk_free) / port_vol
        return -sharpe
    
    # Optimize
    result = minimize(
        negative_sharpe,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    # Clean small weights and normalize
    weights = result.x
    weights[np.abs(weights) < 1e-3] = 0  # More reasonable threshold
    
    if allow_shorting:
        # For portfolios with shorting, normalize by sum of absolute weights
        weights = weights / np.sum(np.abs(weights))
    else:
        # For long-only portfolios, normalize by sum of weights
        weights = weights / np.sum(weights)
    
    print(dict(zip(returns.columns, weights)))
    
    return pd.Series(weights, index=returns.columns), S