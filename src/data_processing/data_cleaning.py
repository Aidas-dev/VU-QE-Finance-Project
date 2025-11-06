import os
import pandas as pd
from datetime import datetime
import openpyxl

def clean_data(df):
    """Clean and prepare financial instrument data"""
    # Making first column the row index, this also removes rows with more values than the first row.
    df = df.set_index(df.columns[0])

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
                    if any(keyword in str(val).lower() 
                          for val in df[col].head() 
                          for keyword in ['px_bid', 'returns'])]
    df = df.drop(columns=cols_to_drop)

    # Removing columns with blank values in any row.
    df = df.dropna(axis=1, how='any')

    # Converting df columns to numeric.
    df = df.apply(pd.to_numeric, errors='coerce')

    # Converting the index to datetime.
    df.index = pd.to_datetime(df.index)

    # Sort by index (ascending = oldest first)
    df = df.sort_index(ascending=True)

    return df

def extract_risk_free_rate(df):
    """Extract risk-free rate from the data and turn it to monthly returns"""
    try:
        # Try to extract the risk-free rate (USGG10YR Index) from the data
        risk_free_rate1 = df['USGG10YR Index'].copy()
        
        # Removing the first row (NaN)
        risk_free_rateYR = risk_free_rate1.iloc[1:]
        
        # Convert annual rate to monthly
        risk_free_rate = (1 + risk_free_rateYR/100) ** (1/12) - 1

        # Convert to percentage
        risk_free_rate = risk_free_rate * 100
        return risk_free_rate
    except KeyError:
        # If column doesn't exist, use default risk-free rate of 2%
        print("Warning: USGG10YR Index not found - using default risk-free rate of 2%")
        default_annual_rate = 0.02
        risk_free_rate = (1 + default_annual_rate) ** (1/12) - 1
        return pd.Series([risk_free_rate] * (len(df) - 1), index=df.index[1:])
