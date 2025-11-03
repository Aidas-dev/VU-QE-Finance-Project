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

#------------------------------------------------------------------------------------------
# Functions:

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
    #df = df.dropna()

    # Removing columns containing "PX_BID" or "returns" in any row.
    cols_to_drop = [col for col in df.columns
                   if df[col].astype(str).str.contains('PX_BID|returns').any()]
    df = df.drop(columns=cols_to_drop)

    # Removing columns with blank values in any row. Since we are calculating the Sharpe ratio of a 118 month portfolio,
    # we won't use those financial instruments that have missing data.
    df = df.dropna(axis=1, how='any')

    # Converting df columns to numeric.
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

# Extracting the risk-free rate (USGG10YR Index):
def extract_risk_free_rate(df):

    # Extracting the risk-free rate (USGG10YR Index) from the data.
    risk_free_rate = df['USGG10YR Index'].copy()

    return risk_free_rate

# Creating a new dataframe with the monthly returns of each instrument in the 118 month portfolio:
def calculate_monthly_returns(price_df):

    # Removing the risk-free rate (USGG10YR Index) column from the data.
    price_df = price_df.drop(columns=['USGG10YR Index'])

    # Calculating simple returns: (Price_t / Price_{t-1}) - 1
    returns_df = price_df.pct_change()

    # Converting to percentage.
    returns_df = returns_df * 100

    # Removing the first row (NaN).
    returns_df = returns_df.iloc[1:]

    return returns_df

#----------------------------------------------------------------------------------------------------------

# Cleaning the data.
clean_df = clean_data(df)
# Risk-free rate.
risk_free_rate = extract_risk_free_rate(clean_df)
# Calculating monthly returns.
monthly_returns = calculate_monthly_returns(clean_df)

#-------------------------------------------------------------------------------------------------------------------

# Exporting the cleaned data to a xlsx file:
clean_df.to_excel('Cleaned-Sharpe-ratio-picked-instruments-price-data.xlsx')

# Exporting the risk-free rate to a xlsx file:
risk_free_rate.to_excel('Risk-free-rate-Sharpe-ratio-picked-instruments-data.xlsx')

# Exporting the monthly returns to a xlsx file:
monthly_returns.to_excel('Monthly-returns-Sharpe-ratio-picked-instruments-data.xlsx')
