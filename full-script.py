#Standard libraries:

import os
import sys
from datetime import datetime
import json

#Imported libraries:

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

#Importing the data:

df = pd.read_csv('Sharpe-ratio-picked-instruments-data.csv')

#------------------------------------------------------------------------------------------
#Functions:

#Cleaning the data:
def clean_data(df):

    #Making first column the row index.
    df = df.set_index (df.columns[0])

    #Removing rows 1 to 6.
    df = df.drop(df.index[0:5])

    #Removing unnamed columns.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    #Removing columns containing "Security." .
    df = df.loc[:, ~df.columns.str.contains('Security')]
    #df = df.dropna()

    #Removing columns containing "PX_BID" or "returns" in any row.
    cols_to_drop = [col for col in df.columns
                   if df[col].astype(str).str.contains('PX_BID|returns').any()]
    df = df.drop(columns=cols_to_drop)

    
    return df

#---------------------------------------------------------------------------------------------

cleanDF = clean_data(df)
print(cleanDF)

#Save file as xlsx:
cleanDF.to_excel('Cleaned-Sharpe-ratio-picked-instruments-data.xlsx')