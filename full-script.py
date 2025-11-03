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

#Importing the data:

df = pd.read_csv('Sharpe-ratio-picked-instruments-data.csv')

