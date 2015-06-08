'''
Created on 25 Sep 2014

@author: Manrich
'''
import pandas
from scipy import stats
from statsmodels.graphics.api import qqplot

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# Load data
data = sm.datasets.sunspots.load_pandas().data

# Format dates
data.index = pandas.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del data["YEAR"]

data.plot(figsize=(12,8))

fig = plt.figure(figsize=(12,8))

# Plot Autocorrelation function
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=40, ax=ax1)

# Plot Partial Autocorrelation function 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)

# Fit and ARMA(2,0) model | Xt = const + L1*X(t-1)+ L2*X(t-2) + random_error
arma_model20 = sm.tsa.ARMA(data, (2,0)).fit()
print arma_model20.params
print arma_model20.aic, arma_model20.bic, arma_model20.hqic

# Fit and ARMA(3,0) model | Xt = const + L1*X(t-1)+ L2*X(t-2) + L3*X(t-3) + random_error
arma_model30 = sm.tsa.ARMA(data, (3,0)).fit()
print arma_model30.params
print arma_model30.aic, arma_model30.bic, arma_model30.hqic

# Plot model residuals
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
resid = arma_model30.resid
ax = resid.plot(ax=ax)

# Test null hypothesis
print stats.normaltest(resid)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q',ax=ax,fit=True)

plt.show()