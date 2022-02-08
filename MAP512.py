#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#%%
plt.style.use('seaborn')
#%%
data_df = pd.read_csv('./All_data_v1.csv', index_col='Unnamed: 0')
data_df['Date & Time'] = data_df['Date & Time'].apply(
    lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
#%% 


# %%

def plot_df(df, columnsax1=['Usage [kW]'], columnsax2=[], plot_type='plot', 
        time_range=('2018-04-05 16:00:00', '2018-04-12 16:00:00'), roll=1):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15,10))
    fig.patch.set_facecolor('white')
    start_idx = df[df['Date & Time'] == pd.Timestamp(time_range[0])].index[0]
    end_idx = df[df['Date & Time'] == pd.Timestamp(time_range[1])].index[0]
    X = df.loc[start_idx:end_idx]['Date & Time']
    
    y1 = df.loc[start_idx:end_idx][columnsax1].rolling(roll).mean()
    lines1 = ax1.plot(X, y1, linewidth=1.5)
    ax1.legend(lines1, columnsax1)

    if len(columnsax2)>0:
        ax2 = ax1.twinx()
        y2 = df.loc[start_idx:end_idx][columnsax2].rolling(roll).mean()
        lines2 = ax2.plot(X,y2, color='red', linewidth=1.5)
        ax2.legend(lines2, columnsax2, loc='upper left')
    
    ax1.set_xlabel('Time')
    plt.show()

# %%
#We fix the dates we will use: '2018-06-09', '2018-08-09'
plot_df(data_df, 
    columnsax1=['Usage [kW]'], 
    #columnsax2=['Direct_Solar_Flux'],   
    time_range=('2018-06-09', '2018-08-09'), 
    roll=100)

#%%
plot_df(data_df, 
    columnsax1=['Usage [kW]'], 
    #columnsax2=['Direct_Solar_Flux'],   
    time_range=('2018-06-11', '2018-06-18'), 
    roll=15)

#%%
ts = data_df.set_index('Date & Time')['Usage [kW]']
ts.head()


# %%
start, m, end = '2018-06-09 00:00:00','2018-07-09 00:00:00','2018-07-23 00:00:00'
#start, m, end = '2018-06-11 00:00:00','2018-06-14 00:00:00','2018-06-16 00:00:00'
ts_train, ts_test = ts[start:m], ts[m:end]

#%%
fig1 = plot_pacf(ts_train, lags=30, alpha=.01, method='ywm')
fig1.suptitle(f'Partial Autocorrelation, start={start}, end={end}')
plt.savefig('pacf')
fig2 = plot_acf(ts_train, lags=30, alpha=.01)
fig2.suptitle(f'Autocorrelation, start={start}, end={end}')
plt.savefig('acf')
plt.show()
#%%
n = len(ts_train)
p, d, q = 2, 0, 0
roll = 50
step = 100
P, D, Q, s = 1, 1, 1, 1440//step*7
model = SARIMAX(ts_train.rolling(roll).mean()[np.arange(0,n,step)], 
            order=(p,d,q),
            seasonal_order=(P,D,Q,s)) 
fit_SARIMAX = model.fit()
ts_pred = fit_SARIMAX.predict(m, end)

#%%
fig, ax = plt.subplots(1,1, figsize=(60,20))
ax.set_xlabel('Date')
ax.set_ylabel('Usage [kW]')
ax.set_title(f'SARIMAX \n (p, d, q) = ({p}, {d}, {q}) \n \
        (P, D, Q, s) = ({P}, {D}, {Q}, {s}) \n \
        roll={roll}, step={step}')
ax.plot(ts[start:end].rolling(roll).mean(), linewidth=1.5)
ax.plot(ts_pred, color='blue', linewidth=1.5)
plt.savefig('SARIMAX_plot')
plt.show()

# %%
ts_pred.head()
# %%
ts
