#%%
import sys
print(sys.version)

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
#from autots import AutoTS, model_forecast

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# %%
data_df = pd.read_csv('./All_data_zone2.csv', index_col='Unnamed: 0')
# %%
data_df['Date & Time'] = data_df['Date & Time'].apply(
    lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
data_df.iloc[0]['Date & Time']
# %%
#date of activation of the sensors
starting_date = data_df[data_df['Usage [kW]']!=0]['Date & Time'].iloc[0]
print('Monitoring started at: ', starting_date)

#dropping null measures (before the activation of the sensors)
data_df.drop(data_df[data_df['Date & Time'] < starting_date].index, inplace=True)
data_df.head()
# %%

def plot(df, columnsax1=['Usage [kW]'], columnsax2=[], plot_type='plot', 
        time_range=('2018-04-05 16:00:00', '2018-04-12 16:00:00'), roll=1):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15,10))
    start_idx = df[df['Date & Time'] == pd.Timestamp(time_range[0])].index[0]
    end_idx = df[df['Date & Time'] == pd.Timestamp(time_range[1])].index[0]
    X = df.loc[start_idx:end_idx]['Date & Time']
    
    y1 = df.loc[start_idx:end_idx][columnsax1].rolling(roll).mean()
    lines1 = ax1.plot(X,y1)
    
    ax2 = ax1.twinx()
    y2 = df.loc[start_idx:end_idx][columnsax2].rolling(roll).mean()
    lines2 = ax2.plot(X,y2, color='red')
    
    ax1.legend(lines1, columnsax1)
    ax2.legend(lines2, columnsax2, loc='upper left')
    ax1.set_xlabel('Time')
    plt.show()

# %%
plot(data_df, 
    columnsax1=['Total Zone 2 [kW]'], 
    columnsax2=['Direct_Solar_Flux'],   
    time_range=('2018-04-06 ', '2018-06-05 08:00:00'), 
    roll=10000)


# %%
model = ARIMA(data_df, order=(0, 1, 1)) 
results_ARIMA = model.fit()
# %%
# 1,1,2 ARIMA Model
model = statsmodels.tsa.arima.model.ARIMA(data_df, order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# %%
statsmodels.tsa.ar_model
# %%
