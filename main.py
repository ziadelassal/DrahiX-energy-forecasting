#%%
import imp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet

import preprocessing_utils as prep
from plot_utils import *
import predictor_funcs as pf
import evaluation as eval

pd.set_option('display.max_columns', 500)
plt.style.use('seaborn')

from importlib import reload

PATH = './All_data_zone2.csv'
START = '2019-09-09 00:00:00' 
M = '2019-09-26 00:00:00'
END = '2019-10-01 00:00:00'


#%% Data importation and preprocessing
data_df = pd.read_csv(PATH, index_col='Unnamed: 0')
data_df = prep.preprocess(data_df)


#%% Correlation Matrix
df_1d_freq = data_df.rolling(1440).mean().iloc[np.arange(0,len(data_df), 1440)]
df_1h_freq = data_df.rolling(60).mean().iloc[np.arange(0,len(data_df), 60)]
plot_corr_matrix(data_df, savefig=False)
plot_corr_matrix(df_1d_freq, 'every-day sampling', savefig=False)
plot_corr_matrix(df_1h_freq, 'every-hour sampling', savefig=False)


#%% Stackplot
stackplot(data_df, 
    time_range=('2019-10-01 00:00:00', '2019-12-01 00:00:00'), 
    columns =  [ 'WaterHeater','Ventilation','Plugs','Lights', 
         'HeatingCoolingTotal'],
    roll=1440,
    savefig=False,
    exportation_name='stackplot')

#%% Plot
plot_df(data_df, 
    columnsax1=['Lights'], 
    columnsax2=['Global_Solar_Flux']  ,
    time_range=('2019-01-01', '2020-10-01'), 
    roll=10080,
    savefig=False, 
    exportation_name='plot_nov2019feb2020_Heating_Airtemp')


#%% PACF ACF
ts = data_df.set_index('DateTime')['Usage'][START:M]
plot_pacf_acf(ts, roll=60, sampling_step=60,lags=120, savefig=False)


#%% AUTOREG
fit_autoreg, ts_pred = pf.predictions_autoreg(
        data_df,
        start=START, 
        m=M, 
        end=END,
        lags=24,
        exogenes_columns = [
            #'Weekday',
            #'Direct_Solar_Flux',
            #'Diffuse_Solar_Flux',
            #'Global_Solar_Flux',
            #'Downwelling_IR_Flux',
            #'SZA',
            #'SAA',
            #'ws',
            #'wd',
            'AirTemp',
            #'rh',
            #'pres',
            #'rain',
            'Weekend'
        ], 
        exogenes=True)

eval.compute_results(data_df, ts_pred, column='Usage', model_type='autoreg', print_results=True)


#%%ARIMA
fit_arima, ts_pred = pf.predictions_arima(
        data_df,
        start=START, 
        m=M, 
        end=END,  
        roll=60, 
        sampling_step=60,
        order=(48,1,0)
    )

eval.compute_results(data_df, ts_pred, column='Usage', model_type='autoreg', print_results=True)



# %% SARIMAX
fit_sarimax, ts_pred = pf.predictions_sarimax(
        data_df,
        start=START, 
        m=M, 
        end=END,  
        roll=60, 
        sampling_step=60,
        order=(1,1,0), tronc_s_order=(2,1,0),
        seasonality=0,
        exogenes=False,
        exogenes_columns = [
            'Weekday',
            #'TimeOfDay',
            #'Direct_Solar_Flux',
            #'Diffuse_Solar_Flux',
            #'Global_Solar_Flux',
            'Downwelling_IR_Flux',
            #'SZA',
            #'SAA',
            #'ws',
            #'wd',
            'AirTemp',
            #'rh',
            #'pres',
            #'rain',
            'Weekend'
        ]
    )

eval.compute_results(data_df, ts_pred, column='Usage', model_type='sarimax', print_results=True)



#%% PROPHET
prophet_model, forecast = pf.predictions_prophet(
        data_df, 
        column=['Usage'], 
        start=START, 
        m=M, 
        end=END,  
        roll=60, 
        sampling_step=60,
        include_history=True,
        changepoint_prior_scale=0.001,
        daily_seasonality=True,
        weekly_seasonality=True,
        weekend_seasonality=True,
        fourier_order=15,
        exogenes=False,
        regressors_columns = [
            'Weekday',
            #'Direct_Solar_Flux',
            #'Diffuse_Solar_Flux',
            'Global_Solar_Flux',
            'Downwelling_IR_Flux',
            #'SZA',
            #'SAA',
            #'ws',
            #'wd',
            'AirTemp',
            'rh',
            #'pres',
            #'rain',
            #'Weekend'
        ]
        )

eval.compute_results(data_df, forecast, model_type='prophet', print_results=True)

