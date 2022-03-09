#predictor
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima

import preprocessing_utils as prep

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation

from importlib import reload



# AUTOREG
def predictions_autoreg(df, column=['Usage'], 
    start='2018-07-20 00:00:00', 
    m='2018-07-28 00:00:00', 
    end='2018-08-01 00:00:00', 
    roll=60, sampling_step=60, 
    lags = 50,
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
        'AirTemp'
        #'rh',
        #'pres',
        #'rain',
        #'Weekend'
        ], 
    exogenes=False
    ):

    '''
    Returns the predictions between the dates m and end.

    Parameters
    -------
    df (dataframe) : the data
    start (string): start date ('YYYY-MM-DD HH:MM:SS')
    m (string): end date of the train set, beginning of the test set
    end (string): end date of the test set
    roll (int): smoothing window
    sampling_step (int): sampling step. 
        Keep one value every sampling_step values
    lags (int): previous values to consider
    exogenes_columns (list of str): the names of the exogenous columns to consider
    exogenes (bool): if the model uses exogenes or not 

    Returns
    -------
    fit_autoreg: the Autoreg model
    ts_pred: Series of predictions, from m to end. 
    '''

    #create a series object
    ts = df.set_index('DateTime')[column]
    exog = df.set_index('DateTime')[exogenes_columns]

    start_timestamp = prep.str_to_date(start)
    m_timestamp = prep.str_to_date(m)
    diff = m_timestamp - start_timestamp
    m2_timestamp = m_timestamp + timedelta(hours=1) #to have a clear separation between train values and pred values
    m2 = str(m2_timestamp)
    end_exog = str(m2_timestamp + diff) #to fit the exog pred values with the exog train values

    ts_train = ts[start:m].fillna(0)
    exog_train = exog[start:m].fillna(0)
    exog_pred = exog[m2:end_exog].fillna(0)
  

    n = len(exog_train)
    p = len(exog_pred)

    ts_train_sampled = ts_train.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)
    exog_train_sampled = exog_train.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)
    exog_pred_sampled = exog_pred.rolling(roll).mean().iloc[
            np.arange(0,p,sampling_step)].fillna(0)
    #exog_pred_sampled = exog_pred_sampled.iloc[1: , :]
    exog_oos = exog_pred_sampled[:n]

    

    if exogenes:
        model = AutoReg(ts_train_sampled,
                        lags = lags,
                        exog=exog_train_sampled)
    else : 
        model = AutoReg(ts_train_sampled, 
                        lags = lags)

    print('Autoreg trainig...')
    fit_autoreg = model.fit()

    print('Prediction...')
    
    if exogenes: 
        ts_pred = fit_autoreg.predict(m2, end, exog=exog_pred_sampled, exog_oos = exog_oos)
    else:
        ts_pred = fit_autoreg.predict(m2, end)

    print('Completed!')

    fig, ax = plt.subplots(1,1, figsize=(30,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Usage [kW]')
    ax.set_title(f'Autoreg \n \
            roll={roll}, sampling_step={sampling_step}')
    ax.plot(ts[start:end].rolling(roll).mean(), linewidth=1.5)
    ax.plot(ts_pred, color='blue', linewidth=1.5)
    plt.savefig('Autoreg_plot')
    plt.show()

    return fit_autoreg, ts_pred

# ARIMA
def predictions_arima(df, column=['Usage'], 
        start='2018-07-20 00:00:00', 
        m='2018-07-28 00:00:00', 
        end='2018-08-01 00:00:00', 
        roll=60, sampling_step=60, 
        order=(1,0,0)
    ):

    '''
    Returns the predictions between the dates m and end.

    Parameters
    -------
    df (dataframe) : the data
    start (string): start date ('YYYY-MM-DD HH:MM:SS')
    m (string): end date of the train set, beginning of the test set
    end (string): end date of the test set
    roll (int): smoothing window
    sampling_step (int): sampling step. 
        Keep one value every sampling_step values
    order (tuple of 3 ints): ARIMA order (p,d,q)

    Returns
    -------
    fit_arima: the ARIMA model
    ts_pred: Series of predictions, from m to end. 
    '''

def predictions_arima(df, column=['Usage'], 
        start='2018-07-20 00:00:00', 
        m='2018-07-28 00:00:00', 
        end='2018-08-01 00:00:00', 
        roll=60, sampling_step=60, 
        order=(1,0,0)
    ):

    '''
    Returns the predictions between the dates m and end.

    Parameters
    -------
    df (dataframe) : the data
    start (string): start date ('YYYY-MM-DD HH:MM:SS')
    m (string): end date of the train set, beginning of the test set
    end (string): end date of the test set
    roll (int): smoothing window
    sampling_step (int): sampling step. 
        Keep one value every sampling_step values
    order (tuple of 3 ints): SARIMAX order (p,d,q)
    tronc_s_order (tuple of 3 ints): SARIMAX seasonal order (P,D,Q),
        without the seasonality 

    Returns
    -------
    fit_arima: the ARIMA model
    ts_pred: Series of predictions, from m to end. 
    '''

    #create a series object
    ts = df.set_index('DateTime')[column]

    ts_train = ts[start:m]

    s = 24*60//sampling_step #daily seasonality

    n = len(ts_train)
    m2 = str(prep.str_to_date(m) + timedelta(hours = 1))

    ts_train_sampled = ts_train.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)
    

    
    model = ARIMA(ts_train_sampled, 
                order=order
            )

    print('ARIMA trainig...')
    fit_arima = model.fit()

    print('Prediction...')
    
    
    ts_pred = fit_arima.predict(m2, end)

    print("ts_pred : ", len(ts_pred), ts_pred.head)

    print('Completed!')

    fig, ax = plt.subplots(1,1, figsize=(30,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Usage [kW]')
    ax.set_title(f'ARIMA \n (p, d, q) = {order} \n \
            roll={roll}, sampling_step={sampling_step}')
    ax.plot(ts[start:end].rolling(roll).mean(), linewidth=1.5)
    ax.plot(ts_pred, color='blue', linewidth=1.5)
    plt.savefig('ARIMA_plot')
    plt.show()

    return fit_arima, ts_pred



# SARIMAX
def predictions_sarimax(df, column=['Usage'], 
        start='2018-07-20 00:00:00', 
        m='2018-07-28 00:00:00', 
        end='2018-08-01 00:00:00', 
        roll=60, sampling_step=60, 
        order=(1,0,0), tronc_s_order=(1,1,0), 
        seasonality = 24,
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
        ], 
        exogenes=True
    ):

    '''
    Returns the predictions between the dates m and end.

    Parameters
    -------
    df (dataframe) : the data
    start (string): start date ('YYYY-MM-DD HH:MM:SS')
    m (string): end date of the train set, beginning of the test set
    end (string): end date of the test set
    roll (int): smoothing window
    sampling_step (int): sampling step. 
        Keep one value every sampling_step values
    order (tuple of 3 ints): SARIMAX order (p,d,q)
    tronc_s_order (tuple of 3 ints): SARIMAX seasonal order (P,D,Q),
        without the seasonality
    seasonality (int): the seasonality of the model
    exogenes_columns (list of str): the names of the exogenous columns to consider
    exogenes (bool): if the model uses exogenes or not 

    Returns
    -------
    fit_sarimax: the SARIMAX model
    ts_pred: Series of predictions, from m to end. 
    '''

    #create a series object
    ts = df.set_index('DateTime')[column]
    exog = df.set_index('DateTime')[exogenes_columns]

    ts_train = ts[start:m]
    exog_train = exog[start:m]
    
    m2 = str(prep.str_to_date(m) + timedelta(hours = 1))
    exog_pred = exog[m2:end]


    s = seasonality*60//sampling_step #weekly seasonality

    seasonal_order = (tronc_s_order[0], tronc_s_order[1], 
                        tronc_s_order[2], s)

    n = len(ts_train)
    p = len(exog_pred)

    ts_train_sampled = ts_train.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)
    exog_train_sampled = exog_train.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)
    exog_pred_sampled = exog_pred.rolling(roll).mean().iloc[
            np.arange(0,p,sampling_step)].fillna(0)
    #exog_pred_sampled = exog_pred_sampled.iloc[1: , :]


    if exogenes:
        model = SARIMAX(ts_train_sampled, 
                    order=order,
                    seasonal_order=seasonal_order,
                    exog=exog_train_sampled)
    else : 
        model = SARIMAX(ts_train_sampled, 
                    order=order,
                    seasonal_order=seasonal_order)
    
    #to find the orders to use
    #model_autoarima = auto_arima(ts_train_sampled, seasonal=True,m=24)
    #print("model auto : ", model_autoarima)

    print('SARIMAX trainig...')
    fit_sarimax = model.fit()

    print('Prediction...')
    
    if exogenes: 
        ts_pred = fit_sarimax.predict(m2, end, exog=exog_pred_sampled)
    else:
        ts_pred = fit_sarimax.predict(m2, end)


    print('Completed!')

    fig, ax = plt.subplots(1,1, figsize=(30,10))
    ax.set_xlabel('Date')
    ax.set_ylabel('Usage [kW]')
    ax.set_title(f'SARIMAX \n (p, d, q) = {order} \n \
            (P, D, Q), s = {seasonal_order}, {s} \n \
            roll={roll}, sampling_step={sampling_step}')
    ax.plot(ts[start:end].rolling(roll).mean(), linewidth=1.5)
    ax.plot(ts_pred, color='blue', linewidth=1.5)
    plt.savefig('SARIMAX_plot')
    plt.show()

    return fit_sarimax, ts_pred

# PROPHET
def predictions_prophet(df, column=['Usage'], 
        start='2018-07-15 00:00:00', 
        m='2018-07-23 00:00:00', 
        end='2018-07-26 00:00:00', 
        roll=1, sampling_step=1,
        include_history=True,
        changepoint_prior_scale=0.05,
        changepoint_range=0.8, 
        add_changepoints=False,
        seasonality_prior_scale=10,
        yealy_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        weekend_seasonality=False,
        fourier_order=5,
        exogenes=True,
        regressors_columns = [
            'Weekday',
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
            #'Weekend'
        ]
        ):

    '''
    Returns the Prophet predictions between the dates m and end.

    Parameters
    -------
    df (dataframe) : the data
    start (string): start date ('YYYY-MM-DD HH:MM:SS')
    m (string): end date of the train set, beginning of the test set
    end (string): end date of the test set
    roll (int): smoothing window
    sampling_step (int): sampling step. 
        Keeps one value every sampling_step values
    include_history (bool): if True, the future variable will include the training values
    changepoint_prior_scale (float between 0 and 1): adjusts the trend flexibility
    changepoint_range (float between 0 and 1): the range of the time series in which 
        the changepoints are included.
        Should be at 0.8; not more than 0.95
    add_changepoints (bool): if True, it adds extremal changepoints
    seasonality_prior_scale (int): controls the flexibility of the seasonality
    yearly seasonlity (bool): includes (or not) a yearly seasonality
        Should be False
    weekly_seasonality (bool): includes (or not) a weekly seasonality
    daily_seasonality (bool): includes (or not) a daily seasonality
    weekend_seasonlity (bool): includes (or not) a seasonality based on the difference 
        between weekdays and weekends
    fourier_order: the fourier order for the seasonalities created by hand
    exogenes (bool): if the model uses exogenes or not 
    regressor_columns (list of str): the names of the exogenous columns to consider

    

    Returns
    -------
    prophet_model: the model
    forecast: the forecast dataframe (also contains additional data)

    '''
    columns = ['DateTime'] + column
    prophet_df = df.fillna(0)
    prophet_df.rename({'DateTime':'ds', column[0]:'y'}, inplace=True, axis=1)

    if weekend_seasonality:
        prophet_df['on_weekday'] = prophet_df['ds'].apply(prep.is_weekday) #add columns to distinguish weekdays from weekends
        prophet_df['on_weekend'] = ~prophet_df['ds'].apply(prep.is_weekday)

    start_timestamp = prep.str_to_date(start)
    m_timestamp = prep.str_to_date(m)
    end_timestamp = prep.str_to_date(end)

    used_mask = (prophet_df['ds']>=start_timestamp) & (prophet_df['ds']<end_timestamp)
    train_mask = (prophet_df['ds']>=start_timestamp) & (prophet_df['ds']<m_timestamp)
    test_mask = (prophet_df['ds']>=m_timestamp) & (prophet_df['ds']<end_timestamp)
 
    prophet_df_used = prophet_df.loc[used_mask].reset_index(drop=True)
    prophet_df_train = prophet_df.loc[train_mask]
    #prophet_df_test = prophet_df.loc[test_mask]
    #n = len(prophet_df_train)
    prophet_df_train['y'] = prophet_df_train['y'].rolling(roll).mean()

    ## Add Changepoints at hours with extremal values
    diff = m_timestamp-start_timestamp
    time = start
    changepoints_list = []

    while (diff.total_seconds()>0):
        if (prep.str_to_date(time).hour==6 or prep.str_to_date(time).hour==14):
                changepoints_list.append(time)
        time = str(prep.str_to_date(time) + timedelta(hours = 1))
        diff -= timedelta(hours=1)

    if add_changepoints:
        prophet_model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yealy_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            changepoint_range=changepoint_range,
            changepoints=changepoints_list
            )

    else : 
        prophet_model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yealy_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            changepoint_range=changepoint_range
        )
    
    ## Seasonality
    if weekend_seasonality:
        prophet_model.add_seasonality(name='weekday', period=1, fourier_order=fourier_order, condition_name='on_weekday')
        prophet_model.add_seasonality(name='weekend', period=1, fourier_order=fourier_order, condition_name='on_weekend')

    if daily_seasonality:
        prophet_model.add_seasonality(name='daily', period=1, fourier_order=fourier_order)
    
    ## Exogenes
    if exogenes:
        for reg in regressors_columns: 
            prophet_model.add_regressor(reg)

    print('Fitting Prophet...')
    prophet_model.fit(prophet_df_train)

    diff = end_timestamp - m_timestamp
    days, seconds = diff.days, diff.seconds 
    periods = (60*24*days + seconds//60)//sampling_step 
    periods2 = diff.total_seconds()//60//sampling_step

    print("périodes : ", periods, periods2)

    if sampling_step not in (1,15,30,60):
        raise ValueError(f'sampling_step should be in (1,15,30,60)')
    if sampling_step==1:
        freq='1min'
    elif sampling_step==15:
        freq='15min'
    elif sampling_step==30:
        freq='30min'
    elif sampling_step==60:
        freq='H'
    
    #specifiy the frequency. Prophet does not detect it himself
    future = prophet_model.make_future_dataframe(
       periods=periods, freq=freq, include_history=include_history)

    if weekend_seasonality:
        future['on_weekday'] = future['ds'].apply(prep.is_weekday)
        future['on_weekend'] = ~future['ds'].apply(prep.is_weekday)
    
    
    if exogenes:
        for reg in regressors_columns:
            future = future.join(prophet_df_used[reg])

    print('Prediction...')
    forecast = prophet_model.predict(future)
    
    print('Completed!')

    fig = prophet_model.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), prophet_model, forecast)
    fig.savefig('prophet_plot_with_changepoints')

    prophet_model.plot(forecast).savefig('prophet_plot')
    prophet_model.plot_components(forecast).savefig('prophet_plot_components')


    return prophet_model, forecast

#reload(prep)