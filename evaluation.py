import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns

from sklearn import metrics

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet


def compute_results(df_true, preds, column='Usage', model_type='prophet', print_results=True):
    """
    Computes y_pred and y_true based on preds and df_true,
    computes and prints metrics, adapting to the output type 
    of the different models.
    
    Parameters
    -------
    df_true (dataframe): real values
    preds (Series, dataframe, depends on the model used): predictions
    column (str): variable to predict
    model (string): name of the model used
    print (bool): is True, prints the results
    
    Returns
    -------
    results (dict): dictionnary of regression metrics and their values
    """

    available_models = ('autoreg', 'arima', 'sarimax', 'prophet')
    if model_type not in available_models: 
        raise KeyError(f'Model should be in {available_models}')
    
    if model_type in ('autoreg', 'arima', 'sarimax'):
        y_pred = preds.values
        dates = preds.index
        start, end = dates[0], dates[-1]

    elif model_type=='prophet':
        y_pred = np.array(preds['yhat'])
        dates = preds['ds']
        start, end = dates.iloc[0], dates.iloc[-1]

    y_true = np.array(df_true.set_index('DateTime').loc[dates][column])
    
    #compute metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 
    mse = metrics.mean_squared_error(y_true, y_pred) 
    #mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    results = {
        'explained_variance': round(explained_variance,5),  
        #'mean_squared_log_error': round(mean_squared_log_error,5),
        'r2': round(r2,5),
        'MAE': round(mean_absolute_error,5),
        'MSE': round(mse,5),
        'RMSE': round(np.sqrt(mse),5)
    }
    
    if print_results:
        print(f'Evaluation metrics for the model {model_type} \n \
        From {start} to {end} \n \
        {results}.')
    
    return results