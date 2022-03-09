import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_df(df, columnsax1=['Usage'], columnsax2=['AirTemp'], plot_type='plot', 
        time_range=('2018-01-01 16:00:00', '2018-04-12 16:00:00'), roll=1,
        savefig=False, exportation_name='plot_df'):

    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(15,10))
    fig.patch.set_facecolor('white')
    start_idx = df[df['DateTime'] == pd.Timestamp(time_range[0])].index[0]
    end_idx = df[df['DateTime'] == pd.Timestamp(time_range[1])].index[0]
    X = df.loc[start_idx:end_idx]['DateTime']
    
    y1 = df.loc[start_idx:end_idx][columnsax1].rolling(roll, min_periods=1).mean()
    lines1 = ax1.plot(X, y1, linewidth=1.5)
    ax1.legend(lines1, columnsax1)

    if len(columnsax2)>0:
        ax2 = ax1.twinx()
        y2 = df.loc[start_idx:end_idx][columnsax2].rolling(roll).mean()
        lines2 = ax2.plot(X,y2, color='red', linewidth=1.5)
        ax2.legend(lines2, columnsax2, loc='upper left')
    
    ax1.set_xlabel('Time')
    ax1.set_title(f'Period from {time_range[0]} to {time_range[1]}\n\
        (Rolling window size = {roll} minutes)')
    if savefig:
        plt.savefig(exportation_name)

    plt.show()


def stackplot(
        df, 
        time_range=('2018-04-05 16:00:00', '2018-04-12 16:00:00'), 
        columns =  ['Plugs', 
            'HeatingCoolingTotal', 'Ventilation', 
            'Lights', 'WaterHeater'],
        roll=200,
        savefig=False,
        exportation_name='stackplot'):

    start_idx = df[df['DateTime'] == pd.Timestamp(time_range[0])].index[0]
    end_idx = df[df['DateTime'] == pd.Timestamp(time_range[1])].index[0]
    df = df.loc[start_idx:end_idx].rolling(roll, on='DateTime').mean()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
    ax.stackplot(
        df['DateTime'], 
        [df[column] for column in columns], 
        labels=columns)
    ax.legend()
    ax.set_ylabel('Energy consumption (kW)')
    ax.set_title(f'Stackplot of the energy consumption from {time_range[0]} to {time_range[1]} \n\
                    (rolling window size = {roll} minutes)')

    if savefig:
        plt.savefig(exportation_name)
    plt.show()


def get_correlation(
    df, 
    col1='HeatingCoolingTotal', col2='AirTemp', 
    time_range=('2018-04-05 16:00:00', '2018-04-12 16:00:00')
    ):

    start_idx = df[df['DateTime'] == pd.Timestamp(time_range[0])].index[0]
    end_idx = df[df['DateTime'] == pd.Timestamp(time_range[1])].index[0]
    df = df.loc[start_idx:end_idx]
    
    return df[col1].corr(df[col2])


def plot_corr_matrix(df, sampling_title='every-minute sampling', savefig=False):
    '''Plots the correlation heatmap of the dataframe df'''
    corrmatrix1 = df.corr()
    fig, ax = plt.subplots(1, 1, figsize=(12,9))
    sns.heatmap(corrmatrix1, cmap='vlag', ax=ax)
    ax.set_title(f'Correlation matrix ({sampling_title})')
    if savefig:
        plt.savefig(f'corrmatrix_{sampling_title}')




def plot_pacf_acf(ts, roll=1, sampling_step=1, lags=30, savefig=False):
    """
    Plot PACF and ACF for AutoReg and ARIMA
    """

    start, end = ts.index[0], ts.index[-1]
    n=len(ts)
    ts = ts.rolling(roll).mean().iloc[
            np.arange(0,n,sampling_step)].fillna(0)

    fig1 = plot_acf(ts, lags=lags, alpha=.05)
    fig1.suptitle(f'Autocorrelation, start={start}, end={end}')
    if savefig:
        plt.savefig('acf')
    
    fig2 = plot_pacf(ts, lags=lags, alpha=.05, method='ywm')
    fig2.suptitle(f'Partial Autocorrelation, start={start}, end={end}')
    if savefig:
        plt.savefig('pacf')
    
    plt.show()

