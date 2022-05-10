import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def str_to_date(s):
    """Converts a string to datetime"""
    return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

def is_weekday(ds):
    """For a given datetime ds, returns if it is a weekday or not"""
    return ds.weekday() <= 4 

def preprocess(data_df):
    """Data cleaning and preprocessing"""

    print('Converting dates to Datetime...')
    data_df['Date & Time'] = data_df['Date & Time'].apply(
        lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    print('Dropping measures before the activation of the sensors...')
    starting_date = data_df[data_df['Usage [kW]']!=0]['Date & Time'].iloc[0]
    data_df.drop(data_df[data_df['Date & Time'] < starting_date].index, inplace=True)

    print('Dropping Total Zone 2 [kW] and Generation [kW]')
    data_df = data_df.drop(['Total Zone 2 [kW]', 'Generation [kW]'], axis=1)

    print('Renaming columns...')
    data_df = data_df.rename(columns={
        'Date & Time' : 'DateTime', 
        'Usage [kW]': 'Usage', 
        'SUM [kW]': 'SUM',
        'Plugs [kW]':'Plugs', 
        'Heating / Cooling Total [kW]':'HeatingCoolingTotal', 
        'Ventilation [kW]' : 'Ventilation',
        'Heaters corridor [kW]':'HeatersCorridor', 
        'Lights  [kW]' : 'Lights', 
        'Water heater [kW]' : 'WaterHeater',
        'Heaters Toilets [kW]' : 'HeatersToilets', 
        'Time in sec' : 'TimeInSec', 
        'Time of day' : 'TimeOfDay' })

    print('Switching signs...')
    neg_columns = ['SUM', 'Plugs', 'HeatingCoolingTotal', 
        'Ventilation', 'HeatersCorridor', 'Lights', 'WaterHeater', 
        'HeatersToilets', ]
    for col in neg_columns:
        data_df[col] = data_df[col].apply(lambda x : -x)

    print('Adding Weekend column...')
    data_df['Weekend'] = data_df['Weekday'].apply(lambda x: 1 if x in (5,6) else 0)

    print('Preprocessing completed!')

    return data_df
