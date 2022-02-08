import pandas as pd 
from datetime import datetime
import numpy as np


data_df = pd.read_csv('./All_data_zone2.csv', index_col='Unnamed: 0')

data_df['Date & Time'] = data_df['Date & Time'].apply(
    lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
data_df.iloc[0]['Date & Time']

#date of activation of the sensors
starting_date = data_df[data_df['Usage [kW]']!=0]['Date & Time'].iloc[0]
print('Monitoring started at: ', starting_date)

#dropping null measures (before the activation of the sensors)
data_df.drop(data_df[data_df['Date & Time'] < starting_date].index, inplace=True)

data_df.to_csv('All_data_v1.csv')