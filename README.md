# DrahiX Power Consumption Forecasting

## Requirements

- fbprophet=0.4
- matplotlib=2.2.2
- numpy=1.19.5
- pandas=0.25.0
- pystan=2.17.1.0
- scikit-learn=0.20.1
- scipy=1.1.0
- seaborn=0.7.1
- statsmodels=0.12.2

## Models

- Autoregressive model
- ARIMA
- SARIMAX
- Prophet

## Dataset

```All_data_zone2.csv```

## Python scripts
- ```preprocessing_utils.py``` contains a preprocessing function for the dataset
- ```plot_utils.py``` contains all the plot functinos used in the project
- ```evaluation.py``` contains the main evaluation function
- ```predictor_functs``` contains the prediction functions. Ther is one function for per model, each of which can be parametrized
- ```main.py``` is a notebook-like Python script. All the results obtained in the report are accessible frm this file.

## Quick start
- Place the file ```All_data_zone2.csv``` in the current directory.
- Execute the importation and the preprocessing cells.
- Execute one of the following cells to:
  - plot the time series
  - plot the correlation matrix
  - fit, predict, plot and asses each model
