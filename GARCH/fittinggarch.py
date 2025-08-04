#!/usr/bin/env python3
# The shebang above is used to tell the system this is a Python script

# Import necessary libraries
import numpy as np
from joblib import Parallel, delayed
from arch import arch_model
import yfinance as yf
import math
import os

# Helper function to perform forecasting for a single model
def forecast_model(data, start_index, end_index, model_name, vol, p, q, o=None, dist='skewt', power=None, horizon=10):

    model = arch_model(data['log_returns'][start_index:end_index],
                       mean='Constant',
                       vol=vol,
                       p=p,
                       q=q,
                       o=o,
                       dist=dist,
                       power=power)
    results = model.fit(update_freq=5, disp='off')
    forecast = results.forecast(start=end_index, horizon=horizon, method='simulation', reindex=True)
    forecast_variances = forecast.variance.values[-1, :]
    forecast_values = np.sqrt(forecast_variances)
    return {
        'Forecasts': forecast_values,
        'Residuals': results.resid[-1],
        'CVolatility': np.sqrt(results.conditional_volatility)[-1]
    }



def forecasting5(data,start_start_date, start_end_date,horizon):

  data_end_index = data.index[data.index>=start_end_date]
  data_start_index= data.index[data.index>=start_start_date]
  lth= len(data_end_index)

  for i in range(0, lth):


    # GARCH models update every 1days

    garch_model = arch_model(data["log_returns"][data_start_index[i]:data_end_index[i]], vol="Garch", p=1, q=1,dist='skewt')
    garch_results = garch_model.fit(update_freq=5,disp='off')


    #print("------****--- Garch model fitted ---****------")

    egarch_model = arch_model(data["log_returns"][data_start_index[i]:data_end_index[i]], vol="EGarch", p=1, q=1,dist='skewt')
    egarch_results = egarch_model.fit(update_freq=5,disp='off')

    #print("------****--- EGarch model fitted ---****------")

    tarch_model = arch_model(data["log_returns"][data_start_index[i]:data_end_index[i]], vol="Garch", p=1, o=1, q=1,dist='skewt',power=1)
    tarch_results = tarch_model.fit(update_freq=5,disp='off')


    #print("------****--- T-Garch model fitted---****------")



    gjr_model = arch_model(data["log_returns"][data_start_index[i]:data_end_index[i]], vol="Garch", p=1, o=1, q=1,dist='skewt')
    arch_model(data["log_returns"][data_start_index[i]:data_end_index[i]], vol="Garch", p=1, o=1, q=1,dist='skewt')
    gjr_results = gjr_model.fit(update_freq=5,disp='off')


    #print("------****--- GJR-Garch model fitted ---****------")

    # GARCH models forecasting 1-day ahead and saving values in data

    forecast_garch = garch_results.forecast(start=data_end_index[i],horizon= horizon, method='simulation', reindex=True)
    data.loc[data_end_index[i], 'Residuals-Garch5'] = garch_results.resid[-1].copy()
    data.loc[data_end_index[i], 'CVolatility-Garch5'] = np.sqrt(garch_results.conditional_volatility)[-1].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+1)'] = np.sqrt(forecast_garch.variance).values[-1][0].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+2)'] = np.sqrt(forecast_garch.variance).values[-1][1].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+3)'] = np.sqrt(forecast_garch.variance).values[-1][2].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+4)'] = np.sqrt(forecast_garch.variance).values[-1][3].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+5)'] = np.sqrt(forecast_garch.variance).values[-1][4].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+6)'] = np.sqrt(forecast_garch.variance).values[-1][5].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+7)'] = np.sqrt(forecast_garch.variance).values[-1][6].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+8)'] = np.sqrt(forecast_garch.variance).values[-1][7].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+9)'] = np.sqrt(forecast_garch.variance).values[-1][8].copy()
    data.loc[data_end_index[i], 'Forecast-Garch(t+10)'] = np.sqrt(forecast_garch.variance).values[-1][-1].copy()


    forecast_egarch = egarch_results.forecast(start=data_end_index[i],horizon= horizon, method='simulation', reindex=True)
    data.loc[data_end_index[i], 'Residuals-EGarch5'] = egarch_results.resid[-1].copy()
    data.loc[data_end_index[i], 'CVolatility-EGarch5'] = np.sqrt(egarch_results.conditional_volatility)[-1].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+1)'] = np.sqrt(forecast_egarch.variance).values[-1][0].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+2)'] = np.sqrt(forecast_egarch.variance).values[-1][1].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+3)'] = np.sqrt(forecast_egarch.variance).values[-1][2].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+4)'] = np.sqrt(forecast_egarch.variance).values[-1][3].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+5)'] = np.sqrt(forecast_egarch.variance).values[-1][4].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+6)'] = np.sqrt(forecast_egarch.variance).values[-1][5].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+7)'] = np.sqrt(forecast_egarch.variance).values[-1][6].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+8)'] = np.sqrt(forecast_egarch.variance).values[-1][7].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+9)'] = np.sqrt(forecast_egarch.variance).values[-1][8].copy()
    data.loc[data_end_index[i], 'Forecast-EGarch(t+10)'] = np.sqrt(forecast_egarch.variance).values[-1][-1].copy()



    forecast_tarch = tarch_results.forecast(start=data_end_index[i],horizon=horizon , method='simulation', reindex=True)
    data.loc[data_end_index[i], 'Residuals-Tarch'] = tarch_results.resid[-1].copy()
    data.loc[data_end_index[i], 'CVolatility-Tarch'] = np.sqrt(tarch_results.conditional_volatility)[-1].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+1)'] = np.sqrt(forecast_tarch.variance).values[-1][0].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+2)'] = np.sqrt(forecast_tarch.variance).values[-1][1].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+3)'] = np.sqrt(forecast_tarch.variance).values[-1][2].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+4)'] = np.sqrt(forecast_tarch.variance).values[-1][3].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+5)'] = np.sqrt(forecast_tarch.variance).values[-1][4].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+6)'] = np.sqrt(forecast_tarch.variance).values[-1][5].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+7)'] = np.sqrt(forecast_tarch.variance).values[-1][6].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+8)'] = np.sqrt(forecast_tarch.variance).values[-1][7].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+9)'] = np.sqrt(forecast_tarch.variance).values[-1][8].copy()
    data.loc[data_end_index[i], 'Forecast-Tarch(t+10)'] = np.sqrt(forecast_tarch.variance).values[-1][-1].copy()


    forecast_gjr = gjr_results.forecast(start=data_end_index[i],horizon= horizon, method='simulation', reindex=True)
    data.loc[data_end_index[i], 'Residuals-GJR'] = gjr_results.resid[-1].copy()
    data.loc[data_end_index[i], 'CVolatility-GJR'] = np.sqrt(gjr_results.conditional_volatility)[-1].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+1)'] = np.sqrt(forecast_gjr.variance).values[-1][0].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+2)'] = np.sqrt(forecast_gjr.variance).values[-1][1].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+3)'] = np.sqrt(forecast_gjr.variance).values[-1][2].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+4)'] = np.sqrt(forecast_gjr.variance).values[-1][3].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+5)'] = np.sqrt(forecast_gjr.variance).values[-1][4].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+6)'] = np.sqrt(forecast_gjr.variance).values[-1][5].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+7)'] = np.sqrt(forecast_gjr.variance).values[-1][6].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+8)'] = np.sqrt(forecast_gjr.variance).values[-1][7].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+9)'] = np.sqrt(forecast_gjr.variance).values[-1][8].copy()
    data.loc[data_end_index[i], 'Forecast-GJR(t+10)'] = np.sqrt(forecast_gjr.variance).values[-1][-1].copy()

  print('data_end',data_end_index[i])
  print('data_start',data_start_index[i])

  return data




# Your Python code to execute, for example:



def main():

    ticker_list = ['RWR', 'XLB', 'XLI', 'XLY', 'XLP', 'XLE', 'XLF', 'XLU', 'XLV', 'XLK']
    # Download stock data from Yahoo Finance
    start_date = "2011-01-01"
    end_date = "2023-10-30"

    horizon = 10
    # Assicurati che la cartella di destinazione esista
    folder_path = f'./filesusa{horizon}'
    os.makedirs(folder_path, exist_ok=True)
    start_start_date = '2011-01-01'
    start_end_date = '2015-01-01'

    for ticker in ticker_list:


        globals()["data_" + ticker] = yf.download(ticker, start=start_date, end=end_date)
        globals()["data_" + ticker]['Returns'] = globals()["data_" + ticker]['Adj Close'].pct_change() * 100
        globals()["data_" + ticker]['log_returns'] = (globals()["data_" + ticker]['Adj Close'].apply(
            lambda x: math.log(x) if math.log(x) else None) - globals()["data_" + ticker]['Adj Close'].shift().apply(
            lambda x: math.log(x) if math.log(x) else None)) * 100
        globals()["data_" + ticker] = globals()["data_" + ticker][1:]
        print(globals()[f"data_{ticker}"].columns)

        print('Starting fitting session ')
        globals()["file_" + ticker] = forecasting5(globals()["data_" + ticker], start_start_date, start_end_date,
                                                   horizon)
        globals()["file_" + ticker].to_csv(f'./files{horizon}/{ticker}')
        # Salva il DataFrame in un file CSV
        print('Starting saving session ')

        file_path = os.path.join(folder_path, f"{ticker}.csv")
        print('Ending saving session ')



# This if statement makes sure that the script runs only if it is executed as the main program
if __name__ == "__main__":
    main()
