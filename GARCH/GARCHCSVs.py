#!/usr/bin/env python3
# The shebang above is used to tell the system this is a Python script

# Import necessary libraries
import numpy as np
from arch import arch_model
import yfinance as yf
import concurrent.futures
import os



def download_data(ticker,start_date,end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change() * 100
    data = data[1:]
    data=data.drop(['High','Low','Close','Volume','Open'],axis=1)
    return data


# Helper function to perform forecasting for a single model
def forecast_model(data, model_type, horizon,starting):
    if model_type == 'GARCH':
        model = arch_model(data, vol="Garch", p=1, q=1, dist='skewt')
    elif model_type == 'EGARCH':
        model = arch_model(data, vol="EGarch", p=1, q=1, dist='skewt')
    elif model_type == 'GJR':
        model = arch_model(data, vol="Garch", p=1, o=1, q=1, dist='skewt')
    elif model_type == 'TARCH':
        model = arch_model(data, vol="Garch", p=1, o=1, q=1, dist='skewt',power=1)
    else:
        raise KeyError

    results = model.fit(update_freq=5, disp='off')
    forecast = results.forecast(start=starting, horizon=horizon, method='simulation', reindex=True)

    return {
        'residuals': results.resid[-1],
        'cvolatility': np.sqrt(results.conditional_volatility)[-1],
        'forecast': np.sqrt(forecast.variance).values[-1,:]
    }


def forecasting(data, start_start_date, start_end_date, horizon):
    data_end_index = data.index[data.index >= start_end_date]
    data_start_index = data.index[data.index >= start_start_date]
    lth = len(data_end_index)
    for i in range(lth):
        for model_type in ['GARCH', 'EGARCH', 'TARCH', 'GJR']:
            result = forecast_model(data["Returns"][data_start_index[i]:data_end_index[i]], model_type, horizon,data_end_index[i])
            data.loc[data_end_index[i], f'Residuals-{model_type}'] = result['residuals']
            data.loc[data_end_index[i], f'CVolatility-{model_type}'] = result['cvolatility']
            for t in range(1, horizon + 1):
                data.loc[data_end_index[i], f'Forecast-{model_type}(t+{t})'] = result['forecast'][t - 1]

    print('Ending Forecasting')

    return data

def process_ticker(ticker, data, start_start_date, start_end_date, horizon):
    folder_path= f'./LONGGARCHFOR{horizon}'
    os.makedirs(folder_path, exist_ok=True)
    processed_data = forecasting(data, start_start_date, start_end_date, horizon)
    processed_data.to_csv(f'./LONGGARCHFOR{horizon}/{ticker}.csv')




def main():

    print(' Downloading data ')

    ticker_list = ['RWR', 'XLB', 'XLI', 'XLY', 'XLP', 'XLE', 'XLF', 'XLU', 'XLV', 'XLK']
    start_date = "2002-01-01"
    end_date = "2023-10-30"
    horizon = 10
    start_start_date = '2002-01-01'
    start_end_date = '2007-01-01'

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_ticker, ticker, download_data(ticker,start_date,end_date), start_start_date, start_end_date,horizon)
            for ticker in ticker_list]
        concurrent.futures.wait(futures)


# This if statement makes sure that the script runs only if it is executed as the main program
if __name__ == "__main__":
    main()
