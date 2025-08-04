import json
import numpy as np
import numpy as np
import yfinance as yf
import concurrent.futures
import math
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forza l'uso della CPU
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband,HyperParameters

def garman_klass_volatility(high, low, close, open):
    """
    Calculate the Garman-Klass volatility estimator.

    Parameters:
    - high: Highest price for the trading day.
    - low: Lowest price for the trading day.
    - close: Closing price for the trading day.
    - open: Opening price for the trading day.

    Reurns:
    - The Garman-Klass volatility estimate for the day.
    """

    term1 = 0.5 *(np.log(high / low))** 2
    term2 = (2 * np.log(2) - 1) * (np.log(close / open)) ** 2
    return  np.sqrt(term1 - term2)*100

def download_data(ticker,start_date,end_date,start_garch_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    # add gold,wti,vix,s&p500 columns
    aux_data = pd.read_csv('./aux_data.csv')
    data['vix'] = aux_data['vix'].values
    data['s&p500'] = aux_data['s&p500'].values
    data['gold'] = aux_data['gold'].values
    data['brent'] = aux_data['brent'].values
    data['10ybond'] = aux_data['10ybond'].values
    data['2ybond'] = aux_data['2ybond'].values
    data['yield'] = aux_data['yield'].values

    # Compute returns
    data["Returns"] = data["Adj Close"].pct_change() * 100
    # Compute log_returns
    data['log_returns'] = (data['Adj Close'].apply(lambda x: math.log(x) if math.log(x) else None) - data[
        'Adj Close'].shift().apply(lambda x: math.log(x) if math.log(x) else None)) * 100
    data = data[1:]

    # Compute historical volatility
    data["Volatility"] = data["log_returns"].rolling(window=5).std()
    data["Volatility20"] = data["log_returns"].rolling(window=20).std()
    data["Volatility10"] = data["log_returns"].rolling(window=10).std()
    data['GKVol'] = garman_klass_volatility(data['High'], data['Low'], data['Close'], data['Open'])
    data = data[20:]
    data['open_to_close'] = data['Close'] - data['Open']
    data['high_to_low'] = data['High'] - data['Low']
    file_path = f'./GARCHFOR10/{ticker}.csv'  # Corrected the file path extension to .csv
    fileg = pd.read_csv(file_path)
    fileg.index = fileg['Date']

    fileg = fileg[start_garch_date:end_date]
    data['CVolGarch'] = fileg['CVolatility-GARCH'].values
    data['ResidualsGarch'] = fileg['Residuals-GARCH'].values
    data['ForGarch10'] = fileg['Forecast-GARCH(t+10)'].values
    data['CVolGJR'] = fileg['CVolatility-GJR'].values
    data['ResidualsGJR'] = fileg['Residuals-GJR'].values
    data['ForGJR10'] = fileg['Forecast-GJR(t+10)'].values
    data['CVolEGarch'] = fileg['CVolatility-EGARCH'].values
    data['ResidualsEGarch'] = fileg['Residuals-EGARCH'].values
    data['ForEGarch10'] = fileg['Forecast-EGARCH(t+10)'].values
    data['CVolTarch'] = fileg['CVolatility-TARCH'].values
    data['ResidualsTarch'] = fileg['Residuals-TARCH'].values
    data['ForTarch10'] = fileg['Forecast-TARCH(t+10)'].values

    return data

def splitting_data_test(data):

    train_data =data[:'2022-12-31']
    val_data = data['2023-01-01':]
    #test_data = data['2023-01-01':]

    return train_data,val_data

def normalization(train_data,val_data):
  train_mean=train_data.mean()
  train_std= train_data.std()

  train_data = (train_data - train_mean) / train_std
  val_data = (val_data - train_mean) / train_std
 # test_data = (test_data - train_mean) / train_std

  return train_data, val_data

def generate_sequences(data,window_size,target_column,predicted_horizon):
  X,y = [],[]
  for i in range(0,len(data) - window_size-predicted_horizon,predicted_horizon):
      X.append(data.values[i:i+window_size])
      y.append(data[target_column].values.flatten()[i+window_size+predicted_horizon])
  return np.array(X), np.array(y)

def preparing_data(data,window_size,target_column,forecast_horizon):
    train_data, val_data= splitting_data_test(data)
    train_mean=train_data.mean()
    train_std=train_data.std()
    train_data, val_data = normalization(train_data, val_data)

    X_train, y_train = generate_sequences(train_data, window_size, target_column, forecast_horizon)
    X_val, y_val = generate_sequences(val_data, window_size, target_column, forecast_horizon)

    return  X_train, y_train,X_val, y_val,train_mean,train_std

def build_model_lstm(hp):
    model = tf.keras.models.Sequential([
        # LSTM layers
        tf.keras.layers.LSTM(hp.Choice('units1lstm', [16, 32, 64, 128, 256, 512]), return_sequences=True),
        tf.keras.layers.LSTM(hp.Choice('units2lstm', [16, 32, 64, 128, 256, 512]), return_sequences=True),
        tf.keras.layers.LSTM(hp.Choice('units3lstm', [16, 32, 64, 128, 256, 512]), return_sequences=False),

        # Dense layer
        tf.keras.layers.Dense(hp.Choice('units1dense', [8, 16, 32,64,128]),
                              activation=hp.Choice('activation', ['relu', 'tanh'])),
        # Dense layer
        tf.keras.layers.Dense(hp.Choice('units1dense', [8, 16, 32,64,128]),
                              activation=hp.Choice('activation', ['relu', 'tanh'])),


        tf.keras.layers.Dense(units=1)  # 1 output unit---> future tenth day value for
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=hp_learning_rate),
        loss=keras.losses.MeanSquaredError(),  # Potrebbe essere più adatto per previsioni di serie temporali
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return model

def load_best_hyperparameters(ticker):
    tuner = Hyperband(
        build_model_lstm,
        objective='val_loss',
        max_epochs=50,  # o il numero di epoche che hai usato
        factor=4,       # o il fattore che hai usato
        directory=f'./TUNER/tuner_lstm/{ticker}',
        project_name='lstm-tuner',
    )

    best_trials = tuner.oracle.get_best_trials(num_trials=5)
    best_params = best_trials[0].hyperparameters.values
    # Definisci il percorso del file JSON dove vuoi salvare i dati
    json_file_path = f"./hp/best_hyperparameters_{ticker}.json"
    # Scrivi i dati nel file JSON
    with open(json_file_path, "w") as file:
        json.dump(best_params, file, indent=4)

    print(f"Migliori iperparametri per {ticker} salvati in {json_file_path}")

def initialize_model(ticker,data,window_size, target_column, forecast_horizon):
    tuner = Hyperband(
        build_model_lstm,
        objective='val_loss',
        max_epochs=40,  # o il numero di epoche che hai usato
        factor=3,  # o il fattore che hai usato
        directory=f'./TUNER/tuner_lstm/{ticker}',
        project_name='lstm-tuner',
    )

    best_trials = tuner.oracle.get_best_trials(num_trials=5)
    best_params = best_trials[0].hyperparameters.values
    hp = HyperParameters()
    list_names = ['units1lstm',
                  'units2lstm',
                  'units3lstm',
                  'units1dense',
                  'activation',
                  'learning_rate']

    for name in list_names:
        hp.Fixed(name, best_params[name])

    print(' Building model')
    model = build_model_lstm(hp)
    X_train, y_train, X_val, y_val,train_mean,train_std = preparing_data(data, window_size, target_column, forecast_horizon)
    model.build((None,) + X_train.shape[1:])
    history = model.fit(X_train, y_train, epochs=best_params['tuner/epochs'], batch_size=32)
    # Valutazione del modello
    loss = model.evaluate(X_val, y_val)
    # Utilizzo del modello per fare previsioni
    predictions = model.predict(X_val)
    # Converti in DataFrame

    results_df = pd.DataFrame({
        'Predictions': predictions.flatten()*train_std+train_mean,  # Usa flatten() se 'predictions' è un array 2D
        'Actual': y_val.flatten()*train_std+train_mean  # Stesso per 'y_test'
    })
    print('saving')
    # Salva il DataFrame in un file CSV
    results_df.to_csv(f'./TUNER/predictions/lstm/{ticker}.csv', index=False)


def main():
    ticker_list = ['RWR', 'XLB', 'XLI', 'XLY', 'XLP', 'XLE', 'XLF', 'XLU', 'XLV', 'XLK']
    start_date = "2015-01-01"
    end_date = "2023-10-30"
    forecast_horizon = 10
    window_size = 30
    start_garch_date = "2015-02-03"
    target_column = 'Volatility10'

    with concurrent.futures.ProcessPoolExecutor() as executor:
       futures = [
           executor.submit(initialize_model,ticker,download_data(ticker,start_date,end_date,start_garch_date),window_size,target_column,forecast_horizon) for ticker in ticker_list]
       concurrent.futures.wait(futures)




# This if statement makes sure that the script runs only if it is executed as the main program
if __name__ == "__main__":
    main()