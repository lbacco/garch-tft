import numpy as np
import yfinance as yf
import concurrent.futures
import math
import pandas as pd
import json
import shutil
import logging
import random
random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from keras_tuner import Hyperband,HyperParameters


def read_ticker_data(data,ticker):

    etf_data=data[data['Symbol'] ==ticker]
    etf_data.index=etf_data['Date']
    etf_data=etf_data.drop(['Date', 'categorical_id', 'Region', 'Symbol', 'date'],axis=1)

    return etf_data
def verifica_salvataggio_file(percorso_file):
    if os.path.exists(percorso_file) and os.path.isfile(percorso_file):
        print(f"Il file {percorso_file} è stato salvato correttamente.")
    else:
        print(f"Attenzione: il file {percorso_file} non esiste o non è stato salvato correttamente.")


def normalization(train_data,val_data,test_data):

  train_mean= train_data.mean()
  train_std=train_data.std()
  train_data = (train_data - train_mean) / train_std
  val_data = (val_data - train_mean) / train_std
  test_data = (test_data - train_mean) / train_std

  return train_data, val_data,test_data
def generate_sequences(data,window_size,target_column,predicted_horizon):
  X,y = [],[]
  for i in range(0,len(data) - window_size-predicted_horizon):
      X.append(data.values[i:i+window_size])
      y.append(data[target_column].values.flatten()[i+window_size+predicted_horizon])
  return np.array(X), np.array(y)

def preparing_data(data,window_size,target_column,forecast_horizon,mode):
    if mode == 'HYPER':
        train_data = data[:'2022-01-01']
    elif mode == 'TEST':
        train_data = data[:'2023-01-01']
    else:
        raise SyntaxError

    val_data = data['2022-01-01':'2022-12-31']
    test_data = data['2023-01-01':]
    train_data, val_data, test_data = normalization(train_data, val_data, test_data)

    X_train, y_train = generate_sequences(train_data, window_size, target_column, forecast_horizon)
    X_val, y_val = generate_sequences(val_data, window_size, target_column, forecast_horizon)
    X_test, y_test = generate_sequences(test_data, window_size, target_column, forecast_horizon)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    y_test = y_test.astype('float32')

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model_lstm(hp):
    model = tf.keras.models.Sequential([
        # LSTM layers with Dropout
        tf.keras.layers.LSTM(hp.Choice('units1lstm', [16, 32, 64, 128, 256]), return_sequences=True),  # Esempio di regolarizzazione L2
        tf.keras.layers.LSTM(hp.Choice('units2lstm', [16, 32, 64, 128, 256]), return_sequences=True),

        tf.keras.layers.LSTM(hp.Choice('units3lstm', [16, 32, 64, 128, 256]), return_sequences=False),

        # Dense layer with regularization
        tf.keras.layers.Dense(hp.Choice('units1dense', [ 16, 32, 64, 128]),
                              activation=hp.Choice('activation1', ['relu', 'tanh']),
                              kernel_regularizer=regularizers.l2(0.00001)),
        tf.keras.layers.Dropout(hp.Choice('dropout1', [0.25,0.5])),
        # Un altro layer Dense
        tf.keras.layers.Dense(hp.Choice('units2dense', [16, 32, 64, 128]),
                              activation=hp.Choice('activation2', ['relu', 'tanh']),
                              kernel_regularizer=regularizers.l2(0.00001)),
        tf.keras.layers.Dense(units=1)  # 1 output unit
    ])

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.MeanSquaredError(),  # Potrebbe essere più adatto per previsioni di serie temporali
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return model

def verifica_esistenza_cartella(percorso_cartella):
    if os.path.exists(percorso_cartella) and os.path.isdir(percorso_cartella):
        print(f"La cartella {percorso_cartella} esiste.")
        return True
    else:
        print(f"La cartella {percorso_cartella} non esiste.")
        return False

def defining_tuning(ticker,data,window_size,target_column,forecast_horizon):


    tuner = Hyperband(
        build_model_lstm,
        objective='val_loss',
        max_epochs=40,  # Sostituisci con il numero massimo di epoche desiderato
        factor=3, # Valore tipico per il fattore è 3
        directory=f'./TUNER/tuner_lstm_hv/{ticker}',
        project_name='lstm-tuner-hv',
        overwrite=True
    )


    X_train, y_train, X_val, y_val,X_test,y_test = preparing_data(data,window_size,target_column,forecast_horizon,'HYPER')
    logging.info("Inizio del processo di ottimizzazione per il ticker %s", ticker)

    tuner.search(x=X_train, y=y_train, epochs=50, validation_data=(X_val, y_val))


    tuner = Hyperband(
        build_model_lstm,
        objective='val_loss',
        max_epochs=40,  # Sostituisci con il numero massimo di epoche desiderato
        factor=3, # Valore tipico per il fattore è 3
        directory=f'./TUNER/tuner_lstm_hv/{ticker}',
        project_name='lstm-tuner-hv',
        overwrite=False
    )
    logging.info("Salvataggio dei migliori iperparametri per %s", ticker)
    try:
        print(tuner)
        best_trials = tuner.oracle.get_best_trials(num_trials=5)
        print(best_trials)
        best_params = best_trials[0].hyperparameters.values
    # Definisci il percorso del file JSON dove vuoi salvare i dati
        json_file_path = f"./hp/lstm/best_hyperparameters_{ticker}.json"
    # Scrivi i dati nel file JSON
        with open(json_file_path, "w") as file:
            json.dump(best_params, file, indent=4)

        verifica_salvataggio_file(json_file_path)
        print(f"Migliori iperparametri per {ticker} salvati in {json_file_path}")
    except Exception as e:
        print(f"Si è verificato un errore salvataggio hyperparametri: {e}")



    hp = HyperParameters()
    list_names = ['units1lstm',
                  'units2lstm',
                  'units3lstm',
                  'units1dense',
                  'activation1',
                  'dropout1',
                  'units2dense',
                  'activation2',
                  'learning_rate']

    for name in list_names:
        hp.Fixed(name, best_params[name])

    logging.info("Building del modello")
    model = build_model_lstm(hp)
    X_train, y_train, X_val, y_val,X_test,y_test= preparing_data(data, window_size, target_column,forecast_horizon, 'TEST')
    logging.info("Inizio dell'addestramento del modello")
    model.build((None,) + X_train.shape[1:])
    model.fit(X_train, y_train, epochs=best_params['tuner/epochs'], batch_size=32)

    logging.info("Inizio della valutazione del modello")
    model.evaluate(X_test, y_test)
    # Utilizzo del modello per fare previsioni
    predictions = model.predict(X_test)

    results_df = pd.DataFrame({
        'Predictions': predictions.flatten(),
        'Actual': y_test.flatten()
    })

    try:
        results_df.to_csv(f'./TUNER/predictions/lstm/{ticker}.csv', index=True)
        results_path= f'./TUNER/predictions/lstm/{ticker}.csv'
        verifica_salvataggio_file(results_path)
    except Exception as e:
        print(f"Si è verificato un errore nel salvataggio predictions: {e}")

    logging.info("Inizio del processo di pulizia delle cartelle")
    try:
        if verifica_esistenza_cartella(f'./TUNER/tuner_lstm_hv/{ticker}'):
            # Operazioni sulla cartella, ad esempio eliminarla
            shutil.rmtree(f'./TUNER/tuner_lstm_hv/{ticker}')
    except Exception as e:
        print(f"Si è verificato un errore nella pulizia cartella: {e}")

def main():


    logging.info('Reading Data 1')

    data= pd.read_csv('./gkg/data/gkg/gkhvdatag.csv')
    ticker_list = ['RWR', 'XLB', 'XLI', 'XLY','XLP', 'XLE', 'XLF', 'XLU', 'XLV', 'XLK']
    forecast_horizon = 10
    window_size= 15
    target_column='Volatility20' #HV

    logging.info('Start Random')

    for ticker in ticker_list:
        tf.keras.backend.clear_session()
        etf_data= read_ticker_data(data,ticker)
        defining_tuning(ticker,etf_data,window_size, target_column,forecast_horizon)


    # Do some stuff to the model
# This if statement makes sure that the script runs only if it is executed as the main program
if __name__ == "__main__":
    main()

