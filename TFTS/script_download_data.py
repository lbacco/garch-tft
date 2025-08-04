# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Script to download  data for a default experiment.

Only downloads data if the csv files are present, unless the "force_download"
argument is supplied. For new datasets, the download_and_unzip(.) can be reused
to pull csv files from an online repository, but may require subsequent
dataset-specific processing.

Usage:
  python3 script_download_data {EXPT_NAME} {OUTPUT_FOLDER} {FORCE_DOWNLOAD}

Command line args:
  EXPT_NAME: Name of experiment to download data for  {e.g. volatility}
  OUTPUT_FOLDER: Path to folder in which
  FORCE_DOWNLOAD: Whether to force data download from scratch.



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import gc
import glob
import os
import shutil
import sys
import emd
from expt_settings.configs import ExperimentConfig
import numpy as np
import pandas as pd
import pyunpack
import wget
import yfinance as yf
import math

# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
  """Downloads a file froma url."""

  print('Pulling data from {} to {}'.format(url, output_path))
  wget.download(url, output_path)
  print('done')


def recreate_folder(path):
  """Deletes and recreates folder."""

  shutil.rmtree(path)
  os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.

  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """

  download_from_url(url, zip_path)

  unzip(zip_path, csv_path, data_folder)

  print('Done.')


# Dataset specific download routines.
def download_volatility(config):
  """Downloads volatility data from yahoo finance."""

  ticker_etf_list = ["AAXJ", "GLD", "AGG", "DBC", "ACWI", "EEM", "EFV"]
  df = pd.DataFrame()
  start_date = "2010-01-01"
  end_date = "2023-08-30"
  # end_date per i file--- MUST BE one day before end_date
  end_date2= "2023-08-29"

  vix_df= yf.download('^VIX', start=start_date, end=end_date)
  gold_df= yf.download('GC=F', start=start_date, end=end_date)
  brent_df= yf.download('CL=F', start=start_date, end= end_date)
  sep500_df= yf.download('^GSPC', start=start_date, end= end_date)

  for etf in ticker_etf_list:
      data = yf.download(etf, start=start_date, end=end_date)
      # add gold,wti,vix,s&p500 columns
      data['vix']=vix_df['Adj Close']
      data['vix'].fillna(method='ffill', inplace=True)
      data['gold']=gold_df['Adj Close']
      data['gold'].fillna(method='ffill', inplace=True)
      data['brent']= brent_df['Adj Close']
      data['brent'].fillna(method='ffill', inplace=True)
      data['s&p500']= sep500_df['Adj Close']
      data['s&p500'].fillna(method='ffill', inplace=True)
      # Compute returns
      data["Returns"] = data["Adj Close"].pct_change() * 100
      # Compute historical volatility with a 7-day window
      data['log_returns'] = (data['Adj Close'].apply(lambda x: math.log(x) if math.log(x) else None) - data['Adj Close'].shift().apply(lambda x: math.log(x) if math.log(x) else None))*100
      data = data[1:]
      # Compute historical volatility with a 7-day window
      data["Volatility"] = data["log_returns"].rolling(window=5).std()
      data["Volatility20"] = data["log_returns"].rolling(window=20).std()
      data["Volatility10"] = data["log_returns"].rolling(window=10).std()
      data = data[20:]
      data['Symbol'] = etf
      data['open_to_close'] = data['Close'] - data['Open']
      df = pd.concat([df, data], axis=0)

  data_folder = config.data_folder
  csv_path = os.path.join(data_folder, 'etfsdata.csv')
  zip_path = os.path.join(data_folder, 'etfsdata.zip')


  print(df)
  # Adds additional date/day fields
  idx = [str(s).split('+')[0] for s in df.index]  # ignore timezones, we don't need them
  dates = pd.to_datetime(idx)
  df.drop(['Close'],axis=1)
  df['date'] = dates
  df['days_from_start'] = (dates - pd.Timestamp(2017, 1, 1)).days
  df['day_of_week'] = dates.dayofweek
  df['day_of_month'] = dates.day
  df['week_of_year'] = dates.strftime('%U')
  df['month'] = dates.month
  df['year'] = dates.year
  df['categorical_id'] = df['Symbol'].copy()

  print(f"NaNs in DataFrame:")
  print(df.isna().sum())

  # Adds static information
  symbol_region_mapping = {
      "AAXJ":'APEC',
      "GLD":'WRLD',
      "AGG":'AMER',
      "DBC":'WRLD',
      "ACWI":'WRLD',
      "EEM":'APAC',
      "EFV":'EUR'
  }

  df['Region'] = df['Symbol'].apply(lambda k: symbol_region_mapping[k])
  # Performs final processing
  output_df_list = []
  for grp in df.groupby('Symbol'):
    sliced = grp[1].copy()
    sliced.sort_values('days_from_start', inplace=True)
    # Impute log volatility values
    sliced['Volatility'].fillna(method='ffill', inplace=True)
    sliced.dropna()
    output_df_list.append(sliced)
  print(df.info())
  df = pd.concat(output_df_list, axis=0)


  output_file = config.data_csv_path
  print('Completed formatting, saving to {}'.format(output_file))
  print(df)
  df.to_csv(output_file)
  print('Done.')


def calculate_IMFs(data, column_name):
    # Calcola le Intrinsec Mode Functions (IMFs) per una colonna specificata
    IMFs = emd.sift.sift(data[column_name].values)

    # Aggiungi le IMFs come colonne al DataFrame
    for i in range(4):
        data.loc[:, f"IMF_{column_name}{i + 1}"] = IMFs[:, i]

    return data


def calculate_IMFs_all(data):
    for name in data.columns:
        calculate_IMFs(name)
def download_returns(config):
    """Downloads returns data ."""
    ticker_etf_list = ["AAXJ", "GLD", "AGG", "DBC", "ACWI", "EEM", "EFV"]
    df = pd.DataFrame()
    start_date = "2017-01-01"
    end_date = "2023-08-30"
    # end_date per i file--- MUST BE one day before end_date
    end_date2 = "2023-08-29"

    vix_df = yf.download('^VIX', start=start_date, end=end_date)
    gold_df = yf.download('GC=F', start=start_date, end=end_date)
    brent_df = yf.download('CL=F', start=start_date, end=end_date)
    sep500_df = yf.download('^GSPC', start=start_date, end=end_date)

    for etf in ticker_etf_list:
        data = yf.download(etf, start=start_date, end=end_date)
        # add gold,wti,vix,s&p500 columns
        data['vix'] = vix_df['Adj Close']
        data['vix'].fillna(method='ffill', inplace=True)
        data['gold'] = gold_df['Adj Close']
        data['gold'].fillna(method='ffill', inplace=True)
        data['brent'] = brent_df['Adj Close']
        data['brent'].fillna(method='ffill', inplace=True)
        data['s&p500'] = sep500_df['Adj Close']
        data['s&p500'].fillna(method='ffill', inplace=True)
        # Compute returns
        data["Returns"] = data["Adj Close"].pct_change() * 100
        # Compute historical volatility with a 7-day window
        data['log_returns'] = (data['Adj Close'].apply(lambda x: math.log(x) if math.log(x) else None) - data[
            'Adj Close'].shift().apply(lambda x: math.log(x) if math.log(x) else None)) * 100
        data = data[1:]
        # Compute historical volatility with a 7-day window
        data["Volatility"] = data["log_returns"].rolling(window=5).std()
        data["Volatility20"] = data["log_returns"].rolling(window=20).std()
        data["Volatility10"] = data["log_returns"].rolling(window=10).std()
        data = data[20:]
        data['Symbol'] = etf
        data['open_to_close'] = data['Close'] - data['Open']
        data = calculate_IMFs(data, 'log_returns')
        df = pd.concat([df, data], axis=0)

    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, 'etfsdata.csv')
    zip_path = os.path.join(data_folder, 'etfsdata.zip')

    print(df)
    # Adds additional date/day fields
    idx = [str(s).split('+')[0] for s in df.index]  # ignore timezones, we don't need them
    dates = pd.to_datetime(idx)
    df.drop(['Close'], axis=1)
    df['date'] = dates
    df['days_from_start'] = (dates - pd.Timestamp(2017, 1, 1)).days
    df['day_of_week'] = dates.dayofweek
    df['day_of_month'] = dates.day
    df['week_of_year'] = dates.strftime('%U')
    df['month'] = dates.month
    df['year'] = dates.year
    df['categorical_id'] = df['Symbol'].copy()

    print(f"NaNs in DataFrame:")
    print(df.isna().sum())

    # Adds static information
    symbol_region_mapping = {
        "AAXJ": 'APEC',
        "GLD": 'WRLD',
        "AGG": 'AMER',
        "DBC": 'WRLD',
        "ACWI": 'WRLD',
        "EEM": 'APAC',
        "EFV": 'EUR'
    }

    df['Region'] = df['Symbol'].apply(lambda k: symbol_region_mapping[k])
    # Performs final processing
    output_df_list = []
    for grp in df.groupby('Symbol'):
        sliced = grp[1].copy()
        sliced.sort_values('days_from_start', inplace=True)
        # Impute log volatility values
        sliced['IMF_log_returns1'].fillna(method='ffill', inplace=True)
        sliced.dropna()
        output_df_list.append(sliced)
    print(df.info())
    df = pd.concat(output_df_list, axis=0)

    output_file = config.data_csv_path
    print('Completed formatting, saving to {}'.format(output_file))
    print(df)
    df.to_csv(output_file)
    print('Done.')

# Core routine.
def main(expt_name, force_download, output_folder):
  """Runs main download routine.

  Args:
    expt_name: Name of experiment
    force_download: Whether to force data download from scratch
    output_folder: Folder path for storing data
  """

  print('#### Running download script ###')

  expt_config = ExperimentConfig(expt_name, output_folder)

  if os.path.exists(expt_config.data_csv_path) and not force_download:
    print('Data has been processed for {}. Skipping download...'.format(
        expt_name))
    sys.exit(0)
  else:
    print('Resetting data folder...')
    recreate_folder(expt_config.data_folder)

  # Default download functions
  download_functions = {
      'volatility': download_volatility,
      'volatility10': download_volatility,
      'volatility20': download_volatility,
      'returns': download_returns
  }

  if expt_name not in download_functions:
    raise ValueError('Unrecongised experiment! name={}'.format(expt_name))

  download_function = download_functions[expt_name]

  # Run data download
  print('Getting {} data...'.format(expt_name))
  download_function(expt_config)

  print('Download completed.')


if __name__ == '__main__':

  def get_args():
    """Returns settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description='Data download configs')
    parser.add_argument(
        'expt_name',
        metavar='e',
        type=str,
        nargs='?',
        choices=experiment_names,
        help='Experiment Name. Default={}'.format(','.join(experiment_names)))
    parser.add_argument(
        'output_folder',
        metavar='f',
        type=str,
        nargs='?',
        default='.',
        help='Path to folder for data download')
    parser.add_argument(
        'force_download',
        metavar='r',
        type=str,
        nargs='?',
        choices=['yes', 'no'],
        default='no',
        help='Whether to re-run data download')

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == '.' else args.output_folder

    return args.expt_name, args.force_download == 'yes', root_folder

  name, force, folder = get_args()

  main(expt_name=name, force_download=force, output_folder=folder)