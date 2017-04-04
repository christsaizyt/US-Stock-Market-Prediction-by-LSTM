import numpy as np
import pandas as pd
from datetime import timedelta
import pickle as pkl
import quandl
from quandl.errors.quandl_error import NotFoundError
import datetime
import time
import re
import os

from keras.models import load_model

ticker_files_format_dict = {'predictions':'_reg.csv',
                            'models':'_reg.h5',
                            'logs':'_reg_log.pkl',
                            'parameters':'_reg_par.pkl'
                            }

def save_lstm_parameters(filename, paras):
    with open(filename + ticker_files_format_dict['parameters'], 'wb') as f:
        pkl.dump(paras, f)

def load_lstm_parameters(filename: object) -> object:
    with open(filename + ticker_files_format_dict['parameters'], 'rb') as f:
        paras = pkl.load(f)
    return paras

def save_lstm_logs(filename, paras):
    with open(filename + ticker_files_format_dict['logs'], 'wb') as f:
        pkl.dump(paras, f)

def load_lstm_logs(filename):
    with open(filename + ticker_files_format_dict['logs'], 'rb') as f:
        paras = pkl.load(f)
        return paras

def save_lstm_models(filname, model):
    model.save(filname + ticker_files_format_dict['models'])

def load_lstm_models(filename):
    if check_files_exists(filename + ticker_files_format_dict['models']) == False:
        return None
    model = load_model(filename + ticker_files_format_dict['models'])
    return model

def save_lstm_predictions(filnemae, df):
    df.to_csv(filnemae + ticker_files_format_dict['predictions'], index_label='Date')

def load_lstm_predictions(filename):
    if check_files_exists(filename + ticker_files_format_dict['predictions']) == False:
        return None
    df_load = pd.read_csv(filename + ticker_files_format_dict['predictions'], index_col='Date')
    return df_load

def returnNumbers(str):
    return re.search(r'\d+', str).group()

def append_date_serires(df, append_len):
    append_date = []
    append_last_date = df.index[-1]
    i = append_len
    while i >= 1:
        append_last_date = append_last_date + timedelta(days=1)
        if append_last_date.isoweekday() > 0 and append_last_date.isoweekday() < 6:
            append_date.append(append_last_date)
            i -= 1
    append_df = pd.DataFrame(index=list(append_date))
    df = pd.concat((df, append_df), axis=0)
    return df

def returnNewDatesArr(lut,tar_date, shit_days):
    start_idx = int(np.argwhere(lut == tar_date[0]))
    end_idx = int(np.argwhere(lut == tar_date[-1])) + 1
    new_sta_idx = start_idx + int(shit_days)
    new_end_idx = end_idx + int(shit_days)
    return lut[new_sta_idx:new_end_idx]

def shift_df_pred(df_tmp, df_pred, end_date, valid_mse, times):
    # df_base: pandas.core.frame.DataFrame
    # df_pred: pandas.core.series.Series, name: p_+10_d
    # df_pred['p_+'+str(paras_cla.pred_len)+'_d']

    df_base = df_tmp.copy()
    lookupDate = df_base.index#.strftime('%Y-%m-%d')
    colname = end_date
    if valid_mse is not None:
        colname = colname + '_' + '%.3f' % valid_mse
    if times > 0:
        colname = colname + '_' + str(times)

    if df_pred.__class__.__name__ == "Series":
        shift_days = returnNumbers(df_pred.name)
        shift_dates_arr = returnNewDatesArr(lookupDate, df_pred.index, shift_days)
        df_base.loc[shift_dates_arr, colname] = np.array(df_pred)

    elif df_pred.__class__.__name__ == "DataFrame":
        # FIXME
        pass

    return df_base

def get_data_from_quandl(ticker, start_date = '2010-01-01', end_date = 'today'):
    if end_date == 'today':
        end_date = str(datetime.date.today())
    while True:
        try:
            df = quandl.get('WIKI/' + ticker,
                            authtoken = '2c24stWyXfdzLVFWxGe4',
                            start_date = start_date,
                            end_date = end_date)
            break
        except NotFoundError:
            print(ticker, 'DatasetNotFound')
            return None
        except:
            print (ticker, 'others error - wait 10 seconds to retry again')
            pass
        time.sleep(10)  # delays for 10 seconds and try again

    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df = df.rename(columns={"Adj. Open": "open", "Adj. High": "high", "Adj. Low": "low",
                                          "Adj. Close": "close", "Adj. Volume": "volume"})

    if np.sum(len(df) - df.count()) != 0:
        print(ticker, 'nan count\n',len(df) - df.count())
        df = df.dropna()
    return df

def get_save_folder(file_path = './'):
    history_folder = file_path + 'history/'
    predictions_folder = history_folder + 'predictions/'
    models_folder = history_folder + 'models/'
    parameters_folder = history_folder + 'parameters/'
    log_folder = history_folder + 'logs/'
    folders = {
        'history':history_folder,
        'predictions':predictions_folder,
        'parameters':parameters_folder,
        'models':models_folder,
        'logs':log_folder
    }
    return folders

def check_and_create_folder(folders):
    '''
    file_path - history - predictions
                        - models
                        - parameters
                        - logs
    '''
    if os.path.exists(folders['history']) == False:
        os.makedirs(folders['history'])
    if os.path.exists(folders['predictions']) == False:
        os.makedirs(folders['predictions'])
    if os.path.exists(folders['parameters']) == False:
        os.makedirs(folders['parameters'])
    if os.path.exists(folders['models']) == False:
        os.makedirs(folders['models'])
    if os.path.exists(folders['logs']) == False:
        os.makedirs(folders['logs'])

def check_files_exists(file_name):
    return os.path.isfile(file_name)

def check_ticker_files_exist(folders, ticker):
    if (check_files_exists(folders['predictions'] + ticker + ticker_files_format_dict['predictions']) == True and
        check_files_exists(folders['parameters'] + ticker + ticker_files_format_dict['parameters']) == True and
        check_files_exists(folders['models'] + ticker + ticker_files_format_dict['models']) == True and
        check_files_exists(folders['logs'] + ticker + ticker_files_format_dict['logs']) == True):
        return True
    return False

def check_ticker_files_status(date_array, file_path, ticker, batch_size):
    up_to_date_flag = False
    can_predict_flag = False
    can_retrain_flag = False

    folders = get_save_folder(file_path)
    check_and_create_folder(folders)
    ticker_files_exist_flag = check_ticker_files_exist(folders, ticker)
    if ticker_files_exist_flag == False:
        return False, False, False

    # if ticker files's are exist, check its log status
    log_file_path = folders['logs'] + ticker
    log_file = load_lstm_logs(log_file_path)

    log_file_pred_last_date_idx = np.argwhere(date_array == log_file['pred_last_date'])[0, 0]
    log_file_train_last_date_idx = np.argwhere(date_array == log_file['train_last_date'])[0, 0]
    len_total = len(date_array)

    if len_total > (log_file_pred_last_date_idx + 1):
        can_predict_flag = True
    else:
        up_to_date_flag = True

    if len_total > (log_file_train_last_date_idx + batch_size):
        can_retrain_flag = True

    return up_to_date_flag, can_predict_flag,can_retrain_flag

def go_thorugh_valid_tickers(tickers):
    valid_tickers = []
    for ticker in tickers:
        df = get_data_from_quandl(ticker)
        if df is not None:
            print (ticker, float(df[-1:]['close']))
            valid_tickers.append((ticker, float(df[-1:]['close'])))
    valid_tickers = sorted(valid_tickers, key=lambda x: x[1], reverse=True)
    return valid_tickers

def display_par_log(ticker, file_path = './'):
    folder = get_save_folder(file_path)
    paras = load_lstm_parameters(folder['parameters']+ticker)
    log = load_lstm_logs(folder['logs']+ticker)
    print (paras)
    print (log['train_last_date'])
    print(log['pred_last_date'])
    for eval_score in log['eval_score']:
        print (eval_score)

def load_all_tickers(file_path = ''):
    with open(file_path + 'tickers.txt') as file:
        tickers_file = file.read()
    tickers = tickers_file.split('\n')
    return tickers
