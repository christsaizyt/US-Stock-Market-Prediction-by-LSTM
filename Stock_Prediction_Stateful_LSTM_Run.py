import numpy as np
np.random.seed(7)

from Stock_Prediction_Stateful_LSTM_Model import rnn_lstm_regression
import Stock_Prediction_General_Method as spgm
import Stock_Prediction_Global_Parameters as spgp
import datetime

def run_regression_script():
    # run regression
    for ticker in g_tickers:

        # Regression Parameters
        paras = spgp.SP_RNN_LSTM_Paras('lstm', ticker)
        paras.save = True
        paras.features = 'ohlcv'
        paras.window_len = 10#120
        paras.pred_len = 10
        paras.valid_len = 20
        paras.out_type = 'MA'
        paras.start_date = '2010-01-01'
        # paras.end_date = 'current'
        paras.batch_size = 1
        paras.epoch = 1#300
        paras.model['hidden_layers'] = [120, 60, 30]
        paras.model['dropout'] = [0.3, 0.5, 0.3]
        paras.model['activation'] = ['relu', 'relu', 'relu']
        paras.model['optimizer'] = 'adam'

        paras.end_date = str(datetime.date.today())

        print(100 * '#')
        df_base = spgm.get_data_from_quandl(paras.ticker, start_date=paras.start_date,end_date=paras.end_date)
        if df_base is None:
            print ('Error ticker name:', ticker, ' - skipped')
            continue

        lstm_reg = rnn_lstm_regression(paras)
        lstm_reg.run()
        print (lstm_reg.df)

    #pd.set_option('display.max_rows', None)

def run_classification_script():
    # Classification Parameters
    paras = spgm.load_lstm_parameters('g_paras_cla')
    pass

#######################################################################################################################
if __name__ == "__main__":
    g_tickers = ['NDAQ']#['NDAQ','AAPL','GOOGL','FB','YHOO','YELP','AMZN','MSFT']

    run_regression_script()


