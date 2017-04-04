import Stock_Prediction_Global_Parameters
from Stock_Prediction_Stateless_LSTM_Model import rnn_lstm_regression

paras = Stock_Prediction_Global_Parameters.SP_RNN_LSTM_Paras('lstm', 'NDAQ')

paras.save = True
paras.features = 'ohlcv'
paras.window_len = 120
paras.pred_len = 10
paras.valid_len = 20
paras.out_type = 'MA'
paras.start_date = '2010-01-01'
# paras.end_date = 'current'

paras.batch_size = 128
paras.epoch = 150
paras.model['hidden_layers'] = [120, 60, 30]
paras.model['dropout'] = [0.7, 0.5, 0.3]
paras.model['activation'] = ['relu', 'relu', 'relu']
paras.model['optimizer'] = 'rmsprop'

# regression setting
paras.out_class_type = 'regression'
paras.n_out_class = 5  # ignore for regression
paras.model['out_layer'] = 1
paras.model['loss'] = 'mse'
paras.model['out_activation'] = 'linear'

# run
lstm_reg1 = rnn_lstm_regression(paras)
lstm_reg1.run()