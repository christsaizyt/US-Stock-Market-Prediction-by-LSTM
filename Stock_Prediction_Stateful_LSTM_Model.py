import numpy as np
np.random.seed(7)

import csv
import time
import pandas as pd
import Stock_Prediction_General_Method as spgm

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import preprocessing

class rnn_lstm(object):

    def __init__(self, paras):
        self.paras = paras
        self.df = None
        self.log = {
            'train_last_date': None,
            'pred_last_date': None,
            'eval_score': None,
            'training_history': None
        } # dict, save two keys: train_last_date and pred_last_date
        print('Ticker: ', self.paras.ticker)

    def preprocessing_data_by_row(self, data):
        '''
        data: N*M np.array
        N: sample
        M: features
        data_T: M*N
        data_T_scale: scaler for column by column, M*N
        data_T_scale_T: N*M
        '''
        if data.size == 0:
            return data, None

        data_T = data.transpose()
        if self.paras.preproc_scaler == 'standard_scaler':
            scaler = preprocessing.StandardScaler().fit(data_T)
        else:  # FIXME
            scaler = preprocessing.StandardScaler().fit(data_T)
        data_T_scale = scaler.transform(data_T)
        data_T_scale_T = data_T_scale.transpose()
        return data_T_scale_T, scaler

    def divide_into_price_volume(self, df):
        '''
        df.columns = [..., o_-10_d,h_-10_d,l_-10_d,c_-10_d,v_-10_d,...]
        return [...,o_-10_d,h_-10_d,l_-10_d,c_-10_d,...], [...,v_-10_d,...]
        '''
        volume_cols = [col for col in df.columns if 'v_' in col]
        return np.array(df.drop(volume_cols, 1)), np.array(df[volume_cols])

    def combine_price_volume(self, X_price, X_volume):
        X_combined = X_price
        if X_volume.size != 0:
            for i in range(len(X_volume[0]) - 1, -1, -1):
                X_combined = np.insert(X_combined, (i + 1) * (self.paras.n_features - 1),
                                       X_volume[:, i], axis=1)
        return X_combined

    def reshape_input(self, X, y):
        '''
        X.shape = [n_sample, window_len*n_features]
        X_reshaped = [n_sample, window_len, n_features]
        '''
        # reshape X to be [samples, time steps, features]
        if X is None or y is None:  # np.isnan(X).any() or np.isnan(y).any():
            return None, None

        n_sample = X.shape[0]
        n_channel = int(self.paras.n_features)
        n_features_per_channel = int(X.shape[1] / n_channel)
        X_reshaped = np.reshape(X, (n_sample, n_features_per_channel, n_channel))
        y_reshaped = np.reshape(y, (n_sample, -1))
        return X_reshaped, y_reshaped

    def build_LSTM_model(self):
        model_lstm = Sequential()
        first = True
        for idx in range(len(self.paras.model['hidden_layers'])):
            if idx == (len(self.paras.model['hidden_layers']) - 1):
                if first == True:
                    model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]),
                                        batch_input_shape=(self.paras.batch_size,
                                                           self.paras.window_len + 1,
                                                           int(self.paras.n_features)),
                                        stateful=True,
                                        return_sequences=False))
                    model_lstm.add(Activation(self.paras.model['activation'][idx]))
                    model_lstm.add(Dropout(self.paras.model['dropout'][idx]))
                else:
                    model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]),
                                        return_sequences=False,
                                        stateful=True,
                                        ))
                    model_lstm.add(Activation(self.paras.model['activation'][idx]))
                    model_lstm.add(Dropout(self.paras.model['dropout'][idx]))
            elif first == True:
                model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]),
                                    batch_input_shape=(self.paras.batch_size,
                                                       self.paras.window_len + 1,
                                                       int(self.paras.n_features)),
                                    stateful=True,
                                    return_sequences=True))
                model_lstm.add(Activation(self.paras.model['activation'][idx]))
                model_lstm.add(Dropout(self.paras.model['dropout'][idx]))
                first = False
            else:
                model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]),
                                    return_sequences=True,
                                    stateful=True,
                                    ))
                model_lstm.add(Activation(self.paras.model['activation'][idx]))
                model_lstm.add(Dropout(self.paras.model['dropout'][idx]))

        # output layer
        model_lstm.add(Dense(output_dim=self.paras.model['out_layer']))
        model_lstm.add(Activation(self.paras.model['out_activation']))
        model_lstm.compile(loss=self.paras.model['loss'], optimizer=self.paras.model['optimizer'])
        print('build LSTM model...')
        return model_lstm

# Regression
class rnn_lstm_regression(rnn_lstm):
    def __init__(self, paras):
        super(rnn_lstm_regression, self).__init__(paras=paras)

    def check_parameters(self):
        if (self.paras.out_class_type == 'classification' or
            self.paras.model['out_activation'] == 'softmax' or
            self.paras.model['loss'] == 'categorical_crossentropy'):
            return False
        return True

    def GetStockData_PriceVolume(self):
        '''
        All data is from Quandl wiki dataset
        Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low  
        Adj. Close  Adj. Volume]
        '''

        # Prepare data frame
        df = spgm.get_data_from_quandl(self.paras.ticker,
                                       start_date=self.paras.start_date,
                                       end_date=self.paras.end_date)

        df_all = df.copy()
        df['MA'] = df['close'].rolling(window=self.paras.pred_len, center=False).mean()

        # Data frame output
        if self.paras.out_type == 'MA':
            df['label'] = df['MA'].shift(-1 * self.paras.pred_len)
        else:
            df['label'] = df['close'].shift(-1 * self.paras.pred_len)

        # Generate input features for time series data
        featureset = list(['label'])
        featuresDict = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume'}
        for i in range(self.paras.window_len, -1, -1):
            for j in list(self.paras.features):
                df[j + '_-' + str(i) + '_d'] = df[featuresDict[j]].shift(1 * i)
                featureset.append(j + '_-' + str(i) + '_d')

        df = df[(self.paras.window_len):][featureset]
        cnt_throw_away = len(df) - int(len(df) / self.paras.batch_size) * self.paras.batch_size
        df = df[cnt_throw_away:]

        return df, df_all

    def preprocessing_data(self, df, featureDropForTraining, with_label_proc=True):
        '''
        df: pd.DataFrame
        X: np.array
        y: np.array
        convert df into X,y
        '''
        if df.empty:
            return None, None, None
        y = np.array(df['label'])
        X_price, X_volume = self.divide_into_price_volume(df.drop(featureDropForTraining, 1))

        X_price, scaler_price = self.preprocessing_data_by_row(X_price)
        X_volume, scaler_volume = self.preprocessing_data_by_row(X_volume)

        # combine price and volume - rearrange
        # [...,o_-10_d,h_-10_d,l_-10_d,c_-10_d,...], [...,v_-10_d,...] -> [..., o_-10_d,h_-10_d,l_-10_d,c_-10_d,v_-10_d,...]
        X_combined = self.combine_price_volume(X_price, X_volume)

        if with_label_proc == True:
            y_normalized = scaler_price.transform(y.reshape(1, -1))
            y_normalized_T = y_normalized.reshape(-1, 1)
        else:
            y_normalized_T = np.repeat(float('nan'), len(y))

        scaler_combined = {'price': scaler_price, 'volume': scaler_volume}
        return X_combined, y_normalized_T, scaler_combined

    def run(self):
        if self.check_parameters() == False:

            raise IndexError('Parameters for LSTM is wrong, check out_class_type')

        ##############################f##################################################
        folders = spgm.get_save_folder(self.paras.save_folder)
        if self.paras.save:
            print('Save Directory: ', folders)
        ################################################################################

        featureDropForTraining = ['label']

        # get data - including remove some incompatible data for batch training
        df, df_all = self.GetStockData_PriceVolume()
        print ('df len:', len(df), 'df_all len', len(df_all), 'df size:',df.size, 'df shape', np.shape(df))
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        #print (df[-(self.paras.pred_len+self.paras.valid_len+self.paras.window_len):][['o_-0_d','h_-0_d','l_-0_d','c_-0_d','label']])

        # get evaluation date
        eval_date_idx = len(df) - self.paras.batch_size * self.paras.valid_len
        print ('first eval date idx: (no data)', eval_date_idx, ',date', df.index[eval_date_idx].strftime('%Y-%m-%d'))

        # get corresponding training dates
        train_len = eval_date_idx - self.paras.pred_len
        print ('first train from ', df.index[0].strftime('%Y-%m-%d'), ' to ', df.index[train_len-1].strftime('%Y-%m-%d'))

        #  preprocessing
        X_train, y_train, scaler_train = self.preprocessing_data(df[:train_len], featureDropForTraining, with_label_proc=True)
        print ('before reshape X_train:', X_train.shape, 'y_train:', y_train.shape)

        # reshape training data to LSTM model
        X_train, y_train = self.reshape_input(X_train, y_train)
        print('after reshape X_train:', X_train.shape, 'y_train:', y_train.shape)

        # build LSTM model
        model_lstm = self.build_LSTM_model()

        # first train LSTM model
        #earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', patience=0, verbose=0, mode='auto')
        self.log['training_history'] = []
        now = time.time()
        for i in range(self.paras.epoch):
            model_lstm.fit(X_train, y_train, nb_epoch=1,
                           batch_size=self.paras.batch_size,
                           verbose=0,
                           shuffle=False)
                           #callbacks=[earlyStopping])
            model_lstm.reset_states()

            loss = model_lstm.evaluate(X_train, y_train,
                                       batch_size=self.paras.batch_size,
                                       verbose=0)
            model_lstm.reset_states()

            self.log['training_history'].append(loss)
            if True:#i % 10 == 0 or i == self.paras.epoch - 1:
                time_diff = (time.time() - now) / (1 if i == 0 else 10)
                now = time.time()
                print(self.paras.ticker, '-',
                      'Init training epoch:', i+1, '/', self.paras.epoch,
                      'Loss:',loss,
                      ' - ', str(int(time_diff)) + 's')

        self.log['eval_score'] = []
        # evaluate and re-train model by batch
        for i in range(self.paras.valid_len):

            # get the next eval dates duration
            next_pred_idx_sta = eval_date_idx + i*self.paras.batch_size
            next_pred_idx_end = eval_date_idx + (i+1)*self.paras.batch_size

            # prepare for the prediction and if there is label on next_pre_idx, do evaluation by the way
            with_label = True
            if next_pred_idx_end > len(df) - self.paras.pred_len:
                with_label = False

            X_pred_next, \
            y_pred_next, \
            scaler_pred = self.preprocessing_data(df[next_pred_idx_sta:next_pred_idx_end],
                                                  featureDropForTraining, with_label_proc=with_label)
            X_pred_next, y_pred_next = self.reshape_input(X_pred_next, y_pred_next)

            score = np.nan
            if with_label == True:
                # evaluate for next batch
                score = model_lstm.evaluate(X_pred_next, y_pred_next,
                                            verbose=0,
                                            batch_size=self.paras.batch_size)
                model_lstm.reset_states()

            print(self.paras.ticker, '-',
                  'eval num: ', i, '/', self.paras.valid_len, ' ',
                  'pred date from:', df.index[next_pred_idx_sta].strftime('%Y-%m-%d'),
                  'to ', df.index[next_pred_idx_end - 1].strftime('%Y-%m-%d'),
                  'score:', score)

            # get the predictions
            preds = model_lstm.predict(X_pred_next,
                                       verbose=0,
                                       batch_size=self.paras.batch_size)
            model_lstm.reset_states()

            av = np.array(scaler_pred['price'].inverse_transform(y_pred_next.reshape(y_pred_next.shape[0], )))
            pv = np.array(scaler_pred['price'].inverse_transform(preds.reshape(preds.shape[0], )))
            df_all.loc[df.index[next_pred_idx_sta:next_pred_idx_end], 'a_+' + str(self.paras.pred_len) + '_d'] = av
            df_all.loc[df.index[next_pred_idx_sta:next_pred_idx_end], 'p_+' + str(self.paras.pred_len) + '_d'] = pv

            # get the next training dates duration
            next_train_idx_sta = train_len + i * self.paras.batch_size
            next_train_idx_end = train_len + (i + 1) * self.paras.batch_size

            with_label = True
            if next_train_idx_end > len(df) - self.paras.pred_len:
                with_label = False
            X_train_next, \
            y_train_next, \
            scaler_next = self.preprocessing_data(df[next_train_idx_sta:next_train_idx_end],
                                                  featureDropForTraining, with_label_proc=with_label)
            X_train_next, y_train_next = self.reshape_input(X_train_next, y_train_next)
            # retrain model for the next batch
            if with_label == True:
                now = time.time()
                for j in range(self.paras.epoch):
                    model_lstm.fit(X_train_next, y_train_next, nb_epoch=1,
                                   batch_size=self.paras.batch_size,
                                   verbose=0,
                                   shuffle=False)
                    model_lstm.reset_states()

            score_after_in_batch_training = model_lstm.evaluate(X_pred_next, y_pred_next,
                                                                batch_size=self.paras.batch_size,
                                                                verbose=0)
            model_lstm.reset_states()
            print(self.paras.ticker, '-',
                  'eval num: ', i, '/', self.paras.valid_len, ' ',
                  'train date from:', df.index[next_train_idx_sta].strftime('%Y-%m-%d'),
                  'to ', df.index[next_train_idx_end - 1].strftime('%Y-%m-%d'),
                  'after retrain this batch, score:', score_after_in_batch_training)
            if score is not np.nan and score_after_in_batch_training is not np.nan:
                self.log['eval_score'].append((df.index[next_pred_idx_sta].strftime('%Y-%m-%d'),
                                               df.index[next_pred_idx_end - 1].strftime('%Y-%m-%d'), score,
                                               score_after_in_batch_training))

        print('score mean:', np.mean([x[2] for x in self.log['eval_score']]))

        self.df = df_all[-(self.paras.valid_len * self.paras.batch_size):]
        if self.paras.save == True: # save model, prediction, parameter, log here!
            # save models
            spgm.save_lstm_models(folders['models'] + self.paras.ticker, model_lstm)

            # save predictions
            spgm.save_lstm_predictions(folders['predictions'] + self.paras.ticker, self.df)

            # save parameters
            spgm.save_lstm_parameters(folders['parameters'] + self.paras.ticker, self.paras)

            # save logs
            self.log['train_last_date'] = self.df.index[-1].strftime('%Y-%m-%d')
            self.log['pred_last_date'] = self.df.index[-1].strftime('%Y-%m-%d')
            spgm.save_lstm_logs(folders['logs'] + self.paras.ticker, self.log)

        # pd.set_option('display.max_rows', None)
        # self.df_pred = self.df["p_+" + str(self.paras.pred_len) + '_d']
        # print (self.df)
