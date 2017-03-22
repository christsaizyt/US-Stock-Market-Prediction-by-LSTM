import os
from sklearn import preprocessing
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import timedelta
import matplotlib.pyplot as plt  # http://matplotlib.org/examples/pylab_examples/subplots_demo.html
import csv
import pandas as pd
import numpy as np
import quandl
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from keras.callbacks import History

class rnn_lstm(object):
    def __init__(self, paras):
        self.paras = paras
        self.df = None

    def get_file_id(self):
        return (self.paras.identify + '_' + str(self.paras.window_len) + '_' + str(self.paras.pred_len) + '_'
                + str(self.paras.features) + '_' + str(self.paras.start_date) + '_' + str(self.paras.end_date)
                + '_' + str(self.paras.model['hidden_layers']) + '_' + str(self.paras.model['dropout']) + '_'
                + str(self.paras.model['activation']))

    def get_save_directory(self):
        if self.paras.save == False:
            return ''

        dir = './history/'
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        file_id = self.get_file_id()
        save_folder = str(dir) + str(file_id)
        os.makedirs(save_folder)
        return (save_folder + '/')

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
        if len(X_volume[0]) != 0:
            for i in range(len(X_volume[0]) - 1, -1, -1):
                X_combined = np.insert(X_combined, (i + 1) * (self.paras.n_features - 1), X_volume[:, i], axis=1)

        return X_combined

    def reshape_input(self, X, y):
        '''
        X.shape = [n_sample, window_len*n_features]
        X_reshaped = [n_sample, window_len, n_features]
        '''
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
                model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]), return_sequences=False))
                model_lstm.add(Activation(self.paras.model['activation'][idx]))
                model_lstm.add(Dropout(self.paras.model['dropout'][idx]))
            elif first == True:
                model_lstm.add(LSTM(input_dim=int(self.paras.n_features),
                                    output_dim=int(self.paras.model['hidden_layers'][idx]),
                                    return_sequences=True))
                model_lstm.add(Activation(self.paras.model['activation'][idx]))
                model_lstm.add(Dropout(self.paras.model['dropout'][idx]))
                first = False
            else:
                model_lstm.add(LSTM(int(self.paras.model['hidden_layers'][idx]), return_sequences=True))
                model_lstm.add(Activation(self.paras.model['activation'][idx]))
                model_lstm.add(Dropout(self.paras.model['dropout'][idx]))

        # output layer
        model_lstm.add(Dense(output_dim=self.paras.model['out_layer']))
        model_lstm.add(Activation(self.paras.model['out_activation']))
        model_lstm.compile(loss=self.paras.model['loss'], optimizer=self.paras.model['optimizer'])
        print('build LSTM model...')
        return model_lstm

    def save_training_model(self, model, name):
        if self.paras.save == True:
            # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
            model.save(self.paras.save_folder + name + '.h5')  # creates a HDF5 file 'my_model.h5'

    def append_date_serires(self, df):
        append_date = []
        append_last_date = df.index[-1]
        i = self.paras.pred_len
        while i >= 1:
            append_last_date = append_last_date + timedelta(days=1)
            if append_last_date.isoweekday() > 0 and append_last_date.isoweekday() < 6:
                append_date.append(append_last_date)
                i -= 1
        append_df = pd.DataFrame(index=list(append_date))
        df = pd.concat((df, append_df), axis=0)
        return df

    def plot_training_curve(self, history):
        #         %matplotlib inline
        #         %pylab inline
        #         pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots

        # LSTM training
        f, ax = plt.subplots()
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        ax.set_title('loss function')
        ax.set_ylabel('mse')
        ax.set_xlabel('epoch')
        ax.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()
        if self.paras.save == True:
            w = csv.writer(open(self.paras.save_folder + "training_curve_model.txt", "w"))
            for key, val in history.history.items():
                w.writerow([key, val])
            for key, val in history.params.items():
                w.writerow([key, val])

# Regression
class rnn_lstm_regression(rnn_lstm):
    def __init__(self, paras):
        super(rnn_lstm_regression, self).__init__(paras=paras)

    def check_parameters(self):
        if (self.paras.out_class_type == 'classification' or
                    self.paras.model['out_activation'] == 'softmax' or self.paras.model[
            'loss'] == 'categorical_crossentropy'):
            return False
        return True

    def GetStockData_PriceVolume(self):
        '''
        All data is from quandl wiki dataset
        Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
        Adj. Close  Adj. Volume]
        '''

        # Prepare data frame
        stkname = "WIKI/" + str(self.paras.ticker)
        df = quandl.get(stkname, authtoken='2c24stWyXfdzLVFWxGe4', start_date=self.paras.start_date,
                        end_date=self.paras.end_date)
        df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        df = df.rename(columns={"Adj. Open": "open", "Adj. High": "high", "Adj. Low": "low",
                                "Adj. Close": "close", "Adj. Volume": "volume"})
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

        df = df[featureset]
        df_lately = df[-self.paras.pred_len:]
        df.dropna(inplace=True)
        df_valid = df[len(df) - self.paras.valid_len: len(df)]
        df = df[0:len(df) - self.paras.valid_len]

        return df, df_valid, df_lately, df_all

    def preprocessing_data(self, df, featureDropForTraining, with_label_proc=True):
        '''
        df: pd.DataFrame
        X: np.array
        y: np.array
        convert df into X,y
        '''
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

    def LSTM_model_predict(self, model, X, y, scaler=None):
        predictions = model.predict(X)
        mse_scaled = np.mean((y - predictions) ** 2)
        print('scaled data mse: ', mse_scaled)

        if scaler != None:
            arr = np.array(scaler.inverse_transform(y.reshape(y.shape[0], )))
            arr2 = np.array(scaler.inverse_transform(predictions.reshape(predictions.shape[0], )))
            return mse_scaled, arr, arr2
        return mse_scaled, None, None

    def save_data_frame_mse(self, df, mses):
        df['actual'] = df['actual']
        df['pred'] = df['pred']
        df = df.rename(columns={"actual": "a_+" + str(self.paras.pred_len) + '_d',
                                "pred": "p_+" + str(self.paras.pred_len) + '_d'})

        df['a_+' + str(self.paras.pred_len) + '_d_diff'] = df["a_+" + str(self.paras.pred_len) + '_d'] - df['close']
        df['p_+' + str(self.paras.pred_len) + '_d_diff'] = df["p_+" + str(self.paras.pred_len) + '_d'] - df['close']
        new_list = ["a_+" + str(self.paras.pred_len) + '_d', "p_+" + str(self.paras.pred_len) + '_d',
                    'a_+' + str(self.paras.pred_len) + '_d_diff', 'p_+' + str(self.paras.pred_len) + '_d_diff']

        default_list = ['open', 'high', 'low', 'close', 'volume']
        original_other_list = set(df.columns) - set(default_list) - set(new_list)
        original_other_list = list(original_other_list)
        df = df[default_list + original_other_list + new_list]
        model_acc = mses[1] / mses[0]
        if self.paras.save == True:
            df.to_csv(self.paras.save_folder + self.paras.ticker + ('_%.2f' % model_acc) + "_data_frame.csv")
            with open(self.paras.save_folder + 'parameters.txt', "w") as text_file:
                text_file.write(self.paras.__str__())
                text_file.write(str(mses[0]) + '\n')
                text_file.write(str(mses[1]) + '\n')
        return df

    def run(self):
        if self.check_parameters() == False:
            raise IndexError('Parameters for LSTM is wrong, check out_class_type')

        ##############################f##################################################
        self.paras.save_folder = self.get_save_directory()
        print('Save Directory: ', self.paras.save_folder)
        ################################################################################

        featureDropForTraining = ['label']

        # get data
        df, df_valid, df_lately, df_all = self.GetStockData_PriceVolume()
        print('df len:', len(df))
        print('df_valid len:', len(df_valid))
        print('df_lately len:', len(df_lately))
        print('df_all len:', len(df_all))

        # preprocessing
        X, y, scaler = self.preprocessing_data(df, featureDropForTraining, with_label_proc=True)
        X_valid, y_valid, scaler_valid = self.preprocessing_data(df_valid, featureDropForTraining, with_label_proc=True)
        X_lately, y_lately, scaler_lately = self.preprocessing_data(df_lately, featureDropForTraining,
                                                                    with_label_proc=False)

        # cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        print('Test shape X:', X_test.shape, ',y:', y_test.shape)

        # reshape input data to LSTM model
        X, y = self.reshape_input(X, y)
        X_train, y_train = self.reshape_input(X_train, y_train)
        X_test, y_test = self.reshape_input(X_test, y_test)
        X_valid, y_valid = self.reshape_input(X_valid, y_valid)
        X_lately, y_lately = self.reshape_input(X_lately, y_lately)
        print('After reshape X_train shape:', X_train.shape)
        print('After reshape y_train shape:', y_train.shape)

        # build LSTM model
        history = History()
        model_lstm = self.build_LSTM_model()
        model_lstm.fit(
            X_train,
            y_train,
            batch_size=self.paras.batch_size,
            nb_epoch=self.paras.epoch,
            validation_split=self.paras.validation_split,
            # validation_data = (X_known_lately, y_known_lately),
            callbacks=[history],
            verbose=1
        )

        # save model
        self.save_training_model(model_lstm, 'lstm_model')

        # validation test + known lately data
        print(' ############## validation on test data ############## ')
        mse_test, tmp, tmp2 = self.LSTM_model_predict(model_lstm, X_test, y_test)

        print(' ############## validation on train/test lately data ############## ')
        mse_traintest, tmp, tmp2 = self.LSTM_model_predict(model_lstm, X[-self.paras.valid_len:],
                                                           y[-self.paras.valid_len:])

        print(' ############## validation on valid data ############## ')
        mse_known_lately, df_all.loc[df_valid.index, 'actual'], df_all.loc[
            df_valid.index, 'pred'] = self.LSTM_model_predict(model_lstm, X_valid, y_valid,
                                                              scaler=scaler_valid['price'])

        # predict lately data
        print(' ############## validation on lately data ############## ')
        mse_lately, df_all.loc[df_lately.index, 'actual'], df_all.loc[
            df_lately.index, 'pred'] = self.LSTM_model_predict(model_lstm, X_lately, y_lately,
                                                               scaler=scaler_lately['price'])

        # rewrite data frame and save / update
        df_all = self.save_data_frame_mse(df_all, mses=[mse_test, mse_known_lately])
        self.df = df_all

        # plot training loss/ validation loss
        self.plot_training_curve(history)

        pd.set_option('display.max_rows', None)
        print(df_all[-(self.paras.pred_len + self.paras.valid_len):])

# Classification
class rnn_lstm_classification(rnn_lstm):
    def __init__(self, paras):
        super(rnn_lstm_classification, self).__init__(paras=paras)

    def check_parameters(self):
        if (self.paras.out_class_type == 'classification' and self.paras.n_out_class > 1 and
                    self.paras.model['out_activation'] == 'softmax' and self.paras.model[
            'loss'] == 'categorical_crossentropy'):
            return True
        return False

    def get_label_claasification(self, df, n_cluster=5):
        '''
        Use KMeans algorithm to get the classification output
        '''
        len_total = len(df)
        df.dropna(inplace=True)
        X = np.array(df)
        X = X.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)

        # resort KMeans label
        centers_ori = np.reshape(kmeans.cluster_centers_,
                                 (1, -1))  # [[ 0.16464226  2.03577568 -0.55692057  0.89430484 -1.52722935]]
        centers_ori_argsort = np.argsort(centers_ori, axis=1)  # [[4 2 0 3 1]]
        centers_new = np.argsort(centers_ori_argsort, axis=1)  # [[2 4 1 3 0]]
        centers_new = np.reshape(centers_new, (-1, 1))
        labels = kmeans.labels_

        # replace label value form centers_ori to centers_new
        labels = [centers_new[labels[i]] for i in range(len(labels))]

        # check how many for each class
        counters = np.repeat(0, n_cluster)
        for i in labels:
            counters[i] += 1
        print('classification counter: ', counters)
        print('classification centers: ', np.sort(centers_ori, axis=1))
        out_labels = np.append(labels, np.repeat(np.nan, len_total - len(df)))
        return out_labels

    def GetStockData_PriceVolume(self):
        '''
        All data is from quandl wiki dataset
        Feature set: [Open  High    Low  Close    Volume  Ex-Dividend  Split Ratio Adj. Open  Adj. High  Adj. Low
        Adj. Close  Adj. Volume]
        '''

        # Prepare data frame
        stkname = "WIKI/" + str(self.paras.ticker)
        df = quandl.get(stkname, authtoken='2c24stWyXfdzLVFWxGe4', start_date=self.paras.start_date,
                        end_date=self.paras.end_date)
        df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        df = df.rename(columns={"Adj. Open": "open", "Adj. High": "high", "Adj. Low": "low",
                                "Adj. Close": "close", "Adj. Volume": "volume"})
        df_all = df.copy()
        df['MA'] = df['close'].rolling(window=self.paras.pred_len, center=False).mean()

        # Data frame output
        if self.paras.out_type == 'MA':
            df['label_diff'] = df['MA'].shift(-1 * (self.paras.pred_len)) - df['MA']
            df['label'] = self.get_label_claasification(df['label_diff'], self.paras.n_out_class)
            df['label'] = df['label'].shift(self.paras.pred_len - 1)  # for MA
        else:
            df['label_diff'] = df['close'].shift(-1 * (self.paras.pred_len)) - df['close']
            df['label'] = self.get_label_claasification(df['label_diff'], self.paras.n_out_class)

        # Generate input features for time series data
        featureset = list(['label'])
        featuresDict = {'c': 'close', 'h': 'high', 'l': 'low', 'o': 'open', 'v': 'volume'}
        for i in range(self.paras.window_len, -1, -1):
            for j in list(self.paras.features):
                df[j + '_-' + str(i) + '_d'] = df[featuresDict[j]].shift(1 * i)
                featureset.append(j + '_-' + str(i) + '_d')

        df = df[featureset]
        df_lately = df[-self.paras.pred_len:]
        df.dropna(inplace=True)
        df_valid = df[len(df) - self.paras.valid_len: len(df)]
        df = df[0:len(df) - self.paras.valid_len]
        return df, df_valid, df_lately, df_all

    def preprocessing_data(self, df, featureDropForTraining, with_label_proc=True):
        '''
        df: pd.DataFrame
        X: np.array
        y: np.array
        convert df into X,y
        '''
        y = np.array(df['label'])
        X_price, X_volume = self.divide_into_price_volume(df.drop(featureDropForTraining, 1))

        X_price, scaler_price = self.preprocessing_data_by_row(X_price)
        X_volume, scaler_volume = self.preprocessing_data_by_row(X_volume)

        # combine price and volume - rearrange
        # [...,o_-10_d,h_-10_d,l_-10_d,c_-10_d,...], [...,v_-10_d,...] -> [..., o_-10_d,h_-10_d,l_-10_d,c_-10_d,v_-10_d,...]
        X_combined = self.combine_price_volume(X_price, X_volume)

        if with_label_proc == True:
            # generate one hot output
            y = y.astype(int)
            y_normalized_T = np.zeros((len(df), self.paras.n_out_class))
            y_normalized_T[np.arange(len(df)), y] = 1
        else:
            y_normalized_T = np.repeat(float('nan'), len(y))

        scaler_combined = {'price': scaler_price, 'volume': scaler_volume}
        return X_combined, y_normalized_T, scaler_combined

        # def w_categorical_crossentropy():
        # reference to https://github.com/fchollet/keras/issues/2115

    def LSTM_model_predict(self, model, X, y):
        predictions = model.predict(X)
        mse_scaled = np.mean((y - predictions) ** 2)
        print('scaled data mse: ', mse_scaled)

        if self.paras.n_out_class % 2 == 0:
            w = range(-int((self.paras.n_out_class) / 2), int((self.paras.n_out_class) / 2 + 1), 1)
            del w[self.paras.n_out_class / 2]
        else:
            w = range(-int((self.paras.n_out_class - 1) / 2), int((self.paras.n_out_class + 1) / 2), 1)
        if len(y[0]) == self.paras.n_out_class:
            arr = np.matmul(y, w)
        else:
            arr = None
        arr2 = np.matmul(predictions, w)
        return mse_scaled, arr, arr2

    def save_data_frame_mse(self, df, mses):
        df['actual'] = df['actual']
        df['pred'] = df['pred']
        df = df.rename(columns={"actual": "a_+" + str(self.paras.pred_len) + '_d',
                                "pred": "p_+" + str(self.paras.pred_len) + '_d'})
        new_list = ["a_+" + str(self.paras.pred_len) + '_d', "p_+" + str(self.paras.pred_len) + '_d']

        default_list = ['open', 'high', 'low', 'close', 'volume']
        original_other_list = set(df.columns) - set(default_list) - set(new_list)
        original_other_list = list(original_other_list)
        df = df[default_list + original_other_list + new_list]
        model_acc = mses[1] / mses[0]
        if self.paras.save == True:
            df.to_csv(self.paras.save_folder + self.paras.ticker + ('_%.2f' % model_acc) + "_data_frame.csv")
            with open(self.paras.save_folder + 'parameters.txt', "w") as text_file:
                text_file.write(self.paras.__str__())
                text_file.write(str(mses[0]) + '\n')
                text_file.write(str(mses[1]) + '\n')
        return df

    def run(self):
        if self.check_parameters() == False:
            raise IndexError('Parameters for LSTM is wrong, check out_class_type')

        ################################################################################
        self.paras.save_folder = self.get_save_directory()
        print('Save Directory: ', self.paras.save_folder)
        ################################################################################

        featureDropForTraining = ['label']

        # get data
        df, df_valid, df_lately, df_all = self.GetStockData_PriceVolume()
        print('df len:', len(df))
        print('df_valid len:', len(df_valid))
        print('df_lately len:', len(df_lately))
        print('df_all len:', len(df_all))

        # preprocessing
        X, y, scaler = self.preprocessing_data(df, featureDropForTraining, with_label_proc=True)
        X_valid, y_valid, scaler_valid = self.preprocessing_data(df_valid, featureDropForTraining, with_label_proc=True)
        X_lately, y_lately, scaler_lately = self.preprocessing_data(df_lately, featureDropForTraining,
                                                                    with_label_proc=False)

        # cross validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('Train shape X:', X_train.shape, ',y:', y_train.shape)
        print('Test shape X:', X_test.shape, ',y:', y_test.shape)

        # reshape input data to LSTM model
        X, y = self.reshape_input(X, y)
        X_train, y_train = self.reshape_input(X_train, y_train)
        X_test, y_test = self.reshape_input(X_test, y_test)
        X_valid, y_valid = self.reshape_input(X_valid, y_valid)
        X_lately, y_lately = self.reshape_input(X_lately, y_lately)
        print('After reshape X_train shape:', X_train.shape)
        print('After reshape y_train shape:', y_train.shape)

        # build LSTM model
        history = History()
        model_lstm = self.build_LSTM_model()
        model_lstm.fit(
            X_train,
            y_train,
            batch_size=self.paras.batch_size,
            nb_epoch=self.paras.epoch,
            validation_split=self.paras.validation_split,
            # validation_data = (X_known_lately, y_known_lately),
            callbacks=[history],
            verbose=1
        )
        # save model
        self.save_training_model(model_lstm, 'lstm_model')

        # validation test + known lately data
        print(' ############## validation on test data ############## ')
        mse_test, tmp, tmp2 = self.LSTM_model_predict(model_lstm, X_test, y_test)

        print(' ############## validation on train/test lately data ############## ')
        mse_traintest, tmp, tmp2 = self.LSTM_model_predict(model_lstm, X[-self.paras.valid_len:],
                                                           y[-self.paras.valid_len:])

        print(' ############## validation on valid data ############## ')
        mse_known_lately, df_all.loc[df_valid.index, 'actual'], df_all.loc[
            df_valid.index, 'pred'] = self.LSTM_model_predict(model_lstm, X_valid, y_valid)

        # predict lately data
        print(' ############## validation on lately data ############## ')
        mse_lately, df_all.loc[df_lately.index, 'actual'], df_all.loc[
            df_lately.index, 'pred'] = self.LSTM_model_predict(model_lstm, X_lately, y_lately)

        # rewrite data frame and save / update
        df_all = self.save_data_frame_mse(df_all, mses=[mse_test, mse_known_lately])
        self.df = df_all

        # plot training loss/ validation loss
        self.plot_training_curve(history)

        pd.set_option('display.max_rows', None)
        print(df_all[-(self.paras.pred_len + self.paras.valid_len):])
