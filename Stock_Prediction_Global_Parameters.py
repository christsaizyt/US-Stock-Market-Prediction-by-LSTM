import datetime

class SP_Global_Paras(object):
    
    def __init__(self, name, ticker):
        self._name = name
        self._identify = name + '_' + ticker + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._save_folder = ''
        self._save = False

        # ------------- INPUT -------------
        self._ticker = ticker
        self._features = 'ohlcv'
        self._window_len = 120
        self._pred_len = 10
        self._valid_len = 20
        
        # ------------- OUTPUT -------------
        self._out_class_type = 'regression'
        self._out_type = 'MA'
        self._n_out_class = 5
        
        self._start_date = '2010-01-01'
        self._end_date = 'current'
        
        # ------------- Preprocessing scaler -------------
        self._preproc_scaler = 'standard_scaler'
        
    def __str__(self):
        returnString = ('%%%%%%%%%% DUMP SP_Global_Paras %%%%%%%%%%\n' + 
                        'name \t' + str(self._name) + '\n' + 
                        'identify \t' + str(self._identify) + '\n' + 
                        'save \t' + str(self._save) + '\n' + 
                        'save_folder \t' + str(self._save_folder) + '\n' + 
                        'ticker \t' + str(self._ticker) + '\n' +
                        'features \t' + str(self._features) + '\n' +
                        'window_len \t' + str(self._window_len) + '\n' +
                        'pred_len \t' + str(self._pred_len) + '\n' +
                        'valid_len \t' + str(self._valid_len) + '\n' +
                        'preproc_scaler \t' + str(self._preproc_scaler) + '\n' +
                        'out_class_type \t' + str(self._out_class_type) + '\n' +
                        'out_type \t' + str(self._out_type) + '\n' +
                        'n_out_class \t' + str(self._n_out_class) + '\n' +
                        'start_date \t' + str(self._start_date) + '\n')# +
                        #'end_date \t' + str(self._end_date) + '\n')
        if self._end_date == 'current':
            returnString = returnString + 'end_date \t' + str(datetime.date.today()) + '\n'
        else:
            returnString = returnString + 'end_date \t' + str(self._end_date) + '\n'
        return returnString
    
    @property
    def identify(self):
        return self._identify
    @identify.setter
    def identify(self, value):
        self._identify = value
        
    @property
    def save_folder(self):
        return self._save_folder
    @save_folder.setter
    def save_folder(self, value):
        self._save_folder = value
        
    @property
    def save(self):
        return self._save
    @save.setter
    def save(self, value):
        self._save = value
        
    @property
    def ticker(self):
        return self._ticker
#     @ticker.setter
#     def ticker(self, value):
#         self._ticker = value
        
    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value):
        self._features = value
        
    @property
    def window_len(self):
        return self._window_len
    @window_len.setter
    def window_len(self, value):
        self._window_len = value
        
    @property
    def pred_len(self):
        return self._pred_len
    @pred_len.setter
    def pred_len(self, value):
        self._pred_len = value
        
    @property
    def valid_len(self):
        return self._valid_len
    @valid_len.setter
    def valid_len(self, value):
        self._valid_len = value
        
    @property
    def preproc_scaler(self):
        return self._preproc_scaler
    @preproc_scaler.setter
    def preproc_scaler(self, value):
        self._preproc_scaler = value
        
    @property
    def out_class_type(self):
        return self._out_class_type
    @out_class_type.setter
    def out_class_type(self, value):
        self._out_class_type = value
        
    @property
    def n_out_class(self):
        return self._n_out_class
    @n_out_class.setter
    def n_out_class(self, value):
        self._n_out_class = value
    
    @property
    def out_type(self):
        return self._out_type
    @out_type.setter
    def out_type(self, value):
        self._out_type = value
        
    @property
    def start_date(self):
        return self._start_date
    @start_date.setter
    def start_date(self, value):
        self._start_date = value
        
    @property
    def end_date(self):
        if self._end_date == 'current':
            return str(datetime.date.today())
        else:
            return self._end_date
    @end_date.setter
    def end_date(self, value):
        self._end_date = value
        
    @property
    def n_features(self):
        return len(list(self.features))

class SP_RNN_LSTM_Paras(SP_Global_Paras):
    
    def __init__(self, name, ticker):
        super(SP_RNN_LSTM_Paras, self).__init__(name, ticker = ticker)

        # ------------- LSTM -------------
        self._batch_size = 128
        self._epoch = 10
        self._validation_split = .05
        self._model = {
            'hidden_layers' : [120, 60, 30],
            'dropout' : [0.5, 0.5, 0.3],
            'activation' : ['relu', 'relu', 'relu'],
            'out_layer' : 1,
            'out_activation' : 'linear',
            'loss' : 'mse',
            'optimizer' : 'rmsprop'
        }
    
    def __str__(self):
        returnString = (super(SP_RNN_LSTM_Paras, self).__str__() + '\n' +
                        '%%%%%%%%%% DUMP SP_RNN_LSTM_Paras %%%%%%%%%%\n' + 
                        'batch_size \t' + str(self._batch_size) + '\n' +
                        'epoch \t' + str(self._epoch) + '\n' +
                        'validation_split \t' + str(self._validation_split) + '\n' +
                        
                        'hidden_layers \t' + str(self._model['hidden_layers']) + '\n'
                        'dropout \t' + str(self._model['dropout']) + '\n'
                        'activation \t' + str(self._model['activation']) + '\n'
                        'out_layer \t' + str(self._model['out_layer']) + '\n'
                        'out_activation \t' + str(self._model['out_activation']) + '\n'
                        'loss \t' + str(self._model['loss']) + '\n'
                        'optimizer \t' + str(self._model['optimizer']) + '\n'
                       )
        return returnString
        
    @property
    def batch_size(self):
        return self._batch_size
    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        
    @property
    def epoch(self):
        return self._epoch
    @epoch.setter
    def epoch(self, value):
        self._epoch = value
    
    @property
    def validation_split(self):
        return self._validation_split
    @validation_split.setter
    def validation_split(self, value):
        self._validation_split = value
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value


