# US Stock Market Prediction by LSTM  

Author: Chris Tsai,  
E-mail: christsaizyt@gmail.com  
Feel free to contact me if you have any comments or suggestions.  
  
# Data  
1. Get data from quandl  
2. Features set = ['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'] (default feature set)  
3. Processing on time-series data  
  for exmaple: 2010/1/1 - 2017/3/1 => X0, X1, X2, ..., Xn for each Xi = ['o', 'h', 'l', 'c', 'v']  
  a). window length(window_len): append window_len days's historical feature set  
  b). label: prediction length(pred_len): predict the moving average close price for pred_len days later  
  c). known lately length(known_lately_len): To do validation  
   
  Divied time-series data to three parts: df, df_known_lately, df_lately  
  for exmaple:   
  window_len = 120, pred_len = 10, known_lately_len = 20 (in this setting we use 6 months data to predict the trend of 2 weeks later.)  
  Some assumption: 10(working days) = 2 week, today = 2017/3/1  
  Roughly describe time interval  
    df_total: 2010/1/1 - 2017/3/1  
    df_total_append_more_historical_features : 2010/7/1 - 2017/3/1  
    df: 2010/7/1 - 2017/1/14 (with label data for training)  
    df_known_lately: 2017/1/15 - 2017/2/15 (with label data but not for training, for evaluate the model)  
    df_latyely: 2017/2/16 - 2017/3/1 (without label data)  
    
  df[i,:] = ['label',   
             'o_-120_d','h_-120_d','l_-120_d','c_-120_d','v_-120_d',  
             'o_-119_d','h_-119_d','l_-119_d','c_-119_d','v_-119_d',  
             ....  
             'o_-1_d','h_-1_d','l_-1_d','c_-1_d','v_-1_d',  
             'o_-0_d','h_-0_d','l_-0_d','c_-0_d','v_-0_d',  
            ]  
  
# Preprocessing    
1. dive the data for each row into price and volume.  
  price:  
  ['o_-120_d','h_-120_d','l_-120_d','c_-120_d','o_-119_d','h_-119_d','l_-119_d','c_-119_d',....,'o_-0_d','h_-0_d','l_-0_d','c_-0_d',]  
  volume:  
  ['v_-120_d','v_-119_d',...,'v_-0_d']  
2. Do standard normalization for price and volume.   
  Here is an assumption: if winodw_len is large enough, it will be a Gaussian Distribution. Normalize to zero mean and unit variance.  
3. Normalization for row data but need to do some data reshape  
  use sklearn preprocessing.StandardScaler()  
4. Rearrange to original format: ['o_-120_d','h_-120_d','l_-120_d','c_-120_d','v_-120_d', ....]  
5. Save the scaler_price (it will be need when do the inverse transformation later)  
6. df -> X, y (for train and test)  
   df_known_lately -> X_known_lately, y_known_lately (for evaluate the model)  
   df_lately -> X_lately  
  
# Cross validation    
split (X, y) into (X_train, y_train) + (X_test, y_test)  
(X_train, y_train) is for training the LSTM model.  
(X_test, y_test) is for test later.  
  
# LSTM model    
1. build LSTM model with input_dim = 5(ohlcv)  
2. Here I use two hidden layers [120, 60] with dropout = 0.5, activation = 'relu'  
3. output layer: activation = 'linear'   
4. loss = 'mse', optimization = 'rmsprop'  
  
# Prediction    
1. df for training / test  
2. df_known_lately - I used it to evaluate the model. Sometimes the model can get a good training/validation loss but could not achieve the same performance for df_known_lately  
3. df_lately - Predict the future trend.  
  
# Result & Performance    
1. Training and validation loss for LSTM model  
2. mse_test: validation on (X_test, y_test)  
3. mse_known_lately: validation on (X_known_lately, y_known_lately)  
4. if mse_test and mse_known_lately is close enough, (maybe) the model is good to predict for (X_lately).  
  
######################################################################################  
%%%%%%%%%% DUMP SP_Global_Paras %%%%%%%%%%  
ticker 	NDAQ  
features 	ohlcv  
window_len 	120  
pred_len 	10  
known_lately_len 	20  
preproc_scaler 	standard_scaler  
out_class_type 	regression  
out_type 	MA  
start_date 	2010-01-01  
end_date 	2017-03-03  
  
%%%%%%%%%% DUMP SP_RNN_LSTM_Paras %%%%%%%%%%  
batch_size 	128  
epoch 	100  
validation_split 	0.1  
hidden_layers 	[120, 60, 100]  
dropout 	[0.5, 0.5, 0.3]  
activation 	['relu', 'relu', 'relu']  
out_layer 	1  
out_activation 	linear  
loss 	mse  
optimizer 	rmsprop  
  
*Training and validation loss for LSTM model*  
![alt tag](https://github.com/christsaizyt/US_Stock_Market_Prediction_by_Machine-Deep_Learning/blob/master/NDAQ_training_curve.png)  
  
*mse_test:*  
############## validation on test data ##############   
scaled data mse:  0.034219994264  
  
*mse_known_lately:*   
############## validation on known lately data ##############   
scaled data mse:  0.202054234164  
  
*Prediction for X_lately*  
a_+10_d: actual 10 days moving avergae for 10 days later  
p_+10_d: predict 10 days moving avergae for 10 days later  
a_+10_d_diff: df['a_+10_d'] - df['close'] to see the future trend for 10 days later  
p_+10_d_diff: df['p_+10_d'] - df['close'] to see the future trend for 10 days later  
  
![alt tag](https://github.com/christsaizyt/US_Stock_Market_Prediction_by_Machine-Deep_Learning/blob/master/NDAQ_predictions.png)  

