# US Stock Market Prediction by LSTM  

Author: Chris Tsai  
E-mail: christsaizyt@gmail.com  
Feel free to contact me if you have any comments or suggestions.  
  
## Data  
1. Get data from quandl  
2. Features set = \[*'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'*\] (default feature set)  
3. Processing on time-series data  
  a). window length (*window_len*): append *window_len* days' historical feature set  
  b). label: prediction length (*pred_len*): predict the moving average close price for *pred_len* days later  
  c). known lately length (*known_lately_len*): To do validation  
4. Divied time-series data into three parts: *df, df_known_lately, df_lately*  
  
## Preprocessing 
1. Divide the data for each row into price and volume.  
2. Do standard normalization for price and volume.  
   *Here is an assumption: if winodw_len is large enough, it will be a Gaussian Distribution. Normalize to zero mean and unit variance.*  
3. Normalization for row data and some data reshape  
   *use sklearn preprocessing.StandardScaler()*  
4. Rearrange to original format  
5. Save the scaler_price (it is necessary while doing the inverse transformation later)  
6. *df -> X, y* (for train and test)  
   *df_known_lately -> X_known_lately, y_known_lately* (for evaluate the model)  
   *df_lately -> X_lately*  
  
## Cross validation    
- Split *(X, y)* into *(X_train, y_train) + (X_test, y_test)*  
- *(X_train, y_train)* is for training the LSTM model.  
- *(X_test, y_test)* is for test later.  
  
## LSTM model    
1. Build LSTM model with input_dim = 5(ohlcv)  
2. Here I use two hidden layers [120, 60] with dropout = 0.5, activation = 'relu'  
3. Output layer: activation = 'linear'   
4. Loss = 'mse', optimization = 'rmsprop'  
  
## Validation and prediction    
1. *df* for training / test  
2. *df_known_lately* - I used it to evaluate the model. Sometimes the model can get a good training/validation loss but could not achieve the same performance for df_known_lately  
3. *df_lately* - Predict the future trend.  
  
## Result & Performance    
1. Training and validation loss for LSTM model  
2. mse_test: validation on (*X_test, y_test*)  
3. mse_known_lately: validation on (*X_known_lately, y_known_lately*)  
4. if mse_test and mse_known_lately is close enough, (maybe) the model is good to predict for (*X_lately*).  
  
***Training and validation loss for LSTM model***  
![alt tag](https://github.com/christsaizyt/US_Stock_Market_Prediction_by_Machine-Deep_Learning/blob/master/NDAQ_training_curve.png)  
  
***mse_test:***  
############## validation on test data ##############   
scaled data mse:  0.034219994264  
  
***mse_known_lately:***   
############## validation on known lately data ##############   
scaled data mse:  0.202054234164  
  
***Prediction***  
a_+10_d: actual 10 days moving avergae for 10 days later  
p_+10_d: predict 10 days moving avergae for 10 days later  
a_+10_d_diff: df['a_+10_d'] - df['close'] to see the future trend for 10 days later  
p_+10_d_diff: df['p_+10_d'] - df['close'] to see the future trend for 10 days later  
  
![alt tag](https://github.com/christsaizyt/US_Stock_Market_Prediction_by_Machine-Deep_Learning/blob/master/NDAQ_predictions.png)  



  
