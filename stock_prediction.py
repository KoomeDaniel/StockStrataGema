print("Executing stock_prediction.py")

# Import necessary libraries and modules
import datetime as dt
import yfinance as yf
import preprocessor as p
import re
import pandas as pd
import numpy as np
import math, random
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nltk
nltk.download('punkt')
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

class StockAnalyzer:
    def __init__(self, quote,timeframe):
        self.quote = quote
        self.today_stock=None
        self.timeframe = timeframe

    def get_historical(self):
        try:
            end = datetime.now()

            if self.timeframe == 'Daily':
                start = datetime(end.year - 3, end.month, end.day)  # Download data for the previous 3 years
            elif self.timeframe == 'Weekly':
                start = datetime(end.year - 5, end.month, end.day)  # Download data for the previous 5 years
            elif self.timeframe == 'Monthly':
                start = datetime(1900, 1, 1)  # Download data for the max available period for monthly data
            elif self.timeframe == 'Yearly':
                start = datetime(1900, 1, 1)  # Download data for the max available period for yearly data
            else:
                raise ValueError("Invalid timeframe")

            data = yf.download(self.quote, start=start, end=end)
            if self.timeframe == 'Weekly':
                data = data.resample('5D').last()  # Resample to 5-day intervals for weekly
            elif self.timeframe == 'Monthly':
                data = data.resample('M').last()   # Resample to monthly
            elif self.timeframe == 'Yearly':
                data = data.resample('Y').last()   # Resample to yearly

            df = pd.DataFrame(data=data)
            df.to_csv(''+self.quote + self.timeframe + '.csv')
            if df.empty:
                ts = TimeSeries(key='6ET31V3K79OLGGJO', output_format='pandas')
                data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+self.quote, outputsize='full')
                if isinstance(data, str):
                    print("Error fetching data:", data)
                else:
                    data = data.head(503).iloc[::-1]
                data = data.reset_index()
                df = pd.DataFrame()
                df['Date'] = data['date']
                df['Open'] = data['1. open']
                df['High'] = data['2. high']
                df['Low'] = data['3. low']
                df['Close'] = data['4. close']
                df['Adj Close'] = data['5. adjusted close']
                df['Volume'] = data['6. volume']
                df.to_csv(''+self.quote+'.csv', index=False)
        except Exception as e:
            print("Error fetching historical data:", str(e))
            

    def arima_algorithm(self, df):
        try:
            unique_vals = df["Code"].unique()
            df = df.set_index("Code")

            def parser(x):
                return datetime.strptime(x, '%Y-%m-%d')

            def arima_model(train, test):
                history = [x for x in train]
                predictions = list()
                for t in range(len(test)):
                    model = ARIMA(history, order=(6, 1, 0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                return predictions

            arima_preds = []
            errors_arima = []

            for company in unique_vals[:10]:
                data = (df.loc[company, :]).reset_index()
                data['Price'] = data['Close']
                quantity_date = data[['Price', 'Date']]
                quantity_date.index = quantity_date['Date'].map(lambda x: parser(x))
                quantity_date.loc[:, 'Price'] = quantity_date['Price'].map(lambda x: float(x))
                quantity_date = quantity_date.fillna(quantity_date.bfill())
                quantity_date = quantity_date.drop(['Date'], axis=1)
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(quantity_date)
                plt.savefig('static/Trends.png')
                plt.close(fig)
                quantity = quantity_date.values
                size = int(len(quantity) * 0.80)
                train, test = quantity[0:size], quantity[size:len(quantity)]

                if len(test) == 0:
                    # If test set is empty, skip this company
                    continue

                predictions = arima_model(train, test)

                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(test,label='Actual Price')
                plt.plot(predictions,label='Predicted Price')
                plt.legend(loc=4)
                plt.savefig('static/ARIMA.png')
                plt.close(fig)
                print()
                print("##############################################################################")

                arima_pred = predictions[-2]
                print("Tomorrow's", company," Closing Price Prediction by ARIMA:",arima_pred)
                #rmse calculation
                error_arima = math.sqrt(mean_squared_error(test, predictions))
                print("ARIMA RMSE:",error_arima)
                print("##############################################################################")
                arima_preds.append(arima_pred)
                errors_arima.append(error_arima)

            if not arima_preds or not errors_arima:
                # If all companies have empty test sets, return None
                return None, None

            # Return the average prediction and error for all companies
            arima_pred_avg = np.mean(arima_preds)
            error_arima_avg = np.mean(errors_arima)

            return arima_pred_avg, error_arima_avg

        except Exception as e:
            print("Error in ARIMA algorithm:", str(e))

    def lstm_algorithm(self, df):
        try:
            dataset_train=df.iloc[0:int(0.8*len(df)),:]
            dataset_test=df.iloc[int(0.8*len(df)):,:]
            training_set=df.iloc[:,4:5].values

            from sklearn.preprocessing import MinMaxScaler
            sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
            training_set_scaled=sc.fit_transform(training_set)

            X_train = []
            y_train = []
            for i in range(7,len(training_set_scaled)):
                X_train.append(training_set_scaled[i-7:i,0])
                y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_forecast=np.array(X_train[-1,1:])
            X_forecast=np.append(X_forecast,y_train[-1])

            X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
            X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))

            from keras.models import Sequential
            from keras.layers import Dense, Dropout, LSTM

            regressor = Sequential()
            regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1], 1)))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.1))

            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.1))

            regressor.add(Dense(units=1))

            regressor.compile(optimizer='adam', loss='mean_squared_error')

            regressor.fit(X_train,y_train,epochs=25,batch_size=32 )

            real_stock_price=dataset_test.iloc[:,4:5].values
            # Make predictions
            dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
            testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
            testing_set=testing_set.reshape(-1,1)

            testing_set=sc.transform(testing_set)

            X_test=[]
            for i in range(7,len(testing_set)):
                X_test.append(testing_set[i-7:i,0])
                #Convert list to numpy arrays
            X_test=np.array(X_test)

            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

            predicted_stock_price=regressor.predict(X_test)
            predicted_stock_price=sc.inverse_transform(predicted_stock_price)

            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(real_stock_price,label='Actual Price')  
            plt.plot(predicted_stock_price,label='Predicted Price')
            
            plt.legend(loc=4)
            plt.savefig('static/LSTM.png')
            plt.close(fig)

            error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

            forecasted_stock_price=regressor.predict(X_forecast)

            forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
            lstm_pred=forecasted_stock_price[0,0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",self.quote," Closing Price Prediction by LSTM: ",lstm_pred)
            print("LSTM RMSE:",error_lstm)
            print("##############################################################################")
            return lstm_pred,error_lstm

        except Exception as e:
            print("Error in LSTM algorithm:", str(e))

    def linear_regression_algorithm(self, df):
        try:
            forecast_out = int(7)
            df['Close after n days'] = df['Close'].shift(-forecast_out)
            df_new = df[['Close', 'Close after n days']]

            y = np.array(df_new.iloc[:-forecast_out, -1])
            y = np.reshape(y, (-1, 1))
            X = np.array(df_new.iloc[:-forecast_out, 0:-1])
            X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])

            X_train = X[0:int(0.8 * len(df)), :]
            X_test = X[int(0.8 * len(df)):, :]
            y_train = y[0:int(0.8 * len(df)), :]
            y_test = y[int(0.8 * len(df)):, :]

            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            X_to_be_forecasted = sc.transform(X_to_be_forecasted)

            from sklearn.linear_model import LinearRegression

            clf = LinearRegression(n_jobs=-1)
            clf.fit(X_train, y_train)

            y_test_pred = clf.predict(X_test)
            y_test_pred = y_test_pred * (1.04)
            import matplotlib.pyplot as plt2
            fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
            plt2.plot(y_test,label='Actual Price' )
            plt2.plot(y_test_pred,label='Predicted Price')
            
            plt2.legend(loc=4)
            plt2.savefig('static/LR.png')
            plt2.close(fig)
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

            forecast_set = clf.predict(X_to_be_forecasted)
            forecast_set = forecast_set * (1.04)
            mean = forecast_set.mean()
            lr_pred = forecast_set[0, 0]
            print()
            print("##############################################################################")
            print("Tomorrow's ",self.quote," Closing Price Prediction by Linear Regression: ",lr_pred)
            print("Linear Regression RMSE:",error_lr)
            print("##############################################################################")

            return df, lr_pred, forecast_set, mean, error_lr
        except Exception as e:
            print("Error in Linear Regression algorithm:", str(e))

    

    def recommending(self, df, mean):
        try:
            close_price = self.today_stock.iloc[-1]['Close']
            if close_price < mean * 0.9:
                idea = "RISE"
                decision = "STRONGLY BUY"
            elif close_price < mean:
                idea = "RISE"
                decision = "BUY"
            elif close_price > mean * 1.1:
                idea = "FALL"
                decision = "STRONGLY SELL"
            elif close_price > mean:
                idea = "FALL"
                decision = "SELL"
            else:
                idea = "STABLE"
                decision = "HOLD"

            return idea, decision
        except Exception as e:
            print("Error in recommending:", str(e))

    def process_data(self):
        try:
            self.get_historical()

            df = pd.read_csv(''+self.quote + self.timeframe + '.csv')
            self.today_stock = df.iloc[-1:] 
            df = df.dropna()

            code_list = []
            for i in range(0, len(df)):
                code_list.append(self.quote)

            df2 = pd.DataFrame(code_list, columns=['Code'])
            df2 = pd.concat([df2, df], axis=1)
            df = df2

            print("quote:", self.quote)

            arima_pred_avg, error_arima_avg = self.arima_algorithm(df)
            print("ARIMA results:", arima_pred_avg, error_arima_avg)

            lstm_pred, error_lstm = self.lstm_algorithm(df)
            print("LSTM results:", lstm_pred, error_lstm)

            df, lr_pred, forecast_set, mean, error_lr = self.linear_regression_algorithm(df)
            print("Linear Regression results:", lr_pred, forecast_set, mean, error_lr)


            idea, decision = self.recommending(df, mean)
            print("Recommendation results:", idea, decision)

            forecast_set = forecast_set.round(2)
            self.today_stock = self.today_stock.round(2) 


            result = {
                "quote": self.quote,
                "arima_pred": round(arima_pred_avg, 2),
                "lstm_pred": round(lstm_pred, 2),
                "lr_pred": round(lr_pred, 2),
                "open_s": self.today_stock['Open'].to_string(index=False),
                "close_s": self.today_stock['Close'].to_string(index=False),
                "adj_close": self.today_stock['Adj Close'].to_string(index=False),
                "idea": idea,
                "decision": decision,
                "high_s": self.today_stock['High'].to_string(index=False),
                "low_s": self.today_stock['Low'].to_string(index=False),
                "vol": self.today_stock['Volume'].to_string(index=False),
                "forecast_set": forecast_set,
                "error_lr": round(error_lr, 2),
                "error_lstm": round(error_lstm, 2),
                "error_arima": round(error_arima_avg, 2)
            }

            print("Results:")
            print(result)

            return result

        except Exception as e:
            print("Error in processing data:", str(e))
            return {
                "error_message": str(e),
                "quote": self.quote
            }
