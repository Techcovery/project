"""
#Import Dependencies
-Numpy: for numerical calculations
-Pandas: to load the data and modify it
-Matplotlib: to plot the data
-Quandllib : For Data Set*/
"""    
import quandl, math
import numpy as np
import pandas as pd
# from sklearn import preprocessing, cross_validation, svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
quandl.ApiConfig.api_key = "mNfysBJ5dcMNfkvB74ny"
style.use('ggplot')

"""
## Get the data set - Amazon from Quandl and View
-Down load the dataset for the range of date required
-View the data of the data frame
"""
start = datetime.datetime(2016,1,1)
end = datetime.date.today()
s = "AMZN"
df = quandl.get("WIKI/" + s, start_date=start, end_date=end)
print(df.head())
"""
## Transform the data to High-Low percentage and Percentage change calculations
-High- Low Percentage : df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
-Percentage Change : (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
-View the Data
"""
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())
"""
##Select the Features and Labels
-Adj.Close is the forecast column
-To forecast out 1% of the entire length of the dataset - use forecast_out = int(math.ceil(0.01 * len(df)))
-Create the lable df['label'] for forecast data
-Features - features are a bunch of the current values
-Label - Price
-Forecast - 1% of the entire length of the dataset out
"""
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
"""
## Features - Preprocessing
-X (features), as our entire dataframe EXCEPT for the label column
-Convert to a numpy array
-Preprocess - features in machine learning to be in a range of -1 to 1 
-Create Label y
"""
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])
print(y)
"""
##Classifier : Linear Regression - Train and Test Data
-User Linear Regression classifier
-Train the machine learning classifier (75 % of data) using fit
-Test using 25% data to get Confidence Score
-n_jobs= -1 - algorithm will use all available threads
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
"""
Forecasting and Predicting
X_lately - variable contains the most recent features, which we're going to predict against
Forecast_set - array of forecasts
Start the forecasts as tomorrow (predict 10% out into the future, saved that last 10% of our data to do this)
Grab the last day in the dataframe, and begin assigning each new forecast to a new day
Next day - one_day is 86,400 seconds. Now we add the forecast to the existing dataframe
"""
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
print(df)
"""
Plot the Date VS Price
"""
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=1)
datenow = datetime.datetime.now()
dstart = datetime.datetime(2016,4,1)
plt.xlim(dstart,datenow)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
