!pip install yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

data = yf.download("GOOGL", start="2012-01-01", end="2020-01-01")

dataset = data[['Open']].values

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

train_size = int(len(dataset_scaled) * 0.8)

train_data = dataset_scaled[:train_size]
test_data = dataset_scaled[train_size - 60:]

X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = Sequential()

model.add(LSTM(50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

# Compile Model
model.compile(optimizer='adam',
              loss='mean_squared_error')

model.fit(X_train,
          y_train,
          epochs=5,
          batch_size=32)

X_test = []
y_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predicted_prices = model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

real_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

plt.figure(figsize=(12, 6))

plt.plot(real_prices, color='blue',
         label='Real Google Stock Price')

plt.plot(predicted_prices, color='red',
         label='Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

plt.show()