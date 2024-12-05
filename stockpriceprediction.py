import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the stock price data
data = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Prompt the user to choose which column to predict
print("Available columns:")
print(data.columns.tolist())
selected_column = input("Enter the column name to predict: ")

# Ensure the selected column exists
if selected_column not in data.columns:
    raise ValueError(f"Column '{selected_column}' not found in the dataset.")

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data[selected_column] = scaler.fit_transform(data[[selected_column]].values)

# Function to prepare the data for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Define the look_back period
look_back = 60

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[0:train_size], data.iloc[train_size:len(data)]

# Prepare the data for LSTM
trainX, trainY = create_dataset(train_data[[selected_column]].values, look_back)
testX, testY = create_dataset(test_data[[selected_column]].values, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=1)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions back to original scale
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Evaluate model performance
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print(f'{selected_column} Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print(f'{selected_column} Test Score: %.2f RMSE' % (testScore))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(data[[selected_column]].values), label='Actual Data')
plt.plot(np.arange(look_back, look_back + len(trainPredict)), trainPredict, label='Train Predictions')
plt.plot(np.arange(look_back + len(trainPredict) + 1, look_back + len(trainPredict) + len(testPredict) + 1), testPredict, label='Test Predictions')
plt.title(f'{selected_column} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
