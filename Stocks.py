import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Data Collection
# For demonstration, let's use a sample dataset. You should replace this with actual stock data.
df = pd.read_csv('path_to_your_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Data Preprocessing
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 4: Model Building
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Step 5: Training the Model
model.compile(optimizer=Adam(), loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=1, epochs=1)

# Step 6: Evaluation
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(df.index, data, label='Actual Stock Price')
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict
plt.plot(df.index, train_predict_plot, label='Train Predict')

test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_step*2)+1:len(data)-1, :] = test_predict
plt.plot(df.index, test_predict_plot, label='Test Predict')

plt.legend()
plt.show()

# Step 7: Prediction
# Use the model to make future predictions
last_60_days = scaled_data[-60:]
X_future = last_60_days.reshape(1, time_step, 1)
future_prediction = model.predict(X_future)
future_prediction = scaler.inverse_transform(future_prediction)
print(f'Predicted Stock Price: {future_prediction[0][0]}')
