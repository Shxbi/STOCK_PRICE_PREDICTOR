import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

  file_path = 'DATASET.csv'
df = pd.read_csv(file_path)
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10  # You may adjust this parameter based on your data
X, Y = create_dataset(data_normalized, look_back)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=100, batch_size=32)

predictions = model.predict(X)

predictions = scaler.inverse_transform(predictions)
Y = scaler.inverse_transform([Y])

rmse = np.sqrt(np.mean(np.square(predictions - Y)))
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(15, 6))
plt.plot(scaler.inverse_transform(data), label='Actual Stock Price')
plt.plot(np.arange(look_back, len(predictions) + look_back), predictions, label='Predicted Stock Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()
