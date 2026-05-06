import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data
y = boston.target.astype(float)
print("Dataset Shape: ", X.shape)
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(y_test.shape)

# Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
    ])

# Compile
model.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mae']
)

# Train
history = model.fit(
    X_train, y_train,
    epochs = 10,
    batch_size = 6,
    validation_split = 0.2,
    verbose=1
)

# Evaluate

loss, mae = model.evaluate(X_test, y_test)

print("\nTest MSE: ", loss)
print("Test MAE: ", mae)

y_pred = model.predict(X_test)

print("\nSample Predictions: ")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {y_pred[i][0]:.2f}")

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model loss")
plt.legend()
plt.show()