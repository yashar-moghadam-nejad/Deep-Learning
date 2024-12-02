import tensorflow as tf
from tensorflow import keras

# Input data
x = tf.constant([1, 2, 3, 4], dtype=float)
y = x * 3 + 5

# Model definition
model = keras.Sequential([
    keras.layers.Dense(units=1, activation=None, input_shape=[1])
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Model training
model.fit(x, y, batch_size=1, epochs=500)

# Testing the model
x_input = tf.constant([50, 30, 200], dtype=float)
x_input = tf.reshape(x_input, (-1, 1))  # Reshape to (3, 1)
y_predict = model.predict(x_input)

# Print the predictions
print("Predictions for inputs [50, 30, 200]:")
print(y_predict)