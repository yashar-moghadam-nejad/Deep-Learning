import tensorflow as tf
from tensorflow import keras

x = tf.constant([1,2,3,4],dtype = float)
y = x * 3 + 5

# model definition
model = keras.Sequential([
    keras.layers.Dense(units=1,activation=None, input_shape=[1])
])
model.compile(optimizer='sgd',loss='mean_squared_error')

# model training
model.fit(x,y,batch_size=1,epochs=500)

# testing the model
x_input = tf.constant([50,30,200],dtype=float)
y_predict = model.predict(x_input)
y_predict

