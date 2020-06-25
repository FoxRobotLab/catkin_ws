from tensorflow import keras

Tx = 30
n_x = 3
n_s = 64
X = keras.layers.Input(shape=(Tx, n_x))
s, a, c = keras.layers.LSTM(n_s, return_sequences=True, return_state=True)(X)
model_LSTM = keras.layers.models(inputs=X, outputs=[s, a, c])
model_LSTM.summary()
