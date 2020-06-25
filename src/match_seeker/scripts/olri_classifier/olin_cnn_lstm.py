from tensorflow import keras

Tx = 30
n_x = 3
n_s = 64
X = keras.layers.Input(shape=(Tx, n_x))
print("This is the X", X)
print("This is the shape", X.shape)
s, a, c = keras.layers.LSTM(n_s, return_sequences=True, return_state=True)(X)
print("This is s", s, "This is a ", a, "This is c", c)
model_LSTM = keras.Model(inputs=X, outputs=[s, a, c])
model_LSTM.summary()
