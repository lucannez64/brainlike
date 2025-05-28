import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, InputLayer, Lambda
from tensorflow.keras import backend as K

# Target spike function: narrow pulse of given width and height
def spike_target(x, width=0.01, height=1.0):
    y = np.zeros_like(x)
    y[np.abs(x) <= width] = height
    return y

# 1) Difference of sigmoids approximation
def build_sigmoid_diff_model(k=100.0, b=0.01, A=1.0):
    def diff_sigmoid(x):
        return A * (1/(1+K.exp(-k*(x + b))) - 1/(1+K.exp(-k*(x - b))))
    model = Sequential([
        InputLayer(input_shape=(1,)),
        Lambda(diff_sigmoid, name='diff_sigmoid')
    ], name='SigmoidDiff')
    model.compile(optimizer='adam', loss='mse')
    return model

# 2) ReLU-based MLP approximation
def build_relu_mlp():
    model = Sequential([
        InputLayer(input_shape=(1,)),
        Dense(64, activation='relu', kernel_initializer='he_normal', use_bias=False),
        Dense(64, activation='relu', kernel_initializer='he_normal', use_bias=False),
        Dense(1, activation='linear')
    ], name='ReLU_MLP')
    model.compile(optimizer='adam', loss='mse')
    return model

# 3) Custom Gaussian activation approximation
def build_custom_gaussian(sigma=0.01, mu=0.0, A=1.0):
    def gaussian(x):
        return A * K.exp(-K.square(x - mu) / (2 * K.square(sigma)))
    model = Sequential([
        InputLayer(input_shape=(1,)),
        Lambda(gaussian, name='Gaussian')
    ], name='CustomGaussian')
    model.compile(optimizer='adam', loss='mse')
    return model

# 4) Simple RNN transient response approximation
def build_rnn_model():
    model = Sequential([
        InputLayer(input_shape=(1, 1)),
        SimpleRNN(10, activation='tanh', kernel_initializer='glorot_normal'),
        Dense(1, activation='linear')
    ], name='SimpleRNN')
    model.compile(optimizer='adam', loss='mse')
    return model

# Demo: train and plot all four approximators
if __name__ == '__main__':
    x = np.linspace(-0.1, 0.1, 1001)
    y = spike_target(x, width=0.005, height=1.0)
    X = x.reshape(-1, 1)

    models = [
        build_sigmoid_diff_model(),
        build_relu_mlp(),
        build_custom_gaussian(),
    ]
    # RNN needs 3D input
    rnn = build_rnn_model()
    models.append(rnn)
    names = ['SigmoidDiff', 'ReLU_MLP', 'CustomGaussian', 'SimpleRNN']

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'k--', label='Target Spike')

    for model, name in zip(models, names):
        if name == 'SimpleRNN':
            X_in = X.reshape(-1, 1, 1)
            model.fit(X_in, y, epochs=50, verbose=0)
            y_pred = model.predict(X_in).flatten()
        elif name in ['ReLU_MLP']:
            model.fit(X, y, epochs=50, verbose=0)
            y_pred = model.predict(X).flatten()
        else:
            # SigmoidDiff and CustomGaussian are analytic
            y_pred = model.predict(X).flatten()
        plt.plot(x, y_pred, label=name)

    plt.xlabel('x'); plt.ylabel('y')
    plt.title('Spike Function Approximations')
    plt.legend(); plt.grid(True)
    plt.show()
