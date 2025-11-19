import numpy as np


def predict(X, w, b):
    return X * w + b


def loss(X, Y, w, b):
    return np.average((predict(X, w, b) - Y) ** 2)

def gradient(X, Y, w, b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average((predict(X, w, b) - Y))
    return (w_gradient, b_gradient)

def train(X, Y, iterations, lr):
    w = b =0
    for i in range(iterations):
        print(f"Iteration {i} => Loss: {loss(X, Y, w, b)}")
        w_gradient, b_gradient = gradient(X, Y, w, b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

X, Y = np.loadtxt("./03_gradient/pizza.txt", skiprows=1, unpack=True)
w, b = train(X, Y, iterations=20000, lr=0.001)
print(f"\nw={w}, {b}")
print(f"Prediction: x={X}, y={predict(20, w, b)}")
