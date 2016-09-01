import numpy as np

import datasets

# randomly initializes a column vector of weights
# with length equal to the numbers of features
# (columns) of X
def random_weights(X):
    return np.random.random((X.shape[1], 1))

# predicts the resultant value for each row
# in the input X, given weights W. The prediction
# for a row is given by w_0*x_0 + w_1*x_1 + ...
def predict(W, X):
    return X.dot(W)

# cost function given weights W, input X,
# and expected output Y. Doesn't use regularization.
def cost(W, X, Y):
    return 1.0/(2*X.shape[0]) * np.sum((predict(W, X) - Y) ** 2)

# Run gradient descent with learning rate `alpha`
# with `iters` iterations
def grad_descent(W, X, Y, alpha, iters):
    for i in range(0, iters):
        new_W = W - alpha * 1.0/(X.shape[0]) * X.T.dot(predict(W, X) - Y)
        np.copyto(W, new_W)

X, Y = datasets.house_prices()
W = random_weights(X)
print("initial cost: ", cost(W, X, Y))
grad_descent(W, X, Y, 0.01, 100000)
print("final cost: ", cost(W, X, Y))
print(W)
