import numpy as np

def categorical_cross_entropy(y_hat, y_true):
    # small epsilon inside log to avoid log(0)
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_hat + eps), axis=1))