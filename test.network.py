import pytest
from sklearn.datasets import load_iris
import train
from network import NueralNetwork
import numpy as np


def test_on_iris():
    data = load_iris()
    X = data.data
    Y = np.zeros(150, 3)
    Y[np.arrange(150), data.target] = 1

    layer_sizes = [4, 3, 3, 5, 3]

    nn = NueralNetwork(layer_sizes, 8)
    train.train(X, Y, nn)