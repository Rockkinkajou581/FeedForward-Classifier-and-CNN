import numpy as np
from activations import ReLu, ReLu_grad, softmax

class NueralNetwork:
    #layer sizes of the form [720, 350, ... ] representing input dimensions and output dimensions for each layer
    def __init__(self, layer_sizes, batch_size):
        self.batch_size = batch_size
        self.weights = []
        self.bias = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.normal(loc= 0.0, scale=np.sqrt(2 / layer_sizes[i]), size = (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.bias.append(b)
        
        self.activation = []
        self.preactivation = []

        self.grad = []
        self.grad_W = []
        self.grad_b = []

    def feed_forward(self, A_zero):
        self.activation = [A_zero]
        self.preactivation = []

        A = A_zero
        for i in range(len(self.weights)):
            Z_i = A @ self.weights[i] + self.bias[i]
            self.preactivation.append(Z_i)
            if(i == len(self.weights) - 1):
                A = softmax(Z_i)
            else:
                A = ReLu(Z_i)
            self.activation.append(A)
    
    def back_prop(self, A_zero, Y_true):
        self.grad_b = []
        self.grad_W = []

        self.feed_forward(A_zero)
        grad_base = (1 / self.batch_size) * (self.activation[-1] - Y_true)
        
        L = len(self.weights)

        self.grad = [None] * L
        self.grad[L - 1] = grad_base


        for i in range(L - 2, -1, -1):
            self.grad[i] = (self.grad[i+1] @ self.weights[i+1].T) * ReLu_grad(self.preactivation[i])

        for i in range(L):
            grad_W_i = np.transpose(self.activation[i]) @ self.grad[i] 
            grad_b_i = np.sum(self.grad[i], axis= 0, keepdims=True)

            self.grad_b.append(grad_b_i)
            self.grad_W.append(grad_W_i)

        