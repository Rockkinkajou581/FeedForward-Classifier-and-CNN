import numpy as np

def ReLu(z):
    return np.maximum(0,z)

def ReLu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    offset = z - np.max(z, axis = 1, keepdims= True)
    s = np.exp(offset)
    sums = np.sum(s, axis = 1, keepdims= True)
    return s / sums 

    
