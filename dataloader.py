
import numpy as np


#input list and true_list are row stacekd matrices with the examples 
def make_batches(input_list, true_list):
    batch_size = 32
    shuffle = np.random.permutation(len(input_list))
    X = input_list[shuffle]
    Y = true_list[shuffle]

    X_batch = []
    Y_batch = []
    for i in range(0, len(X), batch_size):
        if(i + batch_size > len(X)):
            break
        X_batch.append(X[i:batch_size + i:1])
        Y_batch.append(Y[i:batch_size + i:1])

    return X_batch, Y_batch

def split(input_list, true_list):
    validation = int(0.20 * len(input_list))
    shuffle = np.random.permutation(len(input_list))
    shuffle = shuffle[:validation]
    mask = np.ones(len(input_list), dtype= bool)
    mask[shuffle] = False

    X_validation = input_list[shuffle]
    Y_validation = true_list[shuffle]

    X_training = input_list[mask]
    Y_training = true_list[mask] 
    return X_training, Y_training, X_validation, Y_validation


