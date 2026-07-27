
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
        X_batch.append(np.vstack(X[i:batch_size + i:1]))
        Y_batch.append(np.vstack(Y[i:batch_size + i:1]))

    return X_batch, Y_batch

