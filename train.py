import numpy as np
import dataloader as dl

def train(input_data, true_data, nn):
    X_training, Y_training, X_validation, Y_validation = dl.split(input_data, true_data)
    max_epoch = 500
    validation_best = float('inf')
    best_W = None
    best_b = None
    patience = 0
    print(f"{'Epoch':<8}{'Loss':<10}{'Accuracy':<10}")

    for epoch in range(max_epoch):
        X, Y = dl.make_batches(X_training, Y_training, nn.batch_size)
        for X,Y in zip(X,Y):
            nn.gradient_descent(X, Y)
        validation_loss = nn.validate(X_validation, Y_validation)
        nn.evaluate_epoch(Y_validation, epoch, validation_loss)

        if validation_loss < validation_best:
            validation_best = validation_loss
            best_W = [W.copy() for W in nn.weights]
            best_b = [b.copy() for b in nn.bias]
            patience = 0
        else:
            patience += 1
            if (patience > 10):
                nn.weights = best_W
                nn.bias = best_b
                nn.evaluate_final(X_validation, Y_validation)
            
                return
    nn.weights = best_W
    nn.bias = best_b
    nn.evaluate_final(X_validation, Y_validation)

            
    


