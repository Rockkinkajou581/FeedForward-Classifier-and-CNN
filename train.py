import numpy as np
import dataloader as dl

def train(input_data, true_data, nn: NueralNetwork):
    X_training, Y_training, X_validation, Y_validation = dl.split(input_data, true_data)
    epoch = 5
    validation_best = None
    for i in range(epoch):
        X, Y = dl.make_batches(X_training, Y_training)
        for X,Y in zip(X,Y):
            nn.gradient_descent(X, Y)
        validation_loss = nn.validate(X_validation, Y_validation)

        if validation_loss < validation_best or validation_best is None:
            validation_best = validation_loss
            best_W = [x for ]
    
    


