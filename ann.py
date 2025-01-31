#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:24:28 2024

pip install torch torchvision scikit-learn matplotlib

@author: ernesto.acosta
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
from sklearn.metrics import confusion_matrix
import time

from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from helpers import qcprint, TrainingResults

# Define the neural network
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)

        return x

    # Train the model
    def trainme(self, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=100):
        ann_training_begin = time.time()
        # Define the loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
        # Training loop
        for epoch in range(epochs):
            self.train()
            
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                qcprint(LOG_LEVEL_INFO, f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
        # Evaluate on test data
        self.eval()
        with torch.no_grad():
            test_outputs = self(X_test)
            predictions = (test_outputs > 0.5).float()  # Threshold at 0.5
            
            conf_matrix = confusion_matrix(y_test, predictions.numpy())
            TN, FP, FN, TP = conf_matrix.ravel()
        
        ann_training_end = time.time()
        
        best_params = [param.data.clone() for param in self.parameters()]
        best_params_str = json.dumps([param.tolist() for param in best_params])
        
        return TrainingResults(angleValues=best_params_str, trainingTime=ann_training_end-ann_training_begin,
                               truePositive=TP, falsePositive=FP, trueNegative=TN, falseNegative=FN,
                               totalIterations=epochs)
    
    def test(self, angles, validation_data, validation_labels, realhw=True):  # realhw is ignored here, overwritten param not used in ANN
        ann_validation_begin = time.time()
        
        self.eval()
        with torch.no_grad():                
            test_outputs = self(validation_data)
            predictions = (test_outputs > 0.5).float()  # Threshold at 0.5
            conf_matrix = confusion_matrix(validation_labels, predictions.numpy())
            TN, FP, FN, TP = conf_matrix.ravel()
        
        ann_validation_end = time.time()
            
        return TrainingResults(trainingTime= ann_validation_end-ann_validation_begin,
                               truePositive=TP, falsePositive=FP, trueNegative=TN, falseNegative=FN,
                               totalIterations=0)
        