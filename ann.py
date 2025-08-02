#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 23:24:28 2024

pip install torch torchvision scikit-learn matplotlib

@author: ernesto.acosta
"""

import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import config as cf
import helpers as hlp

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

    def plot_loss(self, train_loss_history, val_loss_history, epochs):
        # Plot loss
        epochs = int(epochs)
        
        plt.figure(figsize=(8, 6))

        plt.plot(range(1, epochs + 1), train_loss_history, marker='o', linestyle='-', label='Training Loss')
        plt.plot(range(1, epochs + 1), val_loss_history, marker='s', linestyle='--', label='Test Loss')
    
        plt.title('ANN Training and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        timestamp = int(time.time())
        loss_plot_path = f"{cf.LOG_DIR}ann_trainandtest_loss_{timestamp}.png"
        plt.savefig(loss_plot_path)
        plt.close()  # close the plot to avoid memory issues

    # Train the model
    def trainme(self, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=100):
        ann_training_begin = time.time()
        # Define the loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
        # Track loss values
        train_loss_history = []
        test_loss_history = []
    
        # Training loop
        for epoch in range(epochs):
            hlp.print_avail_resources()
            self.train()
            
            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
    
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Store loss
            train_loss_history.append(loss.item())
            
            # --- Validation step ---
            self.eval()
            with torch.no_grad():
                test_outputs = self(X_test)
                test_loss = criterion(test_outputs, y_test)
                test_loss_history.append(test_loss.item())
        
            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                hlp.qcprint(cf.LOG_LEVEL_INFO, f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
        # Plot loss
        self.plot_loss(train_loss_history, test_loss_history, epochs)
        #self.plot_loss(test_loss_history, epochs, "Validation")
    
        # Evaluate on test data
        #test_criterion = nn.BCELoss() 

        self.eval()
        with torch.no_grad():
            test_outputs = self(X_test)
            #test_loss = test_criterion(test_outputs, y_test)
            #test_loss_history.append(test_loss.item())
   
            predictions = (test_outputs > 0.5).float()  # Threshold at 0.5
            conf_matrix = confusion_matrix(y_test, predictions.numpy())
            TN, FP, FN, TP = conf_matrix.ravel()
        
        ann_training_end = time.time()
        
        best_params = [param.data.clone() for param in self.parameters()]
        best_params_str = json.dumps([param.tolist() for param in best_params])
        
        return hlp.TrainingResults(angleValues=best_params_str, trainingTime=ann_training_end-ann_training_begin,
                               truePositive=TP, falsePositive=FP, trueNegative=TN, falseNegative=FN,
                               totalIterations=epochs)
    
    def test(self, angles, validation_data, validation_labels, realhw=True):  # realhw is ignored here, overwritten param not used in ANN
        ann_validation_begin = time.time()
        
        # Track loss values
        criterion = nn.BCELoss() 
        validation_loss_history = []
        
        self.eval()
        with torch.no_grad():                
            validation_outputs = self(validation_data)
            validation_loss = criterion(validation_outputs, validation_labels)
            validation_loss_history.append(validation_loss.item())
            
            predictions = (validation_outputs > 0.5).float()  # Threshold at 0.5
            conf_matrix = confusion_matrix(validation_labels, predictions.numpy())
            TN, FP, FN, TP = conf_matrix.ravel()
        
        ann_validation_end = time.time()
            
        return hlp.TrainingResults(trainingTime= ann_validation_end-ann_validation_begin,
                               truePositive=TP, falsePositive=FP, trueNegative=TN, falseNegative=FN,
                               totalIterations=0)
        