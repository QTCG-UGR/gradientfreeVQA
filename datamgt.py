#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:32:59 2024

@author: ernesto.acosta
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(dataset_name):
    #1. Uploading the classical data
    dataset = []
    if (dataset_name == "iris"):
        dataset = load_iris()
        
        # For the sake of simplicity, we consider all the four features but only two classes
        dataset_data = dataset.data[:100, :4]
        dataset_target = dataset.target[:100]   # 0 or 1
    
    # Split into train and test subsets
    train_data, test_data, train_labels, test_labels = train_test_split(dataset_data, dataset_target, test_size=0.2,
                                                                        random_state=42)
    
    # Pre-process data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    
    return train_data, test_data, train_labels, test_labels