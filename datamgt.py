#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:32:59 2024

@author: ernesto.acosta
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from config import DS_LENGTH, DS_LENGTH_ALL, DS_SHUFFLE, DS_PCA
from config import DS_VALIDATE_SIZE, DS_TEST_SIZE, DS_BALANCE_UNDERSAMPLE, DS_BALANCE_OVERSAMPLE
from helpers import qcprint
from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_data(dataset_name):
    qcprint(LOG_LEVEL_INFO, f"loading dataset {dataset_name}")

    dataset = []
    ds_target_col = 'target'
    if (dataset_name == "iris"):
        dataset_iris = load_iris()
        ds_target_col = 'target'
        
        dataset = pd.DataFrame(dataset_iris.data, columns=dataset_iris.feature_names)
        dataset = dataset[:100]  # Frist 100 records correspond to first 2 classes (out for 3) in the dataset.
        dataset[ds_target_col] = dataset_iris.target # Add the target column, values 0 and 1 are for the first two classes corresponding to first 100 rows
    elif (dataset_name == "heart_disease"):
        dataset = pd.read_csv('data/heart.csv')
        ds_target_col = 'target'
    elif (dataset_name == "diabetes"):
        dataset = pd.read_csv('data/diabetes.csv')
        ds_target_col = 'Outcome'
    else:
        qcprint(LOG_LEVEL_ERROR, "Unknown dataset {dataset_name}, exit.")
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    qcprint(LOG_LEVEL_INFO, f"Initial dataset length: {len(dataset)}")
    
    # Shuffle Dataset
    if DS_SHUFFLE: 
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    # Cut dataset to desired length before balancing
    dataset = dataset[:DS_LENGTH if DS_LENGTH!=DS_LENGTH_ALL else len(dataset)] 
    
    # Balance Dataset
    if DS_BALANCE_UNDERSAMPLE ^ DS_BALANCE_OVERSAMPLE: #One and only one of the techniques specified
        try:
            X = dataset.loc[:, dataset.columns != ds_target_col]
            y = dataset[[ds_target_col]]
            if DS_BALANCE_UNDERSAMPLE: 
                undersampler = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = undersampler.fit_resample(X, y)
            if DS_BALANCE_OVERSAMPLE:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled_df = pd.DataFrame(y_resampled, columns=[ds_target_col])
                
            # Build back the dataset
            dataset = pd.concat([X_resampled_df, y_resampled_df], axis=1)
        except Exception as e:
            qcprint(LOG_LEVEL_ERROR, f"Could not apply PCA due to error: {e}")
    elif DS_BALANCE_UNDERSAMPLE and DS_BALANCE_OVERSAMPLE:
        qcprint(LOG_LEVEL_ERROR, "Can not apply Resampling since both undersample and oversample have True values")

    # Apply PCA to reduce features to 4
    if DS_PCA:  
        pca = PCA(n_components=4)
        dataset_data = pca.fit_transform(dataset.loc[:, dataset.columns != ds_target_col])
    else: 
        dataset_data = dataset[:, dataset.columns != ds_target_col]
    dataset_labels = dataset[[ds_target_col]].to_numpy().ravel()
    
    qcprint(LOG_LEVEL_INFO, f"Final dataset length: {len(dataset)}")
    
    # Split into train-test and validate subsets
    traintest_data, validation_data, traintest_labels, validation_labels = \
        train_test_split(dataset_data, dataset_labels, test_size=DS_VALIDATE_SIZE, random_state=42)
    
    # Split into train and test subsets
    train_data, test_data, train_labels, test_labels = \
        train_test_split(traintest_data, traintest_labels, test_size=DS_TEST_SIZE, random_state=42)
    
    # Pre-process data
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    validation_data = scaler.transform(validation_data)
    
    return train_data, test_data, train_labels, test_labels, validation_data, validation_labels