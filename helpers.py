# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
# Import Libraries

import os
import datetime
import config as cf
from config import LOG_LEVEL, LOG_FILE
from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from config import LOG_MAX_LENGTH, LOG_KEEP_STARTEND

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

import numpy as np

# HELPER METHODS
def shorten_str(message):
    message = str(message) # force String input
    
    if len(message) <= LOG_MAX_LENGTH:
        return message  # No need to shorten if it's already short enough
    
    # Ensure we don't keep more characters than the length of the string
    if LOG_KEEP_STARTEND + LOG_KEEP_STARTEND >= len(message):
        return message
    
    # Shorten the string by keeping the start and end, and replacing the middle with ' ... '
    return message[:LOG_KEEP_STARTEND] + ' ... ' + message[-LOG_KEEP_STARTEND:]    

def qcprint(log_level, message):
    if log_level >= LOG_LEVEL:
        with open(LOG_FILE, 'a' if os.path.exists(LOG_FILE) else 'w') as f:
            now = datetime.datetime.now()
    
            prnt_line = "" 
            prnt_msg = shorten_str(str(message))
            if prnt_msg.startswith('\n'):
                prnt_line += '\n'
                prnt_msg = prnt_msg[1:]  # Everything after the first newline
            
            prnt_line += now.strftime("%Y-%m-%d %H:%M:%S") + " " + prnt_msg
            
            print(prnt_line, file=f)

# Get State Vector for input data
def get_input_StateVector(featuremap, inputval, in_qubits):
  myq_reg = QuantumRegister(4, "q")
  myCircuit = QuantumCircuit(myq_reg)

  myCircuit.append(featuremap, [x for x in range(in_qubits)]) 
  myparameters = [inputval[i] for i in range(in_qubits)]
  myCircuit = myCircuit.assign_parameters(myparameters)

  myCircuit.remove_final_measurements()
  state_vector = Statevector(myCircuit)
  
  return state_vector

def get_output_StateVector(output, num_qubits, *measured_qubits):
    state_vectors = []
    
    if len(*measured_qubits) == num_qubits:  #complete measure
        for i in range(2**num_qubits):
            # Create a vector with all zeros and set the i-th position to 1
            vector = np.zeros(2**num_qubits)
            if output==1 and i%2 == 1:
                vector[i] = 1
            # Append the corresponding Statevector to the list
            state_vectors.append(Statevector(vector))
    elif len(*measured_qubits) == 1:  #partial measurement in one single qubit
        for i in range(2**num_qubits):
            # Create a vector with all zeros and set the i-th position to 1
            vector = np.zeros(2**num_qubits)
            measure_on = measured_qubits[0][0]
            if output==1 and i == 2**measure_on:
                vector[i] = 1
            # Append the corresponding Statevector to the list
            state_vectors.append(Statevector(vector))
    else:  # partial measure on more than 1 qubit, not supported
        qcprint(cf.LOG_LEVEL_ERROR, "Partial measure on more than one qubti, not yet supported, exiting")
        raise ValueError("Partial measure on more than one qubti, not yet supported")
  
    return state_vectors


mlops_params = {
    # Common params
    "dataset": cf.DS_NAME,
    "dataset length": cf.DS_LENGTH,
    "dataset undersampling": cf.DS_BALANCE_UNDERSAMPLE,
    "dataset oversampling": cf.DS_BALANCE_OVERSAMPLE,
    "Execution Threads": cf.THREADS,
    
    # Evolutionary params
    "Ansatz": cf.EVOVAQ_ANSATZ,
    "Ansatz entanglement": cf.EVOVAQ_ANSATZ_TWOLOCAL_ENTANGLEMENT,
    "RealHW Quiskit": cf.QISKIT_REAL,
    
    # Adiabatic params
    "RealHW DWave": cf.ADIABATIC_REAL,
    "Partitions": cf.ADIABATIC_TRAIN_ANG_PARTS,
    "Range": str(cf.ADIABATIC_TRAIN_LOWER_BOUND)+","+str(cf.ADIABATIC_TRAIN_UPPER_BOUND),
    "Angles": cf.ROTATION_ANGLES_COUNT,
    "Init Angles": cf.ROTATION_ANGLES_INIT,
    "Accuracy Tolerance": cf.ADIABATIC_TRAIN_ITER_TOLERANCE,
    "Measured qubits": cf.ADIABATIC_MEASURE_Q,
    "Training levels": cf.ADIABATIC_TRAIN_DEEP_LEVELS,
    "Training RealHW DWave": cf.ADIABATIC_REAL,
    "Remove Imaginary phase": cf.OPERATOR_REMOVE_IPHASE,
    
    # Classical training params
    "Training Iterations": str(cf.CLASSICAL_TRAIN_MAX_ITER),
    "Optimizer": cf.CLASSICAL_OPTIMIZER,
    "Optimizer Callback": cf.CLASSICAL_OPTIMIZER_LOGCALLBACK,

    # ANN params
    "Hidden neurons": cf.ANN_TRAIN_HIDDEN_SIZE,
    "Learning rate": cf.ANN_TRAIN_LEARNING_RATE,
    "Epochs": cf.ANN_TRAIN_EPOCHS
    }

class TrainingResults():
    def __init__(self, trainingTime=0, accuracy=0, f1=0, 
                    truePositive=0, falsePositive=0,
                    trueNegative=0, falseNegative=0,
                    angleValues=0,
                    totalIterations=0):
        
        self.trainingTime = trainingTime
        self.truePositive = truePositive
        self.falsePositive = falsePositive
        self.trueNegative = trueNegative
        self.falseNegative = falseNegative
        self.angleValues = angleValues
        self.totalIterations = totalIterations
        
        self.accuracy = (self.truePositive + self.trueNegative) / (self.truePositive + self.trueNegative + self.falsePositive + self.falseNegative) if (self.truePositive + self.trueNegative + self.falsePositive + self.falseNegative) != 0 else 0
        self.precision = self.truePositive / (self.truePositive + self.falsePositive) if (self.truePositive + self.falsePositive) != 0 else 0
        self.recall = self.truePositive / (self.truePositive + self.falseNegative) if (self.truePositive + self.falseNegative) != 0 else 0
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) != 0 else 0
        
    def metricsDictFL(self):
        #All values must be float
        return {'Accuracy': self.accuracy, 
                'Precision': self.precision,
                'Recall': self.recall,
                'F1': self.f1, 
                'Training time': self.trainingTime,
                'Training iterations': self.totalIterations,
                'TN': self.trueNegative,
                'FP': self.falsePositive,
                'FN': self.falseNegative,
                'TP': self.truePositive}
    
    def confusionMatrix(self):
        conf_matrix_str = (
            f"Confusion Matrix:\n"
            f"               Predicted 0    Predicted 1\n"
            f"Actual 0       {self.trueNegative:>12} {self.falsePositive:>12}\n"
            f"Actual 1       {self.falseNegative:>12} {self.truePositive:>12}"
        )
        
        return conf_matrix_str
    