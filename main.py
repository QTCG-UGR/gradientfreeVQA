#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:21:20 2024

@author: ernesto.acosta

Run from terminal by using: > nohup caffeinate python3 main.py > output.log 2>&1 &
To run mlflow server run: > mlflow ui
To search for active processes funning, execute: > ps aux | grep main.py
To kill any specific process execute: > kill {process_id}

pip install qiskit_symb
pip install sympy dimod
pip install dwave-preprocessing

"""
import numpy as np
import pickle
from qiskit_symb.quantum_info import Operator
import sympy as sp
from sympy import I
import torch
import time
from datetime import datetime
import mlflow
import json

import config as cf

from datamgt import load_data
from evovaq_impl import EVOVAQ
from qnn_train_adiabatic import QNN_train_adiabatic
from qnn_train_classical import QNN_train_classic
from ann import ANN

from helpers import get_input_StateVector, get_output_StateVector
from helpers import qcprint
from helpers import mlops_params, TrainingResults
from operatorprocessor import OperatorPreprocessor

from pprint import pprint
import traceback

M_EVOLUTIONARY = "VQA EVOLUTIONARY"
M_ADIABATIC = "VQA ADIABATIC"
M_CLASSIC = "VQA CLASSIC"
M_ANN = "ANN"

def buildCircuit(modelName, train_data, test_data, train_labels, test_labels, adiabatic=False):
    vqa =  EVOVAQ(train_data, test_data, train_labels, test_labels)
    
    if (modelName == M_ADIABATIC):  # Handle partial measurement, not supported by EVOVAQ
        if (cf.ADIABATIC_MEASURE_Q != cf.ADIABATIC_MEASURE_Q_ALL):
            if (cf.ADIABATIC_MEASURE_Q < vqa.circuit.num_qubits):
                
                new_data = []
                for instruction, qubits, clbits in vqa.circuit.data:
                    if instruction.name != "measure":  # Keep all gates except "measure"
                        new_data.append((instruction, qubits, clbits))
                vqa.circuit.data = new_data
                
                #Add new partial measurement
                vqa.circuit.measure([cf.ADIABATIC_MEASURE_Q], [cf.ADIABATIC_MEASURE_Q])  # Add new measurements
            else:
                qcprint(cf.LOG_LEVEL_ERROR, "ADIABATIC_MEASURE_Q={cf.ADIABATIC_MEASURE_Q} is set on an invalid qubit, complete measured will be used instead")

    if (modelName == M_ANN):
        ann = ANN(vqa.circuit.num_qubits, cf.ANN_TRAIN_HIDDEN_SIZE)
        return ann

    if (cf.PRINTCIRCUIT):  vqa.print_circuit()
    
    return vqa

def evolutionaryTraining(evol_circuit, train_data, train_labels, test_data, test_labels):
    qcprint(cf.LOG_LEVEL_INFO, "\nEVOLUTIONARY TRAINING")    
    
    resultTrain = TrainingResults()
    resultTrain = evol_circuit.train(False)
    bestAngles = resultTrain.angleValues
    
    resultTest = evol_circuit.test(bestAngles, cf.QISKIT_REAL)
    resultTest.trainingTime = resultTrain.trainingTime
    
    qcprint(cf.LOG_LEVEL_INFO, f"Evolutionary Accuracy: {resultTest.accuracy}. Optimal angles: {resultTrain.angleValues}")
    qcprint(cf.LOG_LEVEL_INFO, f"Evolutionary F1-Score: {resultTest.f1}")
    qcprint(cf.LOG_LEVEL_INFO, f"Evolutionary training took: {resultTest.trainingTime} seconds")
    qcprint(cf.LOG_LEVEL_INFO, f"Performance: {resultTest.metricsDictFL()}")
    qcprint(cf.LOG_LEVEL_INFO, f"ConfusionMatrix: {resultTest.confusionMatrix()}")
    print(f"Training Evolutionary Time: {resultTest.trainingTime}s. Optimal parameters: {resultTrain.angleValues}")
    pprint(resultTest.metricsDictFL())
    
    return resultTest, evol_circuit

def adiabaticTraining(adiab_circuit, train_data, train_labels, test_data, test_labels):
    qcprint(cf.LOG_LEVEL_INFO, "\nADIABATIC TRAINING")
    
    qcprint(cf.LOG_LEVEL_INFO, f"Dataset Lenght: {cf.DS_LENGTH}")
    qcprint(cf.LOG_LEVEL_INFO, f"Number of partitions: {cf.ADIABATIC_TRAIN_ANG_PARTS}")
    qcprint(cf.LOG_LEVEL_INFO, f"Deep level search: {cf.ADIABATIC_TRAIN_DEEP_LEVELS}")
    qcprint(cf.LOG_LEVEL_INFO, f"Accuracy Tolerance per iteration: {cf.ADIABATIC_TRAIN_ITER_TOLERANCE}")
    qcprint(cf.LOG_LEVEL_INFO, f"Parallel processing threads: {cf.THREADS}")
    qcprint(cf.LOG_LEVEL_INFO, f"Measure on qubit: {cf.ADIABATIC_MEASURE_Q}")
    
    # 4.1 prepare Cost Function
    op_Preprocessor = OperatorPreprocessor(cf.ROTATION_ANGLES_COUNT, cf.ADIABATIC_TRAIN_ANG_PARTS)
    q_operator = None
    if (cf.OPERATOR_FILE_LOAD):  # Load Quantum Operator from file
        with open(cf.OPERATOR_FILE_NAME, 'rb') as f:
            q_operator = pickle.load(f)
            
        qcprint(cf.LOG_LEVEL_INFO, "\nAnsatz Operator matrix loaded from file.")
        
    else:  # Build Quantum Operator out from the Circuit
        operator_start = time.time()
        q_operator = Operator(adiab_circuit.ansatz).to_sympy()
        if (cf.OPERATOR_REMOVE_IPHASE):
            q_operator = q_operator.subs(sp.I, 1) # remove Imaginary part on terms
        q_operator = q_operator.subs(op_Preprocessor.theta_replacements) # Replace thetas[i] with a, b, c
        q_operator = op_Preprocessor.matrix2hiperbolic(q_operator) # Convert exps to hiperbolic representation
        q_operator = op_Preprocessor.matrix2expsums(q_operator) # Convert hiperbolic prods into exp sums
        q_operator = q_operator.applyfunc(lambda x: x.rewrite(sp.exp)) # rewrite exp into sp.exp
        q_operator = q_operator.applyfunc(lambda x: x.subs(I, sp.I)) # rewrite I into sp.I
        operator_end = time.time()
        
        qcprint(cf.LOG_LEVEL_DEBUG, f"\nOperator Matrix built.  Took: {operator_end-operator_start} sec")
        #print(f"\nOperator Matrix built.  Took: {operator_end-operator_start} sec")
        
        with open(cf.OPERATOR_FILE_NAME, 'wb') as f: # File persist, binary format
            pickle.dump(q_operator, f)
        with open(cf.OPERATOR_FILE_NAME_TXT, 'w') as f: # File persist, pickle format
            f.write(sp.pretty(q_operator))
        with open(cf.OPERATOR_FILE_NAME_PLAINTXT, 'w') as f: # File persist, plain text
            f.write(str(q_operator))   
        qcprint(cf.LOG_LEVEL_INFO, "Operator matrix saved to file.")
    
    # Training data
    ad_train_data = train_data[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(train_data)]
    ad_train_labels = train_labels[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(train_labels)]
    
    # Testing data
    ad_test_data = test_data[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(test_data)]
    ad_test_labels = test_labels[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(test_labels)]
    
    measured_qubits = []
    for instruction, qargs, cargs in adiab_circuit.circuit.data:
        if instruction.name == 'measure':
            measured_qubits.append(qargs[0]._index)
    
    qcprint(cf.LOG_LEVEL_INFO, f"Measured Qubits: {measured_qubits}")
    
    adiabaticTrainer = QNN_train_adiabatic(cf.ROTATION_ANGLES_COUNT, q_operator, cf.ADIABATIC_TRAIN_ANG_PARTS, 
                                     [np.real(get_input_StateVector(adiab_circuit.feature_map, ad_train_data[i], adiab_circuit.circuit.num_qubits)) for i in range(len(ad_train_data))],
                                     [np.real(get_output_StateVector(j, adiab_circuit.circuit.num_qubits, measured_qubits)) for j in ad_train_labels], 
                                     [np.real(get_input_StateVector(adiab_circuit.feature_map, ad_test_data[i], adiab_circuit.circuit.num_qubits)) for i in range(len(ad_test_data))],
                                     [np.real(get_output_StateVector(j, adiab_circuit.circuit.num_qubits, measured_qubits)) for j in ad_test_labels], 
                                      adiab_circuit, adiab_circuit.circuit.num_qubits)

    result = adiabaticTrainer.train(cf.ADIABATIC_TRAIN_DEEP_LEVELS)
    optimal_sorted = dict(sorted(result.angleValues.items()))
    #optimal_sorted = {key: value.evalf() for key, value in optimal_sorted.items()}
    optimal_sorted = {key: value for key, value in optimal_sorted.items()}
    
    # Validate    
    qcprint(cf.LOG_LEVEL_INFO, f"Adiabatic accuracy: {result.accuracy}. Optimal angles: {optimal_sorted}")
    qcprint(cf.LOG_LEVEL_INFO, f"Adiabatic F1-Score: {result.f1}. Optimal angles: {optimal_sorted}")
    qcprint(cf.LOG_LEVEL_INFO, f"Performance: {result.metricsDictFL()}")
    qcprint(cf.LOG_LEVEL_INFO, f"ConfusionMatrix: {result.confusionMatrix()}")
    print(f"Training VQA Adiabatic. Time: {result.trainingTime}s. Optimal parameters: {optimal_sorted}. Performance: ")
    pprint(result.metricsDictFL())
    
    return result, adiab_circuit


def classicalTraining(classic_circuit, train_data, train_labels, test_data, test_labels):
    qcprint(cf.LOG_LEVEL_INFO, "\nCLASSICAL TRAINING")
    
    cl_train_data = train_data[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(train_data)]
    cl_train_labels = train_labels[0:cf.DS_LENGTH if cf.DS_LENGTH != cf.DS_LENGTH_ALL else len(train_labels)]
    
    classicTrainer = QNN_train_classic(cl_train_data, cl_train_labels, classic_circuit.circuit, cf.ROTATION_ANGLES_COUNT, cf.SHOTS)
    
    resultTrain = classicTrainer.train()
    
    # Test
    resultTest = classic_circuit.test(resultTrain.angleValues, test_data, test_labels, cf.QISKIT_REAL)
    resultTest.trainingTime = resultTrain.trainingTime
    resultTest.totalIterations = resultTrain.totalIterations
    
    qcprint(cf.LOG_LEVEL_INFO, f"Classical Accuracy: {resultTest.accuracy}. Optimal angles: {resultTrain.angleValues}")
    qcprint(cf.LOG_LEVEL_INFO, f"Classical F1-Score: {resultTest.f1}")
    qcprint(cf.LOG_LEVEL_INFO, f"Classical training took: {resultTest.trainingTime} seconds")
    qcprint(cf.LOG_LEVEL_INFO, f"Performance: {resultTest.metricsDictFL()}")
    qcprint(cf.LOG_LEVEL_INFO, f"ConfusionMatrix: {resultTest.confusionMatrix()}")
    print(f"Training Classical Time:{resultTest.trainingTime}s. Optimal parameters: {resultTrain.angleValues}")
    
    pprint(resultTest.metricsDictFL())
    
    return resultTest, classic_circuit

def annTraining(ann, train_data, train_labels, test_data, test_labels):
    qcprint(cf.LOG_LEVEL_INFO, "\nANN TRAINING")
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(train_data, dtype=torch.float32)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)
    
    results = ann.trainme(X_train, y_train, X_test, y_test, cf.ANN_TRAIN_LEARNING_RATE, cf.ANN_TRAIN_EPOCHS)
    
    qcprint(cf.LOG_LEVEL_INFO, f"ANN Accuracy: {results.accuracy}.")
    qcprint(cf.LOG_LEVEL_INFO, f"ANN training took: {results.trainingTime} seconds")
    qcprint(cf.LOG_LEVEL_INFO, f"Performance: {results.metricsDictFL()}")
    qcprint(cf.LOG_LEVEL_INFO, f"ConfusionMatrix: {results.confusionMatrix()}")
    
    print(f"Training ANN. Time:{results.trainingTime}s. Optimal parameters: {results.angleValues}")
    pprint(results.metricsDictFL())
    
    return results, ann

def validate(modelName, circuit, bestAngles, validation_data, validation_labels):
    qcprint(cf.LOG_LEVEL_INFO, f"{modelName} - Validation starts")
    
    if (type(circuit) == ANN): 
        validation_data = torch.tensor(validation_data, dtype=torch.float32)
        validation_labels = torch.tensor(validation_labels, dtype=torch.float32).view(-1, 1)
        bestAngles_str = cf.ANN_PARAMS_INIT 
        
        if cf.ANN_PARAMS_INIT == cf.ROTATION_ANGLES_INIT_RANDOM:
            bestAngles = [torch.randn_like(param) for param in circuit.parameters()]
        else:
            bestAngles = json.loads(bestAngles_str)

        with torch.no_grad():
            for param, new_value in zip(circuit.parameters(), bestAngles):
                if isinstance(new_value, list):  new_value = torch.tensor(new_value)
                
                qcprint(cf.LOG_LEVEL_DEBUG,  f"Assigning ANN parameter.  Shape: {param.shape}, New value shape: {new_value.shape}")
                if param.shape == new_value.shape:
                    param.copy_(new_value if cf.ANN_PARAMS_INIT == cf.ROTATION_ANGLES_INIT_RANDOM else new_value.clone().detach()) 
                else:
                    raise ValueError(f"Shape mismatch: {param.shape} != {new_value.shape}")

    
    resultValidation = circuit.test(bestAngles, validation_data, validation_labels, cf.QISKIT_REAL)
    
    qcprint(cf.LOG_LEVEL_INFO, f" Validation Accuracy: {resultValidation.accuracy}.")
    qcprint(cf.LOG_LEVEL_INFO, f" Validation took: {resultValidation.trainingTime} seconds")
    qcprint(cf.LOG_LEVEL_INFO, f" Validation Performance: {resultValidation.metricsDictFL()}")
    qcprint(cf.LOG_LEVEL_INFO, f" Validation ConfusionMatrix: {resultValidation.confusionMatrix()}")
    
    print(f"Validation {modelName}. Time:{resultValidation.trainingTime}s. Optimal parameters: {resultValidation.angleValues}")
    pprint(resultValidation.metricsDictFL())
    
    return resultValidation

def main():
    qcprint(cf.LOG_LEVEL_INFO, "GRADIENT FREE VQC MODELS COMPARISON")
    
    # 1. Data Load.
    train_data, test_data, train_labels, test_labels, validation_data, validation_labels = load_data(cf.DS_NAME)
    
    for i in range(cf.BATCH_EXECUTIONS):
        try:
            qcprint(cf.LOG_LEVEL_INFO, f"\nExperiment {i+1} of {cf.BATCH_EXECUTIONS}")
            
            # Define a dictionary with model names, their corresponding training functions, and configuration flags
            training_methods = {
                M_EVOLUTIONARY: { "function": evolutionaryTraining, "config_flag": "MODEL_EVOL" },
                M_ADIABATIC: { "function": adiabaticTraining, "config_flag": "MODEL_ADIABATIC" },
                M_CLASSIC: { "function": classicalTraining, "config_flag": "MODEL_CLASSICAL" },
                M_ANN: { "function": annTraining, "config_flag": "MODEL_ANN" }
            }
            
            # Iterate over the training methods
            for model_name, training_info in training_methods.items():
                config_flag = training_info["config_flag"]
                if getattr(cf, config_flag, False):
                    
                    trainResults = TrainingResults()
                    testResults = TrainingResults()
                    bestAngles = cf.ROTATION_ANGLES_INIT
                    circuit = buildCircuit(model_name, train_data, train_labels, test_data, test_labels)
                    
                    if not cf.ML_OPS:
                        if cf.TRAIN:
                            qcprint(cf.LOG_LEVEL_INFO, "Training Starts")
                            trainResults, circuit = training_info["function"](circuit, train_data, train_labels, test_data, test_labels)  # Call the training function directly
                            bestAngles = trainResults.angleValues
                        if cf.VALIDATE:
                            qcprint(cf.LOG_LEVEL_INFO, "Validation Starts")
                            testResults = validate(model_name, circuit, bestAngles, validation_data, validation_labels)
                            
                    else:
                        mlflow.set_experiment(model_name)
                        mlflow.set_tracking_uri(cf.ML_OPS_URI)
                        
                        if cf.TRAIN:
                            qcprint(cf.LOG_LEVEL_INFO, "Training Starts")
                            with mlflow.start_run(run_name=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')):
                                mlops_params["dataset length"] = len(train_data) + len(test_data)
                                mlflow.log_params(mlops_params)
                                mlflow.log_param("Model", model_name)
                                mlflow.log_param("Phase", "training")

                                trainResults, circuit = training_info["function"](circuit, train_data, train_labels, test_data, test_labels)  # Call the training function
                                bestAngles = trainResults.angleValues
                                
                                mlflow.log_metrics(trainResults.metricsDictFL())
                                mlflow.log_param("Result Angles", trainResults.angleValues)  # Cannot save as metrics as value is not Float
                                
                        if cf.VALIDATE:
                            qcprint(cf.LOG_LEVEL_INFO, "Validation Starts")
                            with mlflow.start_run(run_name=datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')):
                                mlops_params["dataset length"] = len(train_data) + len(test_data)
                                mlflow.log_params(mlops_params)
                                mlflow.log_param("Model", model_name)
                                mlflow.log_param("Phase", "validation")
                                
                                testResults = validate(model_name, circuit, bestAngles, validation_data, validation_labels)
                                
                                mlflow.log_metrics(testResults.metricsDictFL())
                                
        except Exception as e:
            print(f"Exception caught in main loop: {e}")
            error_details = traceback.format_exc()
            print(error_details)
            qcprint(cf.LOG_LEVEL_ERROR, f"Exception caught in main loop: {e}")

if __name__ == "__main__":
    main()
    