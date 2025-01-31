#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:19:55 2024

@author: ernesto.acosta
"""

# Import Libraries
import numpy as np
import sympy as sp
from qiskit_algorithms.optimizers import SPSA, COBYLA, ADAM
import matplotlib.pyplot as plt
import random
import time

from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile
from qiskit_ibm_provider import IBMProvider

from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR, PRINT_COST
from config import CLASSICAL_TRAIN_MAX_ITER, LOG_DIR, SHOTS 
from config import CLASSICAL_OPTIMIZER, CLASSICAL_OPTIMIZER_SPSA, CLASSICAL_OPTIMIZER_COBYLA, CLASSICAL_OPTIMIZER_ADAM, CLASSICAL_OPTIMIZER_LOGCALLBACK
from config import ROTATION_ANGLES_INIT, ROTATION_ANGLES_INIT_RANDOM
from config import ADIABATIC_TRAIN_LOWER_BOUND, ADIABATIC_TRAIN_UPPER_BOUND
from config import QISKIT_HUB, QISKIT_GROUP, QISKIT_PROJECT, QISKIT_BACKEND, QISKIT_TOKEN
from helpers import qcprint, TrainingResults


# QRNN CLASSICAL TRAINING
class QNN_train_classic():
    def __init__(self, data, labels, qnn, rot_angles_count, shots):
        self.data = data
        self.labels = labels
        self.qnn = qnn
        self.shots = shots
        
        self.rot_angles_count = rot_angles_count
        self.symbolic_vars = sp.symbols(' '.join([chr(97 + i) for i in range(self.rot_angles_count)]))
    
    def cost_function(self, var_parameters):
        def get_parity(classif):
            parity_1 = 0 
            for state, count in classification.items():
                if state.count('1') % 2 == 1:  
                    parity_1 += count  
                    
            return parity_1 / sum(classification.values()) 
        
        classifications = circuits_execution(self.data, var_parameters, self.qnn, self.shots)

        cost = 0
        for i, classification in enumerate(classifications):
            p = get_parity(classification)
            if self.labels[i]!=1:
                p = 1 - p

            #p = classification.get(self.labels[i])
            cost += -np.log(p + 1e-10)
        cost /= len(self.data)

        return cost
    
    def train(self):
        classic_training_start = time.time()
        
        range_start = ADIABATIC_TRAIN_LOWER_BOUND 
        range_end = ADIABATIC_TRAIN_UPPER_BOUND
        
        qcprint(LOG_LEVEL_INFO, f"\nTRAINING QNN CLASSICAL approach with {CLASSICAL_OPTIMIZER} Optimizer, for: {CLASSICAL_TRAIN_MAX_ITER} iterations")

        init_ansatz_params = {str(var):random.uniform(range_start, range_end) for var in self.symbolic_vars}
        if (ROTATION_ANGLES_INIT != ROTATION_ANGLES_INIT_RANDOM):
            init_ansatz_params = ROTATION_ANGLES_INIT

        init_ansatz_params = [float(value.evalf()) if hasattr(value, 'evalf') else float(value) for value in init_ansatz_params.values()]

        log = OptimizerLog()
        if CLASSICAL_OPTIMIZER == CLASSICAL_OPTIMIZER_SPSA:  #https://qiskit-community.github.io/qiskit-algorithms/apidocs/qiskit_algorithms.optimizers.html
            if CLASSICAL_OPTIMIZER_LOGCALLBACK:
                spsa = SPSA(maxiter=CLASSICAL_TRAIN_MAX_ITER, callback=log.update)
            else:
                spsa = SPSA(maxiter=CLASSICAL_TRAIN_MAX_ITER)
            result = spsa.minimize(fun=self.cost_function, x0=list(init_ansatz_params), 
                                   bounds=[(-1, 1)] * len(init_ansatz_params) )
            
        elif CLASSICAL_OPTIMIZER == CLASSICAL_OPTIMIZER_COBYLA:
            if CLASSICAL_OPTIMIZER_LOGCALLBACK: qcprint(LOG_LEVEL_ERROR, "LogCallback is not supported by Cobyla")
            
            cobyla = COBYLA(maxiter=CLASSICAL_TRAIN_MAX_ITER)
            result = cobyla.minimize(fun=self.cost_function, x0=list(init_ansatz_params))
            
        elif CLASSICAL_OPTIMIZER == CLASSICAL_OPTIMIZER_ADAM:
            if CLASSICAL_OPTIMIZER_LOGCALLBACK: qcprint(LOG_LEVEL_ERROR, "LogCallback is not supported by Adam")
            
            adam = ADAM(maxiter=CLASSICAL_TRAIN_MAX_ITER)
            result = adam.minimize(fun=self.cost_function, x0=list(init_ansatz_params), 
                       bounds=[(-1, 1)] * len(init_ansatz_params))
        else:
            qcprint(LOG_LEVEL_ERROR, f"Invalid Optimizer specified: {CLASSICAL_OPTIMIZER}")
            raise ValueError(f"Invalid Optimizer specified: {CLASSICAL_OPTIMIZER}")
        
        opt_var = result.x

        # Plot Training Results
        if PRINT_COST:
            fig = plt.figure()
            plt.plot(log.evaluations, log.costs)
            plt.xlabel('Steps')
            plt.ylabel('Cost')
            fig = plt.gcf()
            fig.set_size_inches(10, 10)
            plt.title('QNN Classical Training and Validation Loss')
            plt.savefig(LOG_DIR+"qrnn_cost_{}.png".format(time.time()))
            plt.show()
        
        classic_training_end = time.time()
        
        return TrainingResults(trainingTime=classic_training_end-classic_training_start, angleValues=opt_var, totalIterations=CLASSICAL_TRAIN_MAX_ITER)
    
# Optimizer Log Class
class OptimizerLog:
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
        
    def update(self, evaluation, parameter, cost, _stepsize, _accept):
        qcprint(LOG_LEVEL_INFO, "evaluation: " + str(evaluation) + ", cost: " + str(cost))
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)

def label_execution(results, num_qubits):
    shots = sum(results.values())
    #probabilities = {0: 0, 1: 0}
    probabilities = {f"{state:0{num_qubits}b}": 0 for state in range(2**num_qubits)}

    for bitstring, counts in results.items():
     probabilities[bitstring] += counts / shots

    return probabilities

# Circuit run on Q Gate Computer or Simulator
def circuits_execution(data, ansatz_params, qrnn, shots, realhw=False):
    circuits = []

    for d in data:
      bound_circuit = qrnn.assign_parameters(np.concatenate((d,ansatz_params)))
      circuits.append(bound_circuit)
    
    circ_ind = 0
    for c in circuits:
        c.name = "QRNN-" + str(circ_ind)
        circ_ind += 1
    
    if not realhw: # run on simulator
        backend = BasicProvider().get_backend("basic_simulator")
        
        new_circuits = transpile(circuits, backend)
        results = backend.run(new_circuits, shots = SHOTS).result()
    else: # run on real hardware
        IBMProvider.save_account(token=QISKIT_TOKEN, overwrite=True)
        provider = IBMProvider()
        qiskit_instance = str(QISKIT_HUB) + "/" + str(QISKIT_GROUP) + "/" + str(QISKIT_PROJECT)
        provider = IBMProvider(instance=qiskit_instance)
        
        backend = provider.get_backend(QISKIT_BACKEND)
        transpiled = transpile(circuits, backend=backend)
        job = backend.run(transpiled)

        results = job.result()
    
    classification = [label_execution(results.get_counts(c), c.num_qubits) for c in circuits]
    
    return classification