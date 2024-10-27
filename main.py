#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:21:20 2024

@author: ernesto.acosta

pip install qiskit_symb
pip install sympy dimod

"""
import numpy as np
import pickle
from qiskit_symb.quantum_info import Operator
import sympy as sp

from config import EVOVAQ_TRAIN, PRINTCIRCUIT
from config import ADIABATIC_TRAIN
from config import OPERATOR_FILE_LOAD, OPERATOR_FILE_NAME, OPERATOR_FILE_NAME_TXT, OPERATOR_FILE_NAME_PLAINTXT
from config import ADIABATIC_TRAIN_DSLENGTH, ADIABATIC_TRAIN_DEEP_LEVELS
from config import ADIABATIC_TRAIN_ITER_TOLERANCE, COSTFUNC_THREADS
from config import DISCRETE_EXT_PARTS
from config import ROTATION_ANGLES_COUNT
from config import LOG_LEVEL_INFO 

from datamgt import load_data
from evovaq_impl import EVOVAQ
from qnn_train_adiabatic import QNN_train_adiabatic
from helpers import get_input_StateVector, get_output_StateVector
from helpers import qcprint
from operatorprocessor import OperatorPreprocessor

qcprint(LOG_LEVEL_INFO, "GRADIENT FREE VQC TRAINING COMPARISON")

# 1. Data Load.  EVOVAQ implementation loads first 100 records from dataset considering all 4 categories but only 2 classes
train_data, test_data, train_labels, test_labels = load_data("iris")

# 2. Build Variational Circuit
evovaq =  EVOVAQ(train_data, test_data, train_labels, test_labels)
if (PRINTCIRCUIT):
    evovaq.print_circuit()
   
# 3. Evolutionary Training
if (EVOVAQ_TRAIN):
    qcprint(LOG_LEVEL_INFO, "\nEVOLUTIONARY TRAINING")
    evovaq_bestangles = evovaq.train()

    # Validate
    evol_Accuracy=evovaq.test(evovaq_bestangles)
    qcprint(LOG_LEVEL_INFO, f"Evolutionary Accuracy: {evol_Accuracy}. Optimal angles: {evovaq_bestangles}")

# 4. Adiabatic Training
if (ADIABATIC_TRAIN):
    qcprint(LOG_LEVEL_INFO, "\nADIABATIC TRAINING")
    qcprint(LOG_LEVEL_INFO, f"Dataset Lenght: {ADIABATIC_TRAIN_DSLENGTH}")
    qcprint(LOG_LEVEL_INFO, f"Number of partitions: {DISCRETE_EXT_PARTS}")
    qcprint(LOG_LEVEL_INFO, f"Deep level search: {ADIABATIC_TRAIN_DEEP_LEVELS}")
    qcprint(LOG_LEVEL_INFO, f"Accuracy Tolerance per iteration: {ADIABATIC_TRAIN_ITER_TOLERANCE}")
    qcprint(LOG_LEVEL_INFO, f"Pareallel processing threads: {COSTFUNC_THREADS}")
    
    # 4.1 prepare Cost Function
    op_Preprocessor = OperatorPreprocessor(ROTATION_ANGLES_COUNT, DISCRETE_EXT_PARTS)
    q_operator = None
    if (OPERATOR_FILE_LOAD):  # Load Quantum Operator from file
        with open(OPERATOR_FILE_NAME, 'rb') as f:
            q_operator = pickle.load(f)
            
        qcprint(LOG_LEVEL_INFO, "\nAnsatz Operator matrix loaded from file.")
        
    else:  # Build Quantum Operator out from the Circuit
        q_operator = Operator(evovaq.ansatz).to_sympy()
        q_operator = q_operator.applyfunc(lambda x: x.rewrite(sp.exp)) # rewrite exp into sp.exp
        q_operator = q_operator.subs(sp.I, 1) # remove Imaginary part on terms
        q_operator = q_operator.subs(op_Preprocessor.theta_replacements) # Replace thetas[i] with a, b, c
        q_operator = op_Preprocessor.matrix2hiperbolic(q_operator) # Convert exps to hiperbolic representation
        q_operator = op_Preprocessor.matrix2expsums(q_operator) # Convert hiperbolic prods into exp sums
        
        qcprint(LOG_LEVEL_INFO, "\nOperator matrix built")
        
        with open(OPERATOR_FILE_NAME, 'wb') as f: # File persist, binary format
            pickle.dump(q_operator, f)
        with open(OPERATOR_FILE_NAME_TXT, 'w') as f: # File persist, pickle format
            f.write(sp.pretty(q_operator))
        with open(OPERATOR_FILE_NAME_PLAINTXT, 'w') as f: # File persist, plain text
            f.write(str(q_operator))   
        qcprint(LOG_LEVEL_INFO, "Operator matrix saved to file.")
    
    # 4.2 Training
    train_data = train_data[0:ADIABATIC_TRAIN_DSLENGTH]
    train_labels = train_labels[0:ADIABATIC_TRAIN_DSLENGTH]
    
    adiabaticTrainer = QNN_train_adiabatic(ROTATION_ANGLES_COUNT, q_operator, DISCRETE_EXT_PARTS, 
                                     [np.real(get_input_StateVector(evovaq.feature_map, train_data[i], 4)) for i in range(len(train_data))],
                                     [np.real(get_output_StateVector(j)) for j in train_labels], 
                                      evovaq, evovaq.circuit.num_qubits)

    optimal = adiabaticTrainer.train(ADIABATIC_TRAIN_DEEP_LEVELS)
    optimal_sorted = dict(sorted(optimal.items()))
    
    # Validate
    adiab_accuracy = evovaq.test(list(optimal_sorted.values()))
    qcprint(LOG_LEVEL_INFO, f"Adiabatic accuracy: {adiab_accuracy}. Optimal angles: {optimal_sorted}")
    
    