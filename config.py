#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:38:28 2024

@author: ernesto.acosta
"""
import sympy as sp
import time 
import os

LOG_LEVEL_INFO = 1
LOG_LEVEL_DEBUG = 0

LOG_LEVEL = LOG_LEVEL_INFO
LOG_MAX_LENGTH = 300
LOG_KEEP_STARTEND = 140
PRINTCIRCUIT=False

RUN_UNIT_TESTS = False
UNIT_TST_VERBOSE = False

EVOVAQ_TRAIN=True

ADIABATIC_TRAIN=False
ADIABATIC_TRAIN_LOWER_BOUND=-2*sp.pi
ADIABATIC_TRAIN_UPPER_BOUND=2*sp.pi
ADIABATIC_TRAIN_ITER_ACCURACY=True # Validates accuracy per iteration, stops whenever it drops below TOLERANCE
ADIABATIC_TRAIN_ITER_TOLERANCE=0.2 # 20% tolerance
ADIABATIC_TRAIN_DEEP_LEVELS = 10

OPERATOR_FILE_LOAD = True
COSTFUNC_FILE_LOAD = False

EVOVAQ_ANSATZ_REALAMPLITUDES = 'RealAmplitudes'
EVOVAQ_ANSATZ_TWOLOCAL = 'TwoLocal'
EVOVAQ_ANSATZ=EVOVAQ_ANSATZ_TWOLOCAL #original: EVOVAQ_ANSATZ_REALAMPLITUDES
EVOVAQ_ANSATZ_TWOLOCAL_ENTANGLEMENT = [[0, 1], [1, 2], [2, 3], [3, 0]]

ADIABATIC_TRAIN_DSLENGTH = 100
ROTATION_ANGLES_COUNT = 4
DISCRETE_EXT_PARTS = 2

COSTFUNC_THREADS = os.cpu_count() #8 # We can assign max threads available by using os.cpu_count()

SHOTS = 1024
QISKIT_REAL = False
QISKIT_HUB = 'ibm-q'
QISKIT_GROUP = 'open'
QISKIT_PROJECT = 'main'
QISKIT_TOKEN = 'toktok'  # IBM Quantum token
QISKIT_BACKEND = 'ibm_brisbane' #Backend to use

ADIABATIC_REAL = False
DWAVE_TOKEN = 'toktok'  # DWave token
DWAVE_TIMELIMIT = 60

LOG_DIR="./log/"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = LOG_DIR + "log{}.log".format(time.time())

OPERATOR_FILE_DIR = './Operator/'
if not os.path.exists(OPERATOR_FILE_DIR):
    os.makedirs(OPERATOR_FILE_DIR)
OPERATOR_FILE_NAME = OPERATOR_FILE_DIR + "operator"+EVOVAQ_ANSATZ+".pkl"
OPERATOR_FILE_NAME_TXT = OPERATOR_FILE_DIR + "operator"+EVOVAQ_ANSATZ+".txt"
OPERATOR_FILE_NAME_PLAINTXT = OPERATOR_FILE_DIR + "operator"+EVOVAQ_ANSATZ+"_plain.txt"

COSTFUNC_FILE_DIR = './Cost_Function/'
if not os.path.exists(COSTFUNC_FILE_DIR):
    os.makedirs(COSTFUNC_FILE_DIR)
COSTFUNC_FILE_NAME = COSTFUNC_FILE_DIR + 'cost_function.pkl'
COSTFUNC_FILE_NAME_TXT = COSTFUNC_FILE_DIR + 'cost_function.txt'    
