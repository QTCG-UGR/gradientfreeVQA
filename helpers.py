# -*- coding: utf-8 -*-
"""
Created on Sun Jul 02 14:22:45 2023
 
@author: eacosta
"""
# Import Libraries

import os
import datetime
from config import LOG_LEVEL, LOG_FILE
from config import LOG_LEVEL_INFO, LOG_LEVEL_DEBUG
from config import LOG_MAX_LENGTH, LOG_KEEP_STARTEND

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

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

def get_output_StateVector(output):
  # Create a QuantumCircuit with 2 qubits and 2 classical bits
  myq_reg = QuantumRegister(4, "q")
  myCircuit = QuantumCircuit(myq_reg)

  # Add quantum gates to the circuit
  if (output==1):
      myCircuit.x(0)   # TODO: Temporarily set all to 1.  Originally, only last qubit (3)
      myCircuit.x(1)   # TODO: Review this
      myCircuit.x(2)   # TODO: Review this
      myCircuit.x(3)   # TODO: Review this

  myCircuit.remove_final_measurements()
  state_vector = Statevector(myCircuit)
  
  return state_vector