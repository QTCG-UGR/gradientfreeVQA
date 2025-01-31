# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta

Required libraries
!pip install qiskit
etc
"""

#Import configuration
from config import LOG_DIR, SHOTS 
from config import EVOVAQ_ANSATZ, EVOVAQ_ANSATZ_REALAMPLITUDES
from config import EVOVAQ_ANSATZ_TWOLOCAL, EVOVAQ_ANSATZ_TWOLOCAL_ENTANGLEMENT
from config import QISKIT_REAL, QISKIT_HUB, QISKIT_GROUP, QISKIT_PROJECT, QISKIT_BACKEND, QISKIT_TOKEN
from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from helpers import qcprint, TrainingResults

# Import Libraries
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicProvider
from qiskit import transpile

from qiskit_ibm_provider import IBMProvider

from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op
import numpy as np
import time

# Uin Class
class EVOVAQ():

    ptimizer = None
    problem = None
    train_data = None
    test_data = None
    train_labels = None
    test_labels = None
    circuit = None
    rest = None
    feature_map = None
    ansatz = None
    
    def print_circuit(self):
        qcprint(LOG_LEVEL_INFO, "Printing circuit")
        printable = self.circuit.decompose()
        printable.draw(output='mpl', reverse_bits = True, filename=LOG_DIR+("evovaq_circuit_comprised_{}.png".format(time.time())))
        printable = printable.decompose().decompose()
        printable.draw(output='mpl', reverse_bits = True, filename=LOG_DIR+("evovaq_circuit_decomposed_{}.png".format(time.time())))

    def get_label_prediction(self, circuit, features, params, realhw=False):
        #print("---- LABEL PREDICTION -----")
        # Bind the parameters to our quantum classifier
        bound_circuit = circuit.assign_parameters(np.concatenate((features,params)))
        
        qcprint(LOG_LEVEL_DEBUG, f"Validating on {'real' if QISKIT_REAL else 'simulated'} Qiskit hardware")
        
        if not realhw:
            backend = BasicProvider().get_backend("basic_simulator")
            new_circuit = transpile(bound_circuit, backend)
            counts = backend.run(new_circuit,shots = SHOTS).result().get_counts()
        else:
            IBMProvider.save_account(token=QISKIT_TOKEN, overwrite=True)
            provider = IBMProvider()
            qiskit_instance = str(QISKIT_HUB) + "/" + str(QISKIT_GROUP) + "/" + str(QISKIT_PROJECT)
            provider = IBMProvider(instance=qiskit_instance)
            
            backend = provider.get_backend(QISKIT_BACKEND)
            transpiled = transpile(bound_circuit, backend=backend)
            job = backend.run(transpiled)
    
            results = job.result()
            counts = results.get_counts()
        
        # Read the label by considering the parity mapping of the final quantum state
        parity_1 = 0 
        for state, count in counts.items():
            if state.count('1') % 2 == 1:  
                parity_1 += count  
                
        return parity_1 / sum(counts.values())  
    
    def __init__(self, train_data, test_data, train_labels, test_labels):
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        
        
        #2. Building the Variational Quantum Classifier
        # Encode classical data in a quantum system through a FeatureMap
        dim = self.train_data.shape[1]
        self.feature_map = ZZFeatureMap(dim, reps=1, entanglement='linear')
        
        # Define an Ansatz to be trained
        if (EVOVAQ_ANSATZ == EVOVAQ_ANSATZ_REALAMPLITUDES):
            self.ansatz = RealAmplitudes(num_qubits=dim, reps=1, entanglement='circular')
        elif (EVOVAQ_ANSATZ == EVOVAQ_ANSATZ_TWOLOCAL):
            self.ansatz = TwoLocal(dim, 'x', 'crx', EVOVAQ_ANSATZ_TWOLOCAL_ENTANGLEMENT, reps=1)
            self.ansatz._bounds = [(-2 * np.pi, 2 * np.pi)] * self.ansatz.num_parameters
        else:
            self.ansatz = RealAmplitudes(num_qubits=dim, reps=1, entanglement='circular')

        # Put together our quantum classifier
        #self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit = QuantumCircuit(self.train_data.shape[1])
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.barrier()
        self.circuit.compose(self.ansatz, inplace=True)
        
        # Measure all the qubits to retrieve label information
        self.circuit.measure_all()
        """
        #mechanism to read only one qubit
        from qiskit import ClassicalRegister
        cr = ClassicalRegister(1)
        self.circuit.add_register(cr)
        #print("Classical bits:")
        #print(self.circuit.clbits)
        self.circuit.measure(3,0)
        """
        #self.printEVOVAQ_circuit(self.circuit)
        
    def train(self, realhw=False):
        training_start = time.time()
        #3. Defining the cost function to be minimized
        def cost_function(params):
            predictions = [self.get_label_prediction(self.circuit, features, params, realhw) for features in self.train_data]
            return log_loss(self.train_labels, predictions)
        
        #4. Setting up the problem
        self.problem = Problem(self.ansatz.num_parameters, self.ansatz.parameter_bounds, cost_function)
        
        #5. Defining a Memetic Algorithm
        # Define the global search method
        global_search = GA(selection=op.sel_tournament, crossover=op.cx_uniform, mutation=op.mut_gaussian, sigma=0.2, mut_indpb=0.15,
                       cxpb=0.9, tournsize=5)
        
        # Create a neighbour of a possibile solution
        def get_neighbour(problem, current_solution):
            neighbour = current_solution.copy()
            index = np.random.randint(0, len(current_solution))
            _min, _max = problem.param_bounds[0]
            neighbour[index] = np.random.uniform(_min, _max)
            return neighbour
        
        # Define the local search method
        local_search = HC(generate_neighbour=get_neighbour)
        
        # Compose the global and local search method for a Memetic Algorithm 
        self.optimizer = MA(global_search=global_search.evolve_population, sel_for_refinement=op.sel_best, local_search=local_search.stochastic_var, frequency=0.1, intensity=10)

        #6. Training our VQC
        MAX_GEN = 10
        self.res = self.optimizer.optimize(self.problem, 10, max_gen=10, verbose=True, seed=42)
        
        training_end = time.time()
        
        return TrainingResults(trainingTime=training_end-training_start, angleValues=self.res.x, totalIterations=MAX_GEN)
        #res
    
    def test(self, angles, validation_data, validation_labels, realhw=False):
        #7. Testing the optimal solution found
        angleValues = angles
        if type(angles)  == dict:
            angleValues = list(angles.values())
        
        test_predictions = [1 if self.get_label_prediction(self.circuit, features, angleValues, realhw) > 0.5 else 0 for features in validation_data]
        
        conf_matrix = confusion_matrix(validation_labels.tolist(), test_predictions)  #[[TN, FP],[FN, TP]]
        TN, FP, FN, TP = conf_matrix.ravel()

        return TrainingResults(angleValues=angles,
                               truePositive=TP, falsePositive=FP, trueNegative=TN, falseNegative=FN)
    
