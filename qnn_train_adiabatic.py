# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta
"""
import time
import sympy as sp
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor
import random
from itertools import repeat

import dimod
from dwave.system import LeapHybridCQMSampler

from config import LOG_DIR
from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from config import THREADS
from config import QISKIT_REAL
from config import ADIABATIC_REAL, DWAVE_TOKEN, DWAVE_TIMELIMIT
from config import ADIABATIC_TRAIN_LOWER_BOUND, ADIABATIC_TRAIN_UPPER_BOUND
from config import ROTATION_ANGLES_INIT, ROTATION_ANGLES_INIT_RANDOM
from config import ADIABATIC_TRAIN_ITER_ACCURACY,ADIABATIC_TRAIN_ITER_TOLERANCE

from helpers import qcprint, TrainingResults
from sympy2qubo import SympyToQUBOConverter
from operatorprocessor import OperatorPreprocessor

class QNN_train_adiabatic():
    evovaq = None
    ang_div_2 = []
    cqm = None

    rot_angles_count = 0
    parts = 0
    
    num_qubits = 0
    input_list = []
    y_list = []

    operator = []
    operator_expanded = []

    symbolic_vars = []
    var_values = []
    binary_vars = []
    binary_vars_dimod = []
    binary_vars_dimod_dictionary = {}

    theta_replacements = []
    angles = []

    quboProcessor = None
    operatorProcessor = None

    def __init__(self, rot_angles_count, operator, discretization_parts, input_list, y_list, test_data, test_labels, evovaq, num_qubits):
        qcprint(LOG_LEVEL_INFO, "Initializaing Adiabatic trainer")
        self.evovaq = evovaq
        self.quboProcessor = SympyToQUBOConverter()

        self.rot_angles_count = rot_angles_count        
        self.parts = discretization_parts
        self.num_qubits = num_qubits
        self.input_list = input_list
        self.y_list = y_list
        self.test_data = test_data
        self.test_labels = test_labels
        self.binary_vars_dimod = []
        self.binary_vars_dimod_dictionary = {}

        self.binaryVarsinit()

        self.angles = self.discretizeAngles(ADIABATIC_TRAIN_LOWER_BOUND, ADIABATIC_TRAIN_UPPER_BOUND)
        
        # Constraints: Only one value per each rotation angle
        self.cqm = dimod.ConstrainedQuadraticModel()
        for angle in self.symbolic_vars:
            self.cqm.add_constraint(sum(self.binary_vars_by_angle(angle)) == 1, f't{angle} constraint')  # a0+a1+...an=1, b0+b1+..bn=1

        self.operatorProcessor = OperatorPreprocessor(rot_angles_count,discretization_parts)
        self.operator = operator

    # Create Binary variables for each rotation angle theta
    def binaryVarsinit(self):
        # 'a', 'b', 'c', 'd', 'e', ...
        self.symbolic_vars = sp.symbols(' '.join([chr(97 + i) for i in range(self.rot_angles_count)]))

        for var in self.symbolic_vars:
            # 'a0'. 'a1' ... 'an', 'b0', 'b1' ... 'bn'  n=parts
            self.var_values.extend([sp.symbols(f'{var}{i}') for i in range(self.parts)])
            # python binaries: 'ba0'. 'ba1' ... 'ban', 'bb0', 'bb1' ... 'bbn'  n=parts
            self.binary_vars.extend([sp.symbols(f'b{var}{i}') for i in range(self.parts)])
            for i in range(self.parts):
                var_name = f'b{var}{i}'
                var_symbol = sp.Symbol(var_name)
                dimod_var = dimod.Binary(var_symbol)
                self.binary_vars_dimod.append(dimod_var)
                self.binary_vars_dimod_dictionary[var_name] = dimod_var

        self.theta_replacements = {f'θ[{i}]': str(self.symbolic_vars[i]) for i in range(len(self.symbolic_vars))}

    def discretizeAngles(self, range_start, range_end):
        lp = range_end/self.parts
        angles = [((range_start-(lp/2)) + ((i+1)*lp)) for i in range(self.parts)]

        return angles

    # Helper method to define the replacement for all variables with binary coefficients.
    # Works for expressions containing only 'exp' math functions
    def expand_expression(self, expr):
        expanded_expr = expr
        for var in self.symbolic_vars:  # for a, b, c, d....
            for arg in expr.atoms(sp.exp):  # Find all exp() in the expression
                if var in arg.args[0].free_symbols:
                    f_binary_vars = [value for value in self.binary_vars if str(value)[1] == str(var)]
                    f_var_values = [value for value in self.var_values if str(value).startswith(str(var))]
                    coefficient = arg.args[0] / var
                    expanded_part = sp.Add(*[
                        f_binary_vars[i] *
                        sp.exp(coefficient * f_var_values[i])
                        for i in range(self.parts)
                    ])
                    
                    expanded_expr = expanded_expr.replace(arg, sp.Add(expanded_part))

        # Replace Sympy binary variables with Dwave (Dimod) binary variables
        replacement_dict = {
            sp.symbols(f'b{chr(97 + var_idx)}{val_idx}'): sp.Symbol(f'b{chr(97 + var_idx)}{val_idx}')
            for var_idx in range(self.rot_angles_count)
            for val_idx in range(self.parts)
        }

        # Use .subs() to replace symbolic binary variables with placeholders
        final_expr = expanded_expr.subs(replacement_dict)

        return final_expr

    def expand_operator(self, possible_values, fixed_angles):
        #operator_start = time.time()
        qcprint(LOG_LEVEL_DEBUG, f"Expanding expression to {self.parts} potential values for each rotation angle, with fixed angles: {fixed_angles}")
        operator_exp = self.operator.subs(self.theta_replacements) # Substitute θ[i] with a, b, c, d....

        # substitute fixed_angles
        operator_exp = operator_exp.subs(fixed_angles)
        
        # create dict only with angles to be optimized
        possible_values = {var: possible_values for var in self.symbolic_vars if var not in fixed_angles}
        operator_expanded = operator_exp.applyfunc(lambda x: self.operatorProcessor.expand_expression(x, possible_values))
        operator_expanded = operator_expanded.applyfunc(lambda x: x.rewrite(sp.Float))  # Convert all Python floats to SymPy floats
        #operator_end = time.time()
        
        #print(f"\nOperator Expanded.  Took: {operator_end-operator_start} sec")
        
        return operator_expanded

    def operate(self, operator, psi_0):
        psi_0_sp = sp.Matrix(psi_0)
        psi_0_bra = psi_0_sp.H  # Hermitian conjugate of psi_0
        result = psi_0_bra * operator
        result_simplified = result
        
        return result_simplified

    # LOG LOSS
    def log_loss(self, psi_0, psi_f):
        qcprint(LOG_LEVEL_DEBUG, f"Starting log loss on thread: {threading.current_thread().name}")
        qcprint(LOG_LEVEL_DEBUG, f"log loss item. {psi_0}")
        
        out_vs = self.operate(psi_0)

        psi_f_sp = sp.Matrix(psi_f)
        psi_f_sp_immutable = psi_f_sp.as_immutable()
        psi_f_sp_bra = psi_f_sp_immutable.H  # Hermitian conjugate of psi_f
        
        ones = sp.ones(psi_f_sp_bra.rows, psi_f_sp_bra.cols)
        diff_squared = (psi_f_sp_bra * sp.log(out_vs)) + ((ones-psi_f_sp_bra) * sp.log(ones-out_vs))
        out = diff_squared
        
        qcprint(LOG_LEVEL_DEBUG, f"log loss finish on thread: {threading.current_thread().name}")
        qcprint(LOG_LEVEL_DEBUG, f"log loss result value: {out}")

        return out

    # MSE
    def mse(self, operator, psi_0, psi_f):
        qcprint(LOG_LEVEL_DEBUG, f"Starting mse on thread: {threading.current_thread().name}")
        mse_start = time.time()
        
        qcprint(LOG_LEVEL_DEBUG, f"  item. {psi_0}")
        
        out_vs = self.operate(operator, psi_0)

        psi_f_sp = sp.Matrix(psi_f)
        psi_f_sp_immutable = psi_f_sp.as_immutable()

        psi_f_sp_bra = psi_f_sp_immutable.H  # Hermitian conjugate of psi_f
     
        diff_vec = []
        for i in range (psi_f_sp_bra.rows):
            difference = psi_f_sp_bra.row(i) - out_vs  # calculate the difference between expected and predicted
            diff_vec.append(difference.applyfunc(lambda x: x**2)) # square the difference
    
        out = [sum(elements) for elements in zip(*diff_vec)]
        qcprint(LOG_LEVEL_DEBUG, f"  mse result value: {out}")
        
        mse_end = time.time()
        
        qcprint(LOG_LEVEL_DEBUG, f"finished MSE on thread: {threading.current_thread().name}.  Took {mse_end-mse_start} sec")
        #print(f"  MSE Took {mse_end-mse_start} sec")

        return out

    # Helper Method to sum binary variables based on the second character (var)   
    def binary_vars_by_angle(self, angle): #EA1
        b_vars = [dimod.Binary(str(list(var.linear.keys())[0])) for var in self.binary_vars_dimod if str(
            list(var.linear.keys())[0]).startswith('b'+str(angle))]
        return b_vars

    # Parallelize qubo formulation by item in dataset
    def buildQUBO_record(self, operator, in_data, expected_val, real_values, cqm_global):
        start_quborecord = time.time()
        qcprint(LOG_LEVEL_DEBUG, f"Building QUBO from a training record.  Thread: {threading.current_thread().name}")
        qcprint(LOG_LEVEL_DEBUG, f"  in_data: {in_data}, expected value: {expected_val}")
        
        # Calculate the squared error for given item
        mse = self.mse(operator, in_data, expected_val)
        mse_wangles = [mse_i.subs(real_values) for mse_i in mse]
        
        qcprint(LOG_LEVEL_DEBUG, f"mse_wangles :{mse_wangles}")
        
        #begin_evalf = time.time()
        mse_evaltd = [msewa_i.evalf() for msewa_i in mse_wangles]
        #end_evalf = time.time()
        #print(f"  Evalf, took: {end_evalf-begin_evalf} sec") 
        #begin_expand = time.time()
        #mse_evaltd = [sp.expand(mseev_i) for mseev_i in mse_evaltd]  # TODO:  Enable it back!!
        #end_expand = time.time()
        #print(f"  Expanding, took: {end_expand-begin_expand} sec")  
        #start_replacePow= time.time()
        mse_evaltd = [mseetd_i.replace(lambda x: x.is_Pow and x.exp > 1, lambda x: x.base) for mseetd_i in mse_evaltd]
        #end_qreplacePow = time.time()
        #print(f"  ReplacePow, took: {end_qreplacePow-start_replacePow} sec")  
        sum_term = sum(mse_evaltd)
        sum_term *= 1/len(self.input_list)
        
        qcprint(LOG_LEVEL_DEBUG, "cost function term built.")
        
        expr2, bin_vars = self.quboProcessor.create_expression(str(sum_term), self.rot_angles_count, self.parts)

        cqm_iter = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        cqm_iter = self.quboProcessor.sympy_to_qubo(expr2, bin_vars, cqm_iter)

        end_quborecord = time.time()
        
        qcprint(LOG_LEVEL_DEBUG, f"  cqm_iter.variables ({len(cqm_iter.variables)}):  {cqm_iter.variables}")
        qcprint(LOG_LEVEL_DEBUG, f"  cqm_iter.linear :  {cqm_iter.linear}")
        qcprint(LOG_LEVEL_DEBUG, f"  cqm_iter.quadratic :  {cqm_iter.quadratic}")
        qcprint(LOG_LEVEL_DEBUG, f"  cqm_iter.offset :  {cqm_iter.offset}")
        qcprint(LOG_LEVEL_DEBUG, f"Finished building QUBO item on thread: {threading.current_thread().name}, took: {end_quborecord-start_quborecord} sec")   

        #print(f"Building QUBO record, took: {end_quborecord-start_quborecord} sec")             

        self.updateQUBO(cqm_iter)

    # QUBO formulation
    def buildQUBO(self, operator, input_list, y_list):
        qcprint(LOG_LEVEL_DEBUG, "buildQUBO - Building QUBO Formulation")

        buildqubo_start = time.time()

        real_values = {
            sp.symbols(f'{chr(97 + var_idx)}{val_idx}'): self.angles[val_idx]
                for var_idx in range(self.rot_angles_count) # Iterate over variable prefixes (a, b, c, ...)
                for val_idx in range(self.parts) # Iterate over suffixes (0, 1, 2, ...)
        }
                
        if (THREADS>1): # Parallel mode execution
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                qcprint(LOG_LEVEL_INFO, f"Starting Multi Thread ({THREADS} threads) execution over Dataset entries for this training iteration")
                #executor.map(lambda args: self.buildQUBO_record(*args, self.cqm), zip(operator, input_list, y_list, real_values))
                executor.map(self.buildQUBO_record, #repeat(self),
                                      repeat(operator), input_list, y_list, repeat(real_values), repeat(self.cqm))
              
        else:  # Single Thread execution
            for i in range(len(input_list)):
                self.buildQUBO_record(operator, input_list[i], y_list[i], real_values, self.cqm)
        
        qcprint(LOG_LEVEL_DEBUG, "Finished Thread execution for building qubo formulation")
        qcprint(LOG_LEVEL_INFO, "Constrained Quadratic Model:")
        qcprint(LOG_LEVEL_INFO, f"  Variables ({len(self.cqm.variables)}):  {self.cqm.variables}")
        qcprint(LOG_LEVEL_INFO, f"  Linear terms:  {self.cqm.objective.linear}")
        qcprint(LOG_LEVEL_INFO, f"  Quadratic terms:  {self.cqm.objective.quadratic}")
        qcprint(LOG_LEVEL_INFO, f"  Offset :  {self.cqm.objective.offset}")
        qcprint(LOG_LEVEL_INFO, f"  Constraints (total: {str(len(self.cqm.constraints))}) :")
        for label, constraint in self.cqm.constraints.items():
            qcprint(LOG_LEVEL_DEBUG, f"   constraint {label}: {constraint}")
            
        buildqubo_end = time.time()
        qcprint(LOG_LEVEL_INFO, f"Building QUBO took {buildqubo_end-buildqubo_start} sec")
        
        #print(f"Building QUBO Global took {buildqubo_end-buildqubo_start} sec")

        return self.cqm
    
    def updateQUBO(self, qubo_item):
        for var in qubo_item.variables:
            if var.name not in self.cqm.variables: #EA1 changed var for var.name
                self.cqm.add_variable('BINARY', var)

        for var, bias in qubo_item.linear.items():
            if var.name in self.cqm.objective.linear: #EA1 Changed var for var.name
                old_val = self.cqm.objective.get_linear(var.name) #EA1 Changed var for var.name
                self.cqm.objective.add_linear(var.name, old_val+bias) #EA1 Changed var for var.name
            else:
                self.cqm.objective.add_linear(var.name, bias) #EA1 Changed var for var.name
        
        # Add quadratic biases from bqm2 to bqm1
        for (var1, var2), bias in qubo_item.quadratic.items():
            if (var1.name, var2.name) in self.cqm.objective.quadratic:  #EA1 Changed var for var.name
                old_val = self.cqm.objective.get_quadratic(var1.name, var2.name)  #EA1 Changed var for var.name
                self.cqm.objective.add_quadratic(var1.name, var2.name, old_val +bias)  #EA1 Changed var for var.name
            else:
                self.cqm.objective.add_quadratic(var1.name, var2.name, bias)  #EA1 Changed var for var.name
        
        # Add the offset
        self.cqm.objective.offset = self.cqm.objective.offset + qubo_item.offset
        
        return self.cqm

    def train(self, deep_levels):
        range_start = ADIABATIC_TRAIN_LOWER_BOUND 
        range_end = ADIABATIC_TRAIN_UPPER_BOUND
        
        ang_values = {str(var):random.uniform(range_start, range_end) for var in self.symbolic_vars}
        if (ROTATION_ANGLES_INIT != ROTATION_ANGLES_INIT_RANDOM):
            ang_values = ROTATION_ANGLES_INIT
        qcprint(LOG_LEVEL_INFO, f"Angle values before optimization: {ang_values}")
        training_start = time.time()
        
        total_iterations = 0
        training_iterations = 0
        
        iter_result = TrainingResults()
        best_result = TrainingResults()
        last_result = TrainingResults()
        accuracy_values = []
        abortTraining = False
        for deepl in range (deep_levels): 
            if abortTraining:
                break
            
            qcprint(LOG_LEVEL_INFO, f"\nDeep level: {deepl+1}. Segments: {self.parts**(deepl)}")
            seg_size = sp.Abs(range_end - range_start) / self.parts**(deepl)
            qcprint(LOG_LEVEL_INFO, f"Segment size = {seg_size}")
            for seg_num in range(self.parts**(deepl)): # 2**i rangos de búsqueda potencial.  Ej 8 para i = 3
                total_iterations += 1
                seg_start = range_start + (seg_size * seg_num)
                seg_end = seg_start + seg_size
                qcprint(LOG_LEVEL_INFO, f"\n Iteration: {total_iterations}.  Segment {seg_num}.  Starting point {seg_start}, End point: {seg_end}")
                
                lp = sp.Abs(seg_end - seg_start)/self.parts
                possible_values = [((seg_start-(lp/2)) + ((i+1)*lp)) for i in range(self.parts)]
                qcprint(LOG_LEVEL_INFO, f" Discrete values :  {possible_values}")
                
                ang_list = list(ang_values)
                angs2optimize = []
                angs_fixed = {}
                for ang in ang_list:
                    if ang in ang_values and \
                        seg_start <= ang_values.get(ang) <= seg_end:
                        angs2optimize.append(str(ang))
                    else:
                        angs_fixed[str(ang)] = ang_values[ang]
                qcprint(LOG_LEVEL_INFO, f" Angles to optimize: {angs2optimize}")
                qcprint(LOG_LEVEL_INFO, f" Angles to keep fixed: {angs_fixed}")
                
                if (len(angs2optimize) == 0):
                    qcprint(LOG_LEVEL_INFO, " No angles in this segment, skip training on this segment.")
                    continue
                
                iter_result = self.train_iter(seg_start, seg_end, angs2optimize, possible_values, angs_fixed)
                accuracy_values.append(iter_result.accuracy)
                training_iterations +=1
                qcprint(LOG_LEVEL_INFO, f" Optimized angles after iteration: {iter_result.angleValues}")
                if (ADIABATIC_TRAIN_ITER_ACCURACY):
                    qcprint(LOG_LEVEL_INFO, f" Iteration Accuracy: {iter_result.accuracy}.  Previous accuracy: {last_result.accuracy}.  Best accuracy: {best_result.accuracy}")
                
                ang_values.update({ang: iter_result.angleValues[ang] for ang in ang_values if ang in iter_result.angleValues})
          
                if (ADIABATIC_TRAIN_ITER_ACCURACY) and (training_iterations>1 and iter_result.accuracy < last_result.accuracy):
                    if last_result.accuracy - iter_result.accuracy > ADIABATIC_TRAIN_ITER_TOLERANCE:
                        qcprint(LOG_LEVEL_INFO, f" Iteration Accuracy is bellow threshold {ADIABATIC_TRAIN_ITER_TOLERANCE}.  Abort training.")
                        abortTraining = True
                        break
                    else:
                        last_result = iter_result
                        if (iter_result.accuracy > best_result.accuracy):
                            best_result = iter_result
                            best_angles = ang_values.copy()
                else:
                    last_result = iter_result
                    best_result = iter_result
                    best_angles = ang_values.copy()
                    
        training_end = time.time()
        
        qcprint(LOG_LEVEL_INFO, f"\nTotal Iterations: {total_iterations}.  Training Iterations: {training_iterations}.")
        qcprint(LOG_LEVEL_INFO, f"Angle values after optimization: {best_angles}")
        qcprint(LOG_LEVEL_INFO, f"Adiabatic training took: {training_end-training_start} seconds")
        
        # Plotting the cost function
        if (ADIABATIC_TRAIN_ITER_ACCURACY):
            plt.plot(accuracy_values, label='Accuracy over iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Function during Training')
            plt.legend()
            plt.savefig(LOG_DIR+("accuracy_adiabatic_{}.png".format(time.time())), format='png', dpi=300)
        
        return TrainingResults(trainingTime=training_end-training_start, angleValues=best_angles,
                               truePositive=best_result.truePositive, falsePositive=best_result.falsePositive,
                               trueNegative=best_result.trueNegative, falseNegative=best_result.falseNegative, 
                               totalIterations=training_iterations)

    def train_iter(self, range_start, range_end, angs2optimize, possible_values, angs_fixed):
        qcprint(LOG_LEVEL_INFO, f"Training iteration.  Range start: {range_start}, Range end: {range_end}.  Angles to optimize: {angs2optimize}")
        result_angs2optimize = {}
        
        operator_expanded = self.expand_operator(possible_values, angs_fixed)
        self.buildQUBO(operator_expanded, self.input_list, self.y_list) # Initialize empty Summation
        
        optimal = []
        solver = dimod.ExactCQMSolver()

        qcprint(LOG_LEVEL_INFO, f"Training on {'real' if ADIABATIC_REAL else 'simulated'} DWave solver")
        
        start_time = time.time()
        
        if ADIABATIC_REAL:  # Real HW Training
            solver = LeapHybridCQMSampler(token=DWAVE_TOKEN)
            solutions = solver.sample_cqm(self.cqm, label="experiment_" + str(self.parts) + "_parts_{}".format(time.time()), time_limit=DWAVE_TIMELIMIT)
        else:  # Simulated Training
            solutions = solver.sample_cqm(self.cqm)

        end_time = time.time()
        
        qcprint(LOG_LEVEL_INFO, f"Total solutions: {len(solutions)}")
        feasible_sols = solutions.filter(lambda s: s.is_feasible)
        samples_and_energies = [(s, solutions.record.energy[i]) for i, s in enumerate(feasible_sols)]
        sorted_samples = sorted(samples_and_energies, key=lambda x: x[1], reverse=False)
        qcprint(LOG_LEVEL_INFO, f"Feasible solutions: {len(sorted_samples)}")
        if len(sorted_samples) > 0:
            optimal = sorted_samples[0][0]
        else:
            qcprint(LOG_LEVEL_INFO, "No feasible solution found")

        for opt in optimal:
            opt_2c = str(opt)[1] # second character of the binary variable name.  ex: bd1 -> d
            opt_3c = str(opt)[2] # third character of the binary variable name.  ex: bd1 -> 1
            if opt_2c in angs2optimize:
                result_angs2optimize[opt_2c] = possible_values[int(opt_3c)]

        if (ADIABATIC_TRAIN_ITER_ACCURACY):
            all_angles = {**result_angs2optimize, **angs_fixed}
            optimal_sorted = dict(sorted(all_angles.items()))
            iteration_result = self.evovaq.test(optimal_sorted, self.test_data, self.test_labels, QISKIT_REAL)

        qcprint(LOG_LEVEL_INFO, f"Training Adiabatically took {(end_time - start_time)} sec")
        qcprint(LOG_LEVEL_INFO, f"Explored {str(len(solutions))} solutions")
        qcprint(LOG_LEVEL_INFO, f"Feasible solutions: {str(len(feasible_sols))}")
        qcprint(LOG_LEVEL_INFO, f"Optimal solution: {optimal}")
        if (ADIABATIC_TRAIN_ITER_ACCURACY):
            qcprint(LOG_LEVEL_DEBUG, f"Iteration Accuracy: {iteration_result.accuracy}")
        
        return iteration_result

    def get_angles(self, optimal):
        angles4VQC = []
        if ADIABATIC_REAL:  # On real mode, optimal solution is an array
            sol_ind = 0
            for i in range(self.rot_angles_count):  # w, x, y, z, etc
                for j in range(self.parts):
                    if (optimal[sol_ind] == 1):
                        angles4VQC.append(self.angles[j])
                    sol_ind += 1
        else:  # On simulations, optimal solution is a dictionary
            angles_value1 = {key: value for key,
                             value in optimal.items() if value == 1}
            for ang in angles_value1:
                angles4VQC.append(self.angles[int(str(ang)[2])])

        return angles4VQC