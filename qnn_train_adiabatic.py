# -*- coding: utf-8 -*-
"""
Created on May 2024
 
@author: eacosta
"""
import time
from datetime import datetime
import random

import threading
from concurrent.futures import ThreadPoolExecutor
import itertools as itt
from collections import Counter
import traceback

import sympy as sp
import matplotlib.pyplot as plt

import dimod
from dwave.system import LeapHybridCQMSampler

import config as cf
import helpers as hlp
from sympy2qubo import SympyToQUBOConverter
from operatorprocessor import OperatorPreprocessor


def euclidean_distance(a, b):
    sa = sp.sympify(a)
    sb = sp.sympify(b)
    
    sorted_keys = sorted(sa.keys())
    sorted_values = [sa[k] for k in sorted_keys]
    
    sorted_keysb = sorted(sb.keys())
    sorted_valuesb = [sb[k] for k in sorted_keysb]
    
    diff = [a - b for a, b in zip(sorted_values, sorted_valuesb)]
    sum_squares = sum(x**2 for x in diff)
    
    norm = sp.sqrt(sum_squares)
    
    
    return norm

def print_sorted_combinations(level, results, top_k=None):
    combinations = results.keys()
    accuracies = [r.accuracy for r in results.values()]
    
    combo_accuracy_pairs = list(zip(combinations, accuracies))
    combo_accuracy_pairs.sort(key=lambda x: x[1], reverse=True)
    
    max_accuracy = combo_accuracy_pairs[0][1]
    best_combos = [c for c, accuracy in combo_accuracy_pairs if accuracy == max_accuracy]
    distances = []

    file_name = cf.RESULTS_FILE.format(level=level, timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    hlp.qcprint(cf.LOG_LEVEL_INFO, "Writing results to file {file_name}")
    title_line = f"=== LEVEL {level}. ALL COMBINATIONS SORTED BY COST ===\n"
    hlp.qcprint(cf.LOG_LEVEL_INFO, "\n="+title_line)
    with open(file_name, "a") as f: f.write(title_line)
    for i, (combo, accuracy) in enumerate(combo_accuracy_pairs, 1):
        if (combo == '0'):
            continue
        
        distance = min(euclidean_distance(combo, best) for best in best_combos if best != '0')
        distances.append(distance)
        params_str = combo
        print_str = f"{i:3d}. Params: ({params_str}) | Accuracy: {accuracy:.3f} | Distance to Best: {distance:.2f}\n"
        hlp.qcprint(cf.LOG_LEVEL_INFO, print_str)
        with open(file_name, "a") as f: f.write(print_str)
      
    closing_line = "=========================================\n"
    hlp.qcprint(cf.LOG_LEVEL_INFO, closing_line)
    with open(file_name, "a") as f: f.write(closing_line)

    if top_k:
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"=== LEVEL {level}. TOP {top_k} COMBINATIONS ===")
        for i, (combo, accuracy) in enumerate(combo_accuracy_pairs[:top_k], 1):
            if (combo == '0'):
                continue
            
            distance = min(euclidean_distance(combo, best) for best in best_combos if best != '0')
            params_str = combo
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"{i:3d}. Params: ({params_str}) | Accuracy: {accuracy:.3f} | Distance to Best {distance:.2f}")
        hlp.qcprint(cf.LOG_LEVEL_INFO, "===================================\n")

    max_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    hlp.qcprint(cf.LOG_LEVEL_INFO, f"LEVEL {level}. Maximum Distance to Best: {max_distance:.2f}")
    hlp.qcprint(cf.LOG_LEVEL_INFO, f"LEVEL {level}. Average Distance to Best: {avg_distance:.2f}\n")
    
def plot_accuracy_histogram(level, accuracies):
    rounded_accuracies = [round(a, 2) for a in accuracies]
    count = Counter(rounded_accuracies)
    sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)

    labels, values = zip(*sorted_items)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values, tick_label=labels, color='skyblue', edgecolor='black', label='histogram val')
    plt.title('Level'+str(level)+'. Accuracy Frequency (Descending)')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = f"adiabatic_histogram_level{level}_histogram_{format(time.time())}.png"
    plt.savefig(cf.LOG_DIR+(file_name), format='png', dpi=300)
    hlp.qcprint(cf.LOG_LEVEL_INFO, f"Printed histogram with file name {file_name}")
    
def plot_accuracy_history(level, accuracy_values):
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_values, label='Accuracy over iterations')
    plt.title('Level'+str(level)+'. Accuracy Function during Training')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    file_name = f"adiabatic_accuracy_level{level}_history_{format(time.time())}.png"
    plt.savefig(cf.LOG_DIR+(file_name), format='png', dpi=300)
    hlp.qcprint(cf.LOG_LEVEL_INFO, f"Printed cost function with file name {file_name}")

def plot_cost_function(level, cost_history):
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history, label='Training cycle')
    plt.title('Level'+str(level)+'. Cost Function Evolution (MSE)')
    plt.xlabel('Iteration')
    plt.ylabel('Training MSE')
    plt.legend()
    file_name = f"adiabatic_cost_level{level}_{format(time.time())}.png"
    plt.savefig(cf.LOG_DIR+(file_name), format='png', dpi=300)


def generate_value_sets(pivot_value, m, k, seg_start, seg_end):
    step = (seg_end - seg_start) / k
    value_sets = []

    for i in range(m):
        offset = i * step / m  # offset per group
        value_set = [sp.simplify((offset + j * step)+(step/(2*m))) for j in range(k)]
        value_sets.append(value_set)

    return value_sets

def generate_value_set(pivot_value, points, subsegments, segment_size):
    seg_start = pivot_value - (segment_size / 2)
    seg_end = pivot_value + (segment_size / 2)
    step = (seg_end - seg_start) / subsegments
    value_sets = []

    for p in range(points):
        offset = p * step / points
        value_set = [sp.simplify((offset + s * step)+(step/(2*points))) for s in range(subsegments)]
        value_sets.append(value_set)

    return value_sets

def generate_combinations_dictBKP(angles, value_sets):
    angle_names = tuple(str(sym) for sym in angles)
    combinations = list(itt.product(value_sets, repeat=len(angles)))
    combo_dict = {
        idx: {var: val for var, val in zip(angle_names, combo)}
        for idx, combo in enumerate(combinations, start=1)
    }
    return combo_dict

def generate_combinations_dict(angles, value_sets):
    angle_names = tuple(str(sym) for sym in angles)
    value_lists = [value_sets[angle] for angle in angle_names]
    combinations = list(itt.product(*value_lists))
    combo_dict = {
        idx: {var: val for var, val in zip(angle_names, combo)}
        for idx, combo in enumerate(combinations, start=1)
    }
    return combo_dict

class QNN_train_adiabatic():
    vaq = None
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

    def __init__(self, rot_angles_count, operator, discretization_parts, evaluation_points, input_list, y_list, test_data, test_labels, vaq, num_qubits):
        hlp.qcprint(cf.LOG_LEVEL_INFO, "Initializaing Adiabatic trainer")
        self.noise_level = cf.ADIABATIC_NOISE_LEVEL
        self.constraint_noise_prob = cf.ADIABATIC_CONSTRAINT_NOISE_PROB
        self.vaq = vaq
        self.quboProcessor = SympyToQUBOConverter()

        self.rot_angles_count = rot_angles_count        
        self.parts = discretization_parts
        self.points = evaluation_points
        self.num_qubits = num_qubits
        self.input_list = input_list
        self.y_list = y_list
        self.test_data = test_data
        self.test_labels = test_labels
        self.binary_vars_dimod = []
        self.binary_vars_dimod_dictionary = {}

        self.binaryVarsinit()

        self.angles = self.discretizeAngles(cf.ADIABATIC_TRAIN_LOWER_BOUND, cf.ADIABATIC_TRAIN_UPPER_BOUND)
        
        # Constraints: Only one value per each rotation angle
        self.cqm = dimod.ConstrainedQuadraticModel()
        
        for angle in self.symbolic_vars: # a0+a1+...an=1, b0+b1+..bn=1
            if random.random() < self.constraint_noise_prob:
                self.cqm.add_constraint(sum(self.binary_vars_by_angle(angle)) <= 1+self.noise_level, f't{angle} constraint') 
                self.cqm.add_constraint(sum(self.binary_vars_by_angle(angle)) >= 1-self.noise_level,f't{angle} constraint base')
            else:
                self.cqm.add_constraint(sum(self.binary_vars_by_angle(angle)) == 1, f't{angle} constraint')

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

    def expand_operatorBKP(self, possible_values, fixed_angles):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Expanding expression to {self.parts} potential values for each rotation angle, with fixed angles: {fixed_angles}")
        operator_exp = self.operator.subs(self.theta_replacements) # Substitute θ[i] with a, b, c, d....

        # substitute fixed_angles
        operator_exp = operator_exp.subs(fixed_angles)
        
        # create dict only with angles to be optimized
        possible_values = {var: possible_values for var in self.symbolic_vars if var not in fixed_angles}
        operator_expanded = operator_exp.applyfunc(lambda x: self.operatorProcessor.expand_expression(x, possible_values))
        operator_expanded = operator_expanded.applyfunc(lambda x: x.rewrite(sp.Float))  # Convert all Python floats to SymPy floats
        
        return operator_expanded

    def expand_operator(self, possible_values, fixed_angles):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Expanding expression to {self.parts} potential values for each rotation angle, with fixed angles: {fixed_angles}")
        operator_exp = self.operator.subs(self.theta_replacements) # Substitute θ[i] with a, b, c, d....

        # substitute fixed_angles
        operator_exp = operator_exp.subs(fixed_angles)
        
        # create dict only with angles to be optimized
        #possible_values = {var: possible_values for var in self.symbolic_vars if var not in fixed_angles}
        operator_expanded = operator_exp.applyfunc(lambda x: self.operatorProcessor.expand_expression(x, possible_values))
        operator_expanded = operator_expanded.applyfunc(lambda x: x.rewrite(sp.Float))  # Convert all Python floats to SymPy floats
        
        return operator_expanded

    def operate(self, operator, psi_0):
        psi_0_sp = sp.Matrix(psi_0)
        psi_0_bra = psi_0_sp.H  # Hermitian conjugate of psi_0
        result = psi_0_bra * operator
        result_simplified = result
        
        return result_simplified

    # LOG LOSS
    def log_loss(self, psi_0, psi_f):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Starting log loss on thread: {threading.current_thread().name}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"log loss item. {psi_0}")
        
        out_vs = self.operate(psi_0)

        psi_f_sp = sp.Matrix(psi_f)
        psi_f_sp_immutable = psi_f_sp.as_immutable()
        psi_f_sp_bra = psi_f_sp_immutable.H  # Hermitian conjugate of psi_f
        
        ones = sp.ones(psi_f_sp_bra.rows, psi_f_sp_bra.cols)
        diff_squared = (psi_f_sp_bra * sp.log(out_vs)) + ((ones-psi_f_sp_bra) * sp.log(ones-out_vs))
        out = diff_squared
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"log loss finish on thread: {threading.current_thread().name}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"log loss result value: {out}")

        return out

    # MSE
    def mse(self, operator, psi_0, psi_f):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Starting mse on thread: {threading.current_thread().name}")
        mse_start = time.time()
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  item. {psi_0}")
        
        out_vs = self.operate(operator, psi_0)

        psi_f_data = [sv.data for sv in psi_f]  # sv.data is a NumPy array
        #psi_f_sp = sp.Matrix(psi_f)
        # Option 1: Flatten into a column vector of length 16
        psi_f_sp = sp.Matrix(psi_f_data)
        
        # Option 2: If you want a matrix where each row is one statevector (e.g., for batch processing):
        # psi_f_sp = sp.Matrix(psi_f_data).reshape(len(psi_f), -1)  # if needed

        psi_f_sp_immutable = psi_f_sp.as_immutable()
        

        psi_f_sp_bra = psi_f_sp_immutable.H  # Hermitian conjugate of psi_f
     
        diff_vec = []
        for i in range (psi_f_sp_bra.rows):
            difference = psi_f_sp_bra.row(i) - out_vs  # calculate the difference between expected and predicted
            diff_vec.append(difference.applyfunc( lambda x: x**(2) )) # square the difference
    
        summation = [sum(elements) for elements in zip(*diff_vec)]
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Sum result value: {summation}")
        
        if cf.DS_QUANTUM:
            out = [expr**(1/2) for expr in summation]
        else:
            #out = [sum(elements)**(1/2) for elements in zip(*summation)]
            out = sum(summation)**(1/2)
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Sum squared result value: {out}")
        
        mse_end = time.time()
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"finished MSE on thread: {threading.current_thread().name}.  Took {mse_end-mse_start} sec")

        return out

    # Helper Method to sum binary variables based on the second character (var)   
    def binary_vars_by_angle(self, angle): #EA1
        b_vars = [dimod.Binary(str(list(var.linear.keys())[0])) for var in self.binary_vars_dimod if str(
            list(var.linear.keys())[0]).startswith('b'+str(angle))]
        return b_vars

    # Parallelize qubo formulation by item in dataset
    def buildQUBO_record(self, operator, in_data, expected_val, real_values, cqm_global):
        start_quborecord = time.time()
        if cf.DS_QUANTUM:
            in_data = in_data[0]
            expected_val = expected_val[0]
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Building QUBO from a training record.  Thread: {threading.current_thread().name}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  in_data: {in_data}, expected value: {expected_val}")
        
        # Calculate the squared error for given item
        mse = self.mse(operator, in_data, expected_val)
        #mse_wangles = [mse_i.subs(real_values) for mse_i in mse]
        mse_wangles = mse.subs(real_values)
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"mse_wangles :{mse_wangles}")
        
        #mse_evaltd = [msewa_i.evalf() for msewa_i in mse_wangles]
        mse_evaltd = mse_wangles.evalf()
        #mse_evaltd = [mseetd_i.replace(lambda x: x.is_Pow and x.exp > 1, lambda x: x.base) for mseetd_i in mse_evaltd]
        mse_evaltd = mse_evaltd.replace(
            lambda x: x.is_Pow and x.exp > 1,
            lambda x: x.base
        )
        #sum_term = sum(mse_evaltd)
        sum_term = mse_evaltd
        sum_term *= 1/len(self.input_list)
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, "cost function term built.")
        
        expr2, bin_vars = self.quboProcessor.create_expression(str(sum_term), self.rot_angles_count, self.parts)

        cqm_iter = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        cqm_iter = self.quboProcessor.sympy_to_qubo(expr2, bin_vars, cqm_iter)

        end_quborecord = time.time()
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  cqm_iter.variables ({len(cqm_iter.variables)}):  {cqm_iter.variables}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  cqm_iter.linear :  {cqm_iter.linear}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  cqm_iter.quadratic :  {cqm_iter.quadratic}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  cqm_iter.offset :  {cqm_iter.offset}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Finished building QUBO item on thread: {threading.current_thread().name}, took: {end_quborecord-start_quborecord} sec")   

        self.updateQUBO(cqm_iter)

    # QUBO formulation
    def buildQUBO(self, operator, input_list, y_list):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, "buildQUBO - Building QUBO Formulation")

        buildqubo_start = time.time()

        real_values = {
            sp.symbols(f'{chr(97 + var_idx)}{val_idx}'): self.angles[val_idx]
                for var_idx in range(self.rot_angles_count) # Iterate over variable prefixes (a, b, c, ...)
                for val_idx in range(self.parts) # Iterate over suffixes (0, 1, 2, ...)
        }
                
        if (cf.THREADS_QUBO>1): # Parallel mode execution
            with ThreadPoolExecutor(max_workers=cf.THREADS_QUBO) as executor:
                hlp.qcprint(cf.LOG_LEVEL_INFO, f"Starting Multi Thread QUBO construction ({cf.THREADS_QUBO} threads) execution over Dataset entries for this training iteration")
                executor.map(self.buildQUBO_record, 
                             itt.repeat(operator), input_list, y_list, itt.repeat(real_values), itt.repeat(self.cqm))
        else: #Single thread execution for easy debuggiing
            for i in range(len(input_list)):
                self.buildQUBO_record(operator, input_list[i], y_list[i], real_values, self.cqm)
        
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, "Finished Thread execution for building qubo formulation")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, "Constrained Quadratic Model:")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Variables ({len(self.cqm.variables)}):  {self.cqm.variables}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Linear terms:  {self.cqm.objective.linear}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Quadratic terms:  {self.cqm.objective.quadratic}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Offset :  {self.cqm.objective.offset}")
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"  Constraints (total: {str(len(self.cqm.constraints))}) :")
        for label, constraint in self.cqm.constraints.items():
            hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"   constraint {label}: {constraint}")
            
        buildqubo_end = time.time()
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Building QUBO took {buildqubo_end-buildqubo_start} sec")
        
        return self.cqm
    
    def updateQUBO(self, qubo_item):
        for var in qubo_item.variables:
            if var.name not in self.cqm.variables:
                self.cqm.add_variable('BINARY', var)

        for var, bias in qubo_item.linear.items():
            noise = random.uniform(-self.noise_level, self.noise_level)
            if var.name in self.cqm.objective.linear: 
                old_val = self.cqm.objective.get_linear(var.name) 
                self.cqm.objective.add_linear(var.name, old_val + bias + noise)
            else:
                self.cqm.objective.add_linear(var.name, bias+noise) 
        
        # Add quadratic biases from bqm2 to bqm1
        for (var1, var2), bias in qubo_item.quadratic.items():
            noise = random.uniform(-self.noise_level, self.noise_level)
            if (var1.name, var2.name) in self.cqm.objective.quadratic:
                old_val = self.cqm.objective.get_quadratic(var1.name, var2.name) 
                self.cqm.objective.add_quadratic(var1.name, var2.name, old_val +bias + noise)
            else:
                self.cqm.objective.add_quadratic(var1.name, var2.name, bias + noise)
        
        # Add the offset
        self.cqm.objective.offset = self.cqm.objective.offset + qubo_item.offset + random.uniform(-self.noise_level, self.noise_level)
        
        return self.cqm

    def train(self, deep_levels):

        def train_level(level, level_pivot, level_pivot_results, seg_size, level_points):
            
            def train_step(comboindex):
                try:
                    nonlocal stop_reached
                    if stop_reached:
                        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Stop condition reached previously won't execute for combination: {comboindex}\n")
                        return
                        
                    thread_id = threading.get_ident()
                    possible_values = combinations[comboindex]
                    hlp.qcprint(cf.LOG_LEVEL_INFO, f"\nTraining for combination {comboindex} begin.  Thread ID: {thread_id}. Possible values: {possible_values}.")
                    iter_result = self.train_iter(range_start, range_end, angs2optimize, possible_values, angs_fixed)
                    results_dict[str(iter_result.angleValues)] = iter_result
                    accuracy_values.append(iter_result.accuracy)
                    cost_history.append(iter_result.cost)
                    hlp.qcprint(cf.LOG_LEVEL_INFO, f"Training for combination {comboindex} finished.  Thread ID: {thread_id}. Values: {iter_result.angleValues}. Accuracy: {iter_result.accuracy}\n")
                    
                    stop_criteria =  level_pivot_results.accuracy if (level_pivot_results.accuracy > 0) else cf.ADIABATIC_TRAIN_LEVEL_TOLERANCE
                    if iter_result.accuracy >= stop_criteria:
                        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Stop condition reached with Accuracy: {iter_result.accuracy} for combination: {comboindex} - {iter_result.angleValues}\n")
                        stop_reached = True
                except Exception as e:
                    error_msg = traceback.format_exc()
                    print(traceback.format_exc())
                    hlp.qcprint(cf.LOG_LEVEL_ERROR, f"Exception in train_step: {e}\n{error_msg}")
            
            level_best_result = hlp.TrainingResults()
            
            accuracy_values = []
            accuracy_values.append(level_pivot_results.accuracy)
            cost_history = []
            cost_history.append(level_pivot_results.cost)
            angs2optimize = angs2optimize = [str(ang) for ang in level_pivot]
            angs_fixed = {}        
            results_dict = {}
            results_dict[str(level_pivot_results.angleValues)] = level_pivot_results
            value_sets = {}
            stop_reached = False
            
            for ang in level_pivot:
                value_sets[ang] = generate_value_set(level_pivot[ang], level_points, self.parts, seg_size)
            combinations = generate_combinations_dict(self.symbolic_vars, value_sets)
            shuffled_keys = list(combinations.keys())
            if cf.ADIABATIC_TRAIN_LEVEL_SUFFLE_COMBINATIONS: random.shuffle(shuffled_keys)
            
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"\nLEVEL: {level} Will train for {len(combinations)} combinations, with pivot: {level_pivot} and region size¨: {seg_size}.")
            hlp.print_avail_resources()
            
            level_training_start = time.time()
            if (cf.THREADS_MAIN>1): # Parallel mode execution
                with ThreadPoolExecutor(max_workers=cf.THREADS_MAIN) as executor:
                    hlp.qcprint(cf.LOG_LEVEL_INFO, f"Starting Multi Thread Training ({cf.THREADS_MAIN} threads)")
                    executor.map(train_step, shuffled_keys)
            else: # Single thread execution for easy debuggiing
                hlp.qcprint(cf.LOG_LEVEL_INFO, "Starting Single Thread training")
                for it in shuffled_keys:
                    train_step(it)
            level_training_end = time.time()
                    
            level_best_combination, level_best_result = max(
                ((k, v) for k, v in results_dict.items() if k != '0'),
                key=lambda item: item[1].accuracy
            )
            
            if (len(results_dict) > 0):
                print_sorted_combinations(level, results_dict, top_k=cf.ADIABATIC_TRAIN_TOPK)
                if (cf.ADIABATIC_PLOT_TRAIN_ITER_HISTOGRAM): plot_accuracy_histogram(level, [r.accuracy for r in results_dict.values()])
                if (cf.ADIABATIC_TRAIN_ITER_ACCURACY): plot_accuracy_history(level, accuracy_values)
                if (cf.PLOT_COST): plot_cost_function(level, cost_history)
            
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"\nLEVEL {level}. Training Iterations: {len(accuracy_values)}.")
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"\nLEVEL {level}. Training results:")
            [hlp.qcprint(cf.LOG_LEVEL_INFO, f"Key: {k}, Value: {v.accuracy}") for k, v in results_dict.items()]
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"Angle values after optimization: {best_result.angleValues}, Accuracy: {best_result.accuracy}")
            hlp.qcprint(cf.LOG_LEVEL_INFO, f"Adiabatic training took: {level_training_end-level_training_start} seconds")
            
            return level_best_result, len(results_dict)
            
        range_start = cf.ADIABATIC_TRAIN_LOWER_BOUND 
        range_end = cf.ADIABATIC_TRAIN_UPPER_BOUND
        
        total_iterations = 0
        best_result = hlp.TrainingResults()
        
        pivot = {str(var):random.uniform(range_start, range_end) for var in self.symbolic_vars}
        if (cf.ROTATION_ANGLES_INIT != cf.ROTATION_ANGLES_INIT_RANDOM):
            pivot = cf.ROTATION_ANGLES_INIT
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Angle values before optimization: {pivot}")  
        pivot_results = hlp.TrainingResults(accuracy=0, cost=1) # Initial results None, accuracy 0
        
        training_start = time.time()
        segment_size = range_end-range_start
        points=cf.ADIABATIC_TRAIN_POINTS
        
        for i in range(deep_levels):
            best_result, level_iterations = train_level(i, pivot, pivot_results, segment_size, points)
            
            pivot = best_result.angleValues
            pivot_results = best_result
            segment_size = segment_size*(1-cf.ADIABATIC_TRAIN_RANGE_DOWNSCALING)
            points = max(1, round(points * (1-cf.ADIABATIC_TRAIN_POINTS_DOWNSCALING)))
            
            total_iterations += level_iterations
            
        training_end = time.time()

        return hlp.TrainingResults(trainingTime=training_end-training_start, accuracy=best_result.accuracy,f1=best_result.f1,
                               angleValues=best_result.angleValues,
                               truePositive=best_result.truePositive, falsePositive=best_result.falsePositive,
                               trueNegative=best_result.trueNegative, falseNegative=best_result.falseNegative, 
                               totalIterations=total_iterations)

    def train_iter(self, range_start, range_end, angs2optimize, possible_values, angs_fixed):
        hlp.qcprint(cf.LOG_LEVEL_DEBUG, f"Training iteration.  Range start: {range_start}, Range end: {range_end}.  Angles to optimize: {angs2optimize}")
        result_angs2optimize = {}
        
        operator_expanded = self.expand_operator(possible_values, angs_fixed)
        self.buildQUBO(operator_expanded, self.input_list, self.y_list)
        
        optimal = []
        solver = dimod.ExactCQMSolver()

        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Training on {'real' if cf.ADIABATIC_REAL else 'simulated'} DWave solver")
        
        start_time = time.time()
        
        if cf.ADIABATIC_REAL:  # Real HW Training
            solver = LeapHybridCQMSampler(token=cf.DWAVE_TOKEN)
            solutions = solver.sample_cqm(self.cqm, label="experiment_" + str(self.parts) + "_parts_{}".format(time.time()), time_limit=cf.DWAVE_TIMELIMIT)
        else:  # Simulated Training
            solutions = solver.sample_cqm(self.cqm)

        end_time = time.time()
        
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Total solutions: {len(solutions)}")
        feasible_sols = solutions.filter(lambda s: s.is_feasible)
        samples_and_energies = [(s, solutions.record.energy[i]) for i, s in enumerate(feasible_sols)]
        sorted_samples = sorted(samples_and_energies, key=lambda x: x[1], reverse=False)
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Feasible solutions: {len(sorted_samples)}: {sorted_samples}")
        if len(sorted_samples) > 0:
            optimal = sorted_samples[0][0]
        else:
            hlp.qcprint(cf.LOG_LEVEL_INFO, "No feasible solution found")

        for opt in (k for k in optimal if optimal[k] == 1):
            opt_2c = str(opt)[1] # second character of the binary variable name.  ex: bd1 -> d
            opt_3c = str(opt)[2] # third character of the binary variable name.  ex: bd1 -> 1
            if opt_2c in angs2optimize:
                result_angs2optimize[opt_2c] = possible_values[opt_2c][int(opt_3c)]

        hlp.qcprint(cf.LOG_LEVEL_INFO, f"result_angs2optimize COMPLETE AFTER: {result_angs2optimize}")

        all_angles = {**result_angs2optimize, **angs_fixed}
        optimal_sorted = dict(sorted(all_angles.items()))
        iteration_result = self.vaq.test(optimal_sorted, self.test_data, self.test_labels, cf.QISKIT_REAL)

        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Training Adiabatically took {(end_time - start_time)} sec")
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Explored {str(len(solutions))} solutions")
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Feasible solutions: {str(len(feasible_sols))}")
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Optimal solution: {optimal}")
        hlp.qcprint(cf.LOG_LEVEL_INFO, f"Iteration Accuracy: {iteration_result.accuracy}")
        
        return iteration_result

    def get_angles(self, optimal):
        angles4VQC = []
        if cf.ADIABATIC_REAL:  # On real mode, optimal solution is an array
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