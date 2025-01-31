#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:14:49 2024

@author: ernesto.acosta
"""

import sympy as sp
import itertools
from helpers import qcprint
from config import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_ERROR
from config import RUN_UNIT_TESTS, RUN_UNIT_TESTS_VERBOSE

class OperatorPreprocessor:
    rot_angles_count = 0
    parts = 0
    angles = []
    symbolic_vars = []
    theta_replacements = []
    var_values = []
    binary_vars = []
    
    def __init__(self, rot_angles_count, parts):
        self.rot_angles_count = rot_angles_count
        self.parts = parts
        
        self.symbolic_vars = sp.symbols(' '.join([chr(97 + i) for i in range(rot_angles_count)]))
        
        for var in self.symbolic_vars:  # a, b, c, d....
            self.var_values.extend([sp.symbols(f'{var}{i}') for i in range(parts)]) # 'a0'. 'a1' ... 'an', 'b0', 'b1' ... 'bn'  n=parts
            self.binary_vars.extend([sp.symbols(f'b{var}{i}') for i in range(parts)])             # binaries: 'ba0'. 'ba1' ... 'ban', 'bb0', 'bb1' ... 'bbn'  n=parts
        
        self.theta_replacements = {f'θ[{i}]': str(self.symbolic_vars[i]) for i in range(len(self.symbolic_vars))}

    def is_exp_pair(self, term):
        if not isinstance(term, sp.Add):
            return False
    
        if len(term.args) != 2:  # Try to split the term into two parts (exp(A*x) + exp(-A*x))
            return False
    
        term1, term2 = term.args
        if (isinstance(term1, (sp.Mul, sp.exp)) and isinstance(term2, (sp.Mul, sp.exp))):
            exp1 = sp.Add.identity
            exp2 = sp.Add.identity
            if isinstance(term1, sp.Mul):  # -exp(x)
                exp1 = term1.args[1]
            elif isinstance(term1, sp.exp):
                exp1 = term1
            if isinstance(term2, sp.Mul):  # -exp(x)
                exp2 = term2.args[1]
            elif isinstance(term2, sp.exp):
                exp2 = term2
            
            exp1_arg = exp1.args[0]
            exp2_arg = exp2.args[0]
            
            if (exp1_arg == -exp2_arg) or (exp2_arg == -exp1_arg) :
                return True
 
        return False
    
    def exppair2hiperb(self, term):
        if not self.is_exp_pair(term):
            return term
        
        term1_negative_coeff = False
        term2_negative_coeff = False
        
        term1 = term.args[0]
        term2 = term.args[1]
        if isinstance(term1, sp.Mul):  # -exp(x) or N*exp(x)
            term1_exp = term1.args[1]
            term1_exp_arg = term1_exp.args[0]
            if isinstance(term1.args[0], sp.Integer): # Coefficient like -1
                term1_coeff_num = sp.Abs(term1.args[0])
                term1_coeff_denom = 1
            elif isinstance(term1.args[0], sp.Float): # Coefficient like 0.5
                term_rational = sp.nsimplify(term1.args[0])
                if (isinstance(term_rational, sp.Rational)):
                    term1_coeff_num = sp.Abs(term_rational.p)
                    term1_coeff_denom = sp.Abs(term_rational.q)
                else:
                    term1_coeff_num = sp.Abs(term1.args[0])
                    term1_coeff_denom = 1
            elif isinstance(term1.args[0], sp.Rational): # Coefficient like 1/2
                term1_coeff_num = sp.Abs(term1.args[0].p) if len(term1.args)>0 and isinstance(term1.args[0], sp.Number) else 1    
                term1_coeff_denom = sp.Abs(term1.args[0].q) if len(term1.args)>0 and isinstance(term1.args[0], sp.Number) else 1
            if (term1.args[0] < 0): term1_negative_coeff = True
        elif isinstance(term1, sp.exp):
            exp1 = term1
            term1_exp_arg = exp1.args[0]
            term1_coeff_num = 1
            term1_coeff_denom = 1
        if isinstance(term2, sp.Mul):  # -exp(x) or N*exp(x)
            if (term2.args[0] < 0): term2_negative_coeff = True

        factor = 1
        if term1_negative_coeff != term2_negative_coeff:
            if term1_negative_coeff == True and (term1_coeff_num/term1_coeff_denom) > 0:
                factor = -1
                
            return factor*(2*term1_coeff_num/term1_coeff_denom) * sp.sinh(term1_exp_arg)
        else:
            if term1_negative_coeff == True:
                factor = -1
            
            return factor*(2*term1_coeff_num/term1_coeff_denom) * sp.cosh(term1_exp_arg)
 
    def matrix2hiperbolic(self, matrix):
        result = sp.MutableDenseMatrix.zeros(matrix.rows, matrix.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                qcprint(LOG_LEVEL_DEBUG, f"Original matrix element [{i}, {j}] : {matrix[i, j]}")
                result[i, j] = self.term2hiperbolic(matrix[i, j])
                qcprint(LOG_LEVEL_DEBUG, f"Transformed matrix element [{i}, {j}] : {result[i, j]}")
        
        return result
 
    def term2hiperbolic(self, expr):
        result = expr
        if self.is_exp_pair(expr):
            return self.exppair2hiperb(expr)
        
        if isinstance(expr, sp.Basic):
            new_args = []
            
            for arg in expr.args:
                transformed_arg = self.term2hiperbolic(arg)
                new_args.append(transformed_arg)
            
            if len(new_args)>0:  # Reconstruct the expression with the transformed argument
                result = expr.func(*new_args)
                
        return result 
   
    def hiperbprod2expsum(self, expr):
        result = expr
        if isinstance(expr, sp.Mul):
            expr_expanded = (expr).rewrite(sp.exp).expand()
            result = sp.powsimp(expr_expanded, force=True)

        return result
    
    def matrix2expsums(self, matrix):
        result = sp.MutableDenseMatrix.zeros(matrix.rows, matrix.cols)
        
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result[i, j] = self.term2expsums(matrix[i, j])
            
        return result
    
    def term2expsums(self, expr):
        if isinstance(expr, sp.Mul) or isinstance(expr, sp.cosh) or isinstance(expr, sp.sinh) or isinstance(expr, sp.cos) or isinstance(expr, sp.sin):
            exponents = []
    
            coeff = 1
            cosh = False
            sinh = False
            cos = False
            sin = False
            
            if isinstance(expr, sp.cosh) or isinstance(expr, sp.sinh) or isinstance(expr, sp.cos) or isinstance(expr, sp.sin):
                coeff = 1
                exponents.append(expr.args[0])
                if isinstance(expr, sp.cosh): cosh = True
                if isinstance(expr, sp.sinh): sinh = True
                if isinstance(expr, sp.cos): cos = True
                if isinstance(expr, sp.sin): sin = True
            
            elif isinstance(expr, sp.Mul):
                for arg in expr.args:
                    if isinstance(arg, sp.Number):
                        coeff = arg
                    elif isinstance(arg, sp.cosh) or isinstance(arg, sp.sinh) or isinstance(arg, sp.cos) or isinstance(arg, sp.sin):
                        exponents.append(arg.args[0])
                        if isinstance(arg, sp.cosh): cosh = True
                        elif isinstance(arg, sp.sinh): sinh = True
                        elif isinstance(arg, sp.cos): cos = True
                        elif isinstance(arg, sp.sin): sin = True
                    elif arg.is_Mul:
                        # If the argument is a multiplication, iterate through its factors
                        for sub_arg in arg.args:
                            if sub_arg.is_Add and len(sub_arg.args) == 1 and \
                                (sub_arg.has(sp.cosh) or sub_arg.has(sp.sinh) or sub_arg.has(sp.cos) or sub_arg.has(sp.sin)):
    
                                exponents.append(sub_arg.args[0].args[0])  # Get the hiperb's argument
                                if sub_arg.has(sp.cosh): cosh = True
                                elif sub_arg.has(sp.sinh): sinh = True
                                elif sub_arg.has(sp.cos): cos = True
                                elif sub_arg.has(sp.sin): sin = True
    
            # Create the sum of exponentials from the collected exponents
            if exponents:
                sum_expr = None
                sign_combinations = list(itertools.product([1, -1], repeat=len(exponents)))
                
                # Iterate over each sign combination
                for signs in sign_combinations:
                    exp_elem = sp.Add(*[sign * elem for sign, elem in zip(signs, exponents)]) 
                    if sum_expr == None:
                        sum_expr = sp.exp(exp_elem)
                    else:
                        if (cosh):
                            sum_expr += sp.exp(exp_elem)
                        elif (sinh):
                            sum_expr -= sp.exp(exp_elem)
                        elif (cos):
                            sum_expr += sp.exp(exp_elem)
                        elif (sin):
                                sum_expr -= sp.exp(exp_elem)
    
                if cos: coeff *= sp.I
                if sin: coeff *= -sp.I
    
                # Divide by the appropriate factor (2^N, where N is the number of terms)
                return (coeff / (2 ** len(exponents))) * sum_expr
    
        return expr
    
    def expand_expression(self, expr, possible_values):
        if isinstance(expr, sp.Add) or isinstance(expr, sp.Mul):
            expr_expanded = 0
            for term in expr.args:
                if isinstance(term, sp.Number) or (term == sp.I):  # Number term like -0.5 defined as a Mul: -1 * 0.5, or I
                    return term
                else:
                    expr_expanded += self.expand_term(term, possible_values).evalf()
            
            return expr_expanded
        
        return expr
    
    def expand_term(self, expr, possible_values):
        coeff, exp_term = expr.as_coeff_Mul()
        
        # Ensure that the exponential term is of type exp
        exp_arguments = []
        if exp_term.func == sp.exp:
            exp_arguments = exp_term.args[0].as_ordered_terms()
        else:
            qcprint(LOG_LEVEL_ERROR, "The expression must contain an exponential term, cannot expand term {expr}.")
            #raise ValueError("The expression must contain an exponential term.")
        
        expanded_expr = 0
        terms_with_coeff = {}
        
        for arg in exp_arguments:
            if arg.is_Mul:
                coeffs, var = arg.as_coeff_Mul()  # Separate coefficient and variable
                if var in possible_values:
                    terms_with_coeff[var] = coeffs
            elif arg in possible_values:
                terms_with_coeff[arg] = 1  # If no coefficient is present, assume it is 1
        
        terms = list(terms_with_coeff.keys())
        
        # Create all combinations of predefined values (using Cartesian product)
        combination_values = itertools.product(*(possible_values[term] for term in terms))
        for combination in combination_values:
            binary_product = 1
            exp_argument = 0
            
            comb = 0
            ang_num = 0
            for ang in terms:
                ang_index = possible_values[ang].index(combination[ang_num])
                binary_var = sp.symbols(f'b{ang}{ang_index}')
                binary_product *= binary_var
                coeff = terms_with_coeff[ang]
                exp_argument += coeff * combination[ang_num]
                
                ang_num += 1
                comb += 1
            
            # Multiply binary variables with the corresponding exponential term
            expanded_expr += binary_product * sp.exp(exp_argument)
        
        # Multiply the result by the coefficient outside the exponential
        expanded_expr *= coeff
        
        return expanded_expr
 
    def discretizeAngles(self, parts):
        # Angles ϴ (radians)
        L = 1 * sp.pi
        spoint = 0
        lp = L/parts
        
        # Take mid point of each part
        return [((spoint-(lp/2)) + ((i+1)*lp)) for i in range(parts)]
 
    def unit_test(self, tst_name, expr, verbose=True):
        if verbose: print(f"\nTest: {tst_name}")
        if verbose: print(f"Original: {expr}")
        
        expr_hiperb = processor.term2hiperbolic(expr)
        if verbose: print(f"Transformed to hiperb: {expr_hiperb}")
        
        if isinstance(expr_hiperb, sp.Mul):
            hiperb_count = expr_hiperb.count(sp.cosh) + expr_hiperb.count(sp.sinh)
            
            if hiperb_count == 1:
                expr_expsums = processor.term2expsums(expr_hiperb)
                if verbose: print(f"Transformed to exp sums: {expr_expsums}")
            elif hiperb_count > 1:
                expr_expsums = processor.hiperbprod2expsum(expr_hiperb)
                if verbose: print(f"Transformed to exp sums: {expr_expsums}")
        
        return expr_hiperb, expr_expsums
 
# Run unit test
if (RUN_UNIT_TESTS):
    processor = OperatorPreprocessor(4,2)
    
    # Define variables and expression
    a, b, c, d = sp.symbols('a b c d')

    tests = {}
    tests['cosh(a/2)'] = (sp.exp(a/2) + sp.exp(-a/2),
                        2*sp.cosh(a/2),
                        sp.exp(a/2) + sp.exp(-a/2))
    
    tests['cosh(c/2)'] = (sp.exp(c/2) + sp.exp(-c/2),
                        2*sp.cosh(c/2),
                        sp.exp(c/2) + sp.exp(-c/2))

    tests['cosh(-a/2)'] = (sp.exp(-a/2) + sp.exp(a/2),
                        2.*sp.cosh(a/2),
                        sp.exp(a/2) + sp.exp(-a/2))
    
    tests['-cosh(a/2)'] = (-sp.exp(a/2) - sp.exp(-a/2),
                        -2*sp.cosh(a/2),
                        -sp.exp(a/2) - sp.exp(-a/2))
    
    tests['sinh(a/2)'] = (sp.exp(a/2) - sp.exp(-a/2),
                        2*sp.sinh(a/2),
                        sp.exp(a/2) - sp.exp(-a/2))

    tests['sinh(-a/2)'] = (sp.exp(-a/2) - sp.exp(a/2),
                        -2*sp.sinh(a/2),
                        sp.exp(-a/2) - sp.exp(a/2))
    
    tests['-sinh(a/2)'] = (-sp.exp(a/2) + sp.exp(-a/2),
                        -2*sp.sinh(a/2),
                        -sp.exp(a/2) + sp.exp(-a/2))
    
    tests['cosh(a)/2*cosh(b)/2'] = ( ((sp.exp(a)/2) + (sp.exp(-a)/2))*((sp.exp(b)/2) + (sp.exp(-b)/2)),
                               sp.cosh(a) * sp.cosh(b),
                               (1/4) * (sp.exp(a+b) + (sp.exp(a-b)) + (sp.exp(-a+b)) + (sp.exp(-a-b))) )

    tests['sinh(a)/2*sinh(b)/2'] = ( ((sp.exp(a)/2)-(sp.exp(-a)/2))*((sp.exp(b)/2)-(sp.exp(-b)/2)),
                               sp.sinh(a) * sp.sinh(b),
                               (1/4) * (sp.exp(a+b) - (sp.exp(a-b)) - (sp.exp(-a+b)) + (sp.exp(-a-b))) )
    
    tests['cosh(a)*sinh(b)*cosh(c)'] = ( ((sp.exp(a)+sp.exp(-a))/2) * ((sp.exp(b)-sp.exp(-b))/2) * ((sp.exp(c)+sp.exp(-c))/2),
                               sp.cosh(a) * sp.sinh(b) * sp.cosh(c),
                               (1/8) * (sp.exp(a+b+c) + sp.exp(a+b-c) - sp.exp(a-b+c) - sp.exp(a-b-c) + sp.exp(-a+b+c) + sp.exp(-a+b-c) - sp.exp(-a-b+c) - sp.exp(-a-b-c)) )
    
    for tst in tests:
        hiperb, expsums = processor.unit_test(tst, tests[tst][0], RUN_UNIT_TESTS_VERBOSE) 
        try:
            if sp.simplify(hiperb - tests[tst][1]) == 0 and sp.simplify(expsums - tests[tst][2])==0:
                print(f"Test {tst} - OK")
            else:
                print(f"Test {tst}- NOK")
        except Exception as e:
            print(f"An error occurred: {e}")

