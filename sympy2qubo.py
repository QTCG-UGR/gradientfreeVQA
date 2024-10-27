from helpers import shorten_str, qcprint
from helpers import LOG_LEVEL_DEBUG, LOG_LEVEL_INFO
from config import RUN_UNIT_TESTS, UNIT_TST_VERBOSE
import sympy as sp
import dimod

class SympyToQUBOConverter:
    
    def sympy_to_qubo2(self, sympy_expr, binary_vars, bpm):
        simplified_expr = sympy_expr.expand()
        
        # Function to remove **2, **3 etc. since binary variables satisfy x**2 = x
        def remove_powers(expr):
            return expr.replace(lambda x: x.is_Pow and x.exp > 1, lambda x: x.base)
        
        simplified_expr = remove_powers(simplified_expr)
        
        # Iterate over terms in the expression
        for variables, coeff in simplified_expr.items():
            degree = len(variables)
    
            if degree == 0:
                # Offset term
                bpm.set_objective_offset(coeff)
            elif degree == 1:
                # Linear term
                var = variables[0]
                if var not in bpm.variables:
                    bpm.add_variable('BINARY', var)
                bpm.set_linear(var, coeff)
            elif degree == 2:
                # Quadratic term
                var1, var2 = variables
                if var1 not in bpm.variables:
                    bpm.add_variable('BINARY', var1)
                if var2 not in bpm.variables:
                    bpm.add_variable('BINARY', var2)
                bpm.set_quadratic(var1, var2, coeff)
            else:
                # Higher-order term
                poly = dimod.BinaryPolynomial({variables: coeff}, vartype='BINARY')
                bpm.set_objective(poly)  # Add higher-order polynomial to the objective
        
        return bpm
    
    
    def sympy_to_qubo(self, sympy_expr, binary_vars, bqm):
        qcprint(LOG_LEVEL_DEBUG, "Starting SymPy to QUBO conversion")
        
        # Recursive function to process SymPy expressions
        def process_expression(expr):
            qcprint(LOG_LEVEL_DEBUG, f"\nProcessing expr: {expr}")

            if isinstance(expr, sp.Add):
                for term in expr.args:
                    process_expression(term)
                    
            elif isinstance(expr, sp.Mul):
                coeff, factors = expr.as_coeff_mul()                
                binary_factors = [str(f) for f in factors if str(f) in binary_vars] # Filter only the binary variables (ignore non-binary constants)
                
                if len(binary_factors) == 1:
                    if (expr.is_Mul) and len(expr.as_coeff_Mul())>0 and expr.as_coeff_Mul()[0].is_number:
                        coeff = sp.Float(expr.as_coeff_Mul()[0]) # EA properly handling float coefficients
                    sympy_var = binary_factors[0]
                    bqm.add_variable(binary_vars[str(sympy_var)], float(coeff))
                    qcprint(LOG_LEVEL_DEBUG, f"  adding variable:{binary_vars[str(sympy_var)]} with value:{float(coeff)}")
                    qcprint(LOG_LEVEL_DEBUG, f"  bqm.linear: {bqm.linear}")
                
                elif len(binary_factors) == 2:
                    if (expr.is_Mul) and len(expr.as_coeff_Mul())>0 and expr.as_coeff_Mul()[0].is_number:
                        coeff = sp.Float(expr.as_coeff_Mul()[0]) # EA trying to properly handle float coefficients
                    sympy_var1, sympy_var2 = binary_factors
                    bqm.add_interaction(binary_vars[str(sympy_var1)], binary_vars[str(sympy_var2)], float(coeff))
                    qcprint(LOG_LEVEL_DEBUG, f"  adding variables:{binary_vars[sympy_var1]} and {binary_vars[sympy_var2]} with value:{float(coeff)}")
                    qcprint(LOG_LEVEL_DEBUG, f"  bqm.quadratic: {bqm.quadratic}")
                
                elif isinstance(expr, sp.Mul) and expr.args and len(expr.args)>1 and isinstance(expr.args[1], sp.Pow): # Handle powers with coefficients different than 1
                    coef, *binvarnexp = expr.args
                    base, exp = binvarnexp[0].args # EA  TODO: verify binvarnexp length with longer expressions
                    if exp >= 2:
                        if str(base) in binary_vars:
                            bqm.add_variable(binary_vars[str(base)], float(coef))
                            qcprint(LOG_LEVEL_DEBUG, f"  adding variable:{binary_vars[str(base)]} with value:{float(coeff)}")
                            qcprint(LOG_LEVEL_DEBUG, f"  bqm.linear: {bqm.linear}")
                        else:
                            process_expression(sp.expand(expr))
                    else:
                        process_expression(base)       
                        
                else:
                    # Recursively process complex factors
                    for factor in factors:
                        process_expression(factor)

            elif isinstance(expr, sp.Pow):
                base, exp = expr.args
                if exp == 2:
                    if str(base) in binary_vars:
                        bqm.add_variable(binary_vars[str(base)], float(1))
                        qcprint(LOG_LEVEL_DEBUG, f"  adding variable:{binary_vars[str(base)]} with value:{float(1)}")
                        qcprint(LOG_LEVEL_DEBUG, f"  bqm.linear: {bqm.linear}")
                    elif (isinstance(base, sp.Add) or isinstance(base, sp.Mul)):
                        expanded_expr = sp.expand(expr)
                        expanded_expr = expanded_expr.replace(lambda x: isinstance(x, sp.Pow) and x.exp == 2, lambda x: x.base)
                        process_expression(expanded_expr)
                else:
                    process_expression(base) # Handle non-quadratic powers  as power to 1
                    
            elif isinstance(expr, sp.Symbol):
                process_expression(sp.Mul(1, expr, evaluate=False))
                
            else:
                # Handle constant terms
                if isinstance(expr, float) or expr.is_number:
                    bqm.offset += float(expr)
                    qcprint(LOG_LEVEL_DEBUG, f"  adding offset:{float(expr)}")
                    qcprint(LOG_LEVEL_DEBUG, f"  bqm.linear: {bqm.linear}")
                else:
                    print(f"FLOAT IS NOT SP FLOAT: {expr}")

        process_expression(sympy_expr)
        
        qcprint(LOG_LEVEL_DEBUG, "Finished SymPy to QUBO conversion")

        return bqm

    # Creates SymPy expression out of a String representation
    def create_expression(self, expr_str, num_angles, num_parts):
        qcprint(LOG_LEVEL_DEBUG, "Starting Creating SymPy expression")

        # Step 1: Create symbolic variables dynamically
        binary_vars = {}
        
        for angle in range(num_angles):
            for var in range(num_parts):
                var_name = f'b{chr(97 + angle)}{var}'  # 'a', 'b', 'c', ...
                binary_vars[var_name] = sp.symbols(var_name)
    
        # Step 2: Replace variables in the expression string with the created binary variables
        for var_name, sym_var in binary_vars.items():
            expr_str = expr_str.replace(var_name, f"sp.symbols('{var_name}')")
    
        # Step 3: Create the SymPy expression from the modified string
        expr = eval(expr_str)
    
        qcprint(LOG_LEVEL_DEBUG, "Finished Creating SymPy expression")
        
        return expr, binary_vars


    def unit_test(self, tst_name, expr, num_angles, num_parts, verbose=True):
        if verbose: 
            print(f"---- Test {tst_name} info -----")
            print(f"Original expression : {shorten_str(expr)}")
            print(f"Original type  : {type(expr)}")
        
        expression, binary_vars = self.create_expression(expr, num_angles, num_parts)
        
        if verbose:
            print(f"Converted expression: {shorten_str(expression)}")
            print(f"Converted type: {type(expression)}")
            print(f"binary_vars: {binary_vars}")
        
        # Convert the SymPy expression to QUBO
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        bqm = self.sympy_to_qubo(expression, binary_vars, bqm)
        
        if verbose:
            print(f"QUBO expression: {shorten_str(bqm)}")
            
            # Check results
            print("Linear terms :", bqm.linear)
            print("Quadratic terms:", bqm.quadratic)
            print("Offset         :", bqm.offset)
            print("--------------------")
        
        return bqm, binary_vars
       
# Run unit test
if (RUN_UNIT_TESTS):
    t_converer = SympyToQUBOConverter()
    
    TST_PARTS = 5
    # Test 1 - One binary variable with coefficient
    expr = "0.620139779296071*ba2"
    t_bqm, t_bvars = t_converer.unit_test('1', expr, 1, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.linear[t_bvars['ba2']] == 0.620139779296071):
            print("Test 1 - OK")
        else:
            print("Test 1 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 1.1 - One binary variable without coefficient
    expr = "ba2"
    t_bqm, t_bvars = t_converer.unit_test('1.1', expr, 1, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.linear[t_bvars['ba2']] == 1):
            print("Test 1.1 - OK")
        else:
            print("Test 1.1 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 2 - Two binary variables with coefficients
    expr = "-1.75727192077171*ba0 - 2.60338035395396*ba1" 
    t_bqm, t_bvars = t_converer.unit_test('2', expr, 1, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.linear[t_bvars['ba0']] == -1.75727192077171) and \
            (t_bqm.linear[t_bvars['ba1']] == -2.60338035395396):
            print("Test 2 - OK")
        else:
            print("Test 2 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 3 - One interaction of two binary variables with one coefficient
    expr = "(0.620139779296071*ba2)*bd0" 
    t_bqm, t_bvars = t_converer.unit_test('3', expr, 4, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.quadratic.get((t_bvars['bd0'], t_bvars['ba2'])) == 0.620139779296071):
            print("Test 3 - OK")
        else:
            print("Test 3 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 4 - Power expression
    expr = "(0.620139779296071*ba2)**2"
    t_bqm, t_bvars = t_converer.unit_test('4', expr, 1, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.linear[t_bvars['ba2']] == 0.38457334586537967):
            print("Test 4 - OK")
        else:
            print("Test 4 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # Test 5 - One interaction of two binary variables with no coefficients
    expr = "ba2*bb0" 
    t_bqm, t_bvars = t_converer.unit_test('5', expr, 4, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.quadratic.get((t_bvars['ba2'], t_bvars['bb0'])) == 1):
            print("Test 5 - OK")
        else:
            print("Test 5 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Test 6 - Complex case
    expr = "0.725069074608376*(0.106820778675001*ba0 + 0.337148230588305*ba1 + 0.620139779296071*ba2 + ba3)**2" 
    t_bqm, t_bvars = t_converer.unit_test('6', expr, 1, TST_PARTS, UNIT_TST_VERBOSE)
    try:
        if (t_bqm.linear[t_bvars['ba0']] == 0.00827353028679825) and \
            (t_bqm.linear[t_bvars['ba1']] == 0.08241782544368009) and \
            (t_bqm.linear[t_bvars['ba2']] == 0.27884224000565777) and \
            (t_bqm.linear[t_bvars['ba3']] == 0.725069074608376) and \
            (t_bqm.quadratic.get((t_bvars['ba0'], t_bvars['ba1'])) == 0.052225908320692266) and \
            (t_bqm.quadratic.get((t_bvars['ba2'], t_bvars['ba1'])) == 0.3031934765994219) and \
            (t_bqm.quadratic.get((t_bvars['ba2'], t_bvars['ba0'])) == 0.09606268199307107) and \
            (t_bqm.quadratic.get((t_bvars['ba3'], t_bvars['ba1'])) == 0.48891151111702735) and \
            (t_bqm.quadratic.get((t_bvars['ba3'], t_bvars['ba0'])) == 0.15490488628565824) and \
            (t_bqm.quadratic.get((t_bvars['ba3'], t_bvars['ba2'])) == 0.8992883518040895):
            print("Test 6 - OK")
        else:
            print("Test 6 - NOK")
    except Exception as e:
        print(f"An error occurred: {e}")
