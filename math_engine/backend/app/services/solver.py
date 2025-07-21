"""
Math solver service for equation solving and expression simplification.
"""
import time
from typing import List, Optional, Dict, Any, Tuple
import sympy as sp
from sympy import solve, simplify, expand, factor, collect, cancel, apart

from app.core.exceptions import SolverError, SimplificationError
from app.models.schemas import SolutionStep, SolutionResult
from app.services.parser import MathParser


class MathSolver:
    """Mathematical equation solver and expression simplifier."""
    
    def __init__(self):
        """Initialize the solver."""
        self.parser = MathParser()
        self.step_counter = 0
    
    def solve_equation(
        self,
        expression: str,
        variable: Optional[str] = None,
        include_steps: bool = True,
        numerical: bool = False,
        precision: int = 10
    ) -> SolutionResult:
        """
        Solve a mathematical equation or simplify an expression.
        
        Args:
            expression: The mathematical expression/equation to solve
            variable: Variable to solve for (if equation)
            include_steps: Whether to include step-by-step solution
            numerical: Whether to compute numerical approximations
            precision: Numerical precision for approximations
            
        Returns:
            SolutionResult with the complete solution
            
        Raises:
            SolverError: If solving fails
        """
        start_time = time.time()
        self.step_counter = 0
        steps = []
        
        try:
            # Check if it's an equation or expression
            if '=' in expression:
                # Handle equation
                result = self._solve_equation_string(expression, variable, steps, include_steps)
                # Parse left side for AST
                left_str = expression.split('=')[0].strip()
                expr, ast_tree = self.parser.parse_expression(left_str)
            else:
                # Handle expression simplification
                expr, ast_tree = self.parser.parse_expression(expression)
                
                if include_steps:
                    steps.append(self._create_step(
                        expression=str(expr),
                        description="Parsed input expression",
                        rule_applied="parsing"
                    ))
                
                result = self._simplify_expression(expr, steps, include_steps)
            
            # Apply numerical evaluation if requested
            if numerical and result:
                result = self._apply_numerical_evaluation(result, precision)
            
            execution_time = (time.time() - start_time) * 1000
            
            return SolutionResult(
                original_expression=expression,
                final_result=str(result),
                steps=steps if include_steps else [],
                execution_time_ms=execution_time,
                ast_tree=ast_tree,
                visualization_data=self._generate_visualization_data_from_string(expression, result)
            )
            
        except Exception as e:
            raise SolverError(f"Failed to solve expression '{expression}': {str(e)}")
    
    def _apply_numerical_evaluation(self, result: Any, precision: int) -> Any:
        """Apply numerical evaluation to results."""
        try:
            if isinstance(result, list):
                numerical_results = []
                for sol in result:
                    try:
                        if hasattr(sol, 'evalf'):
                            num_val = complex(sol.evalf(precision))
                            if abs(num_val.imag) < 1e-10:  # Essentially real
                                numerical_results.append(float(num_val.real))
                            else:
                                numerical_results.append(num_val)
                        else:
                            numerical_results.append(str(sol))
                    except:
                        numerical_results.append(str(sol))
                return numerical_results
            else:
                try:
                    if hasattr(result, 'evalf'):
                        num_val = complex(result.evalf(precision))
                        if abs(num_val.imag) < 1e-10:  # Essentially real
                            return float(num_val.real)
                        else:
                            return num_val
                    else:
                        return str(result)
                except:
                    return str(result)
        except Exception:
            return result  # Return original if numerical evaluation fails
    
    def _solve_equation_string(
        self,
        equation_str: str,
        variable: Optional[str],
        steps: List[SolutionStep],
        include_steps: bool
    ) -> Any:
        """Solve an equation given as a string."""
        try:
            # Split equation at '=' sign
            if '=' not in equation_str:
                raise SolverError("No equation found (missing '=' sign)")
            
            left_str, right_str = equation_str.split('=', 1)
            
            # Parse both sides
            left_expr, _ = self.parser.parse_expression(left_str.strip())
            right_expr, _ = self.parser.parse_expression(right_str.strip())
            
            # Create equation (left - right = 0)
            equation = sp.Eq(left_expr, right_expr)
            
            if include_steps:
                steps.append(self._create_step(
                    expression=f"{left_str.strip()} = {right_str.strip()}",
                    description="Original equation",
                    rule_applied="input"
                ))
                
                steps.append(self._create_step(
                    expression=str(equation),
                    description="Formed equation object",
                    rule_applied="equation_formation"
                ))
            
            # Determine variable to solve for
            if variable:
                solve_var = sp.Symbol(variable)
            else:
                # Auto-detect variable
                variables = equation.free_symbols
                if len(variables) == 1:
                    solve_var = list(variables)[0]
                elif len(variables) > 1:
                    # Use the first alphabetically
                    solve_var = sorted(variables, key=str)[0]
                else:
                    raise SolverError("No variables found in equation")
            
            if include_steps:
                steps.append(self._create_step(
                    expression=f"Solving for {solve_var}",
                    description=f"Identified variable to solve: {solve_var}",
                    rule_applied="variable_identification"
                ))
            
            # Solve the equation
            solutions = solve(equation, solve_var)
            
            if include_steps:
                steps.append(self._create_step(
                    expression=str(solutions),
                    description=f"Found {len(solutions)} solution(s)",
                    rule_applied="equation_solving"
                ))
            
            return solutions
            
        except Exception as e:
            raise SolverError(f"Equation solving failed: {str(e)}")
    
    def _generate_visualization_data_from_string(self, expression: str, result: Any) -> Dict[str, Any]:
        """Generate visualization data from string expression."""
        try:
            # Try to parse the expression for visualization
            if '=' in expression:
                left_str = expression.split('=')[0].strip()
                expr, _ = self.parser.parse_expression(left_str)
            else:
                expr, _ = self.parser.parse_expression(expression)
            
            return self._generate_visualization_data(expr, result)
        except Exception:
            return {"error": "Visualization data generation failed"}
    
    def _solve_equation(
        self,
        expr: sp.Basic,
        variable: Optional[str],
        steps: List[SolutionStep],
        include_steps: bool
    ) -> Any:
        """Solve an equation."""
        try:
            # Split equation at '=' sign
            expr_str = str(expr)
            if '=' not in expr_str:
                raise SolverError("No equation found (missing '=' sign)")
            
            left, right = expr_str.split('=', 1)
            left_expr = sp.parse_expr(left)
            right_expr = sp.parse_expr(right)
            
            # Create equation (left - right = 0)
            equation = sp.Eq(left_expr, right_expr)
            
            if include_steps:
                steps.append(self._create_step(
                    expression=str(equation),
                    description="Formed equation",
                    rule_applied="equation_formation"
                ))
            
            # Determine variable to solve for
            if variable:
                solve_var = sp.Symbol(variable)
            else:
                # Auto-detect variable
                variables = equation.free_symbols
                if len(variables) == 1:
                    solve_var = list(variables)[0]
                elif len(variables) > 1:
                    # Use the first alphabetically
                    solve_var = sorted(variables, key=str)[0]
                else:
                    raise SolverError("No variables found in equation")
            
            if include_steps:
                steps.append(self._create_step(
                    expression=f"Solving for {solve_var}",
                    description=f"Identified variable to solve: {solve_var}",
                    rule_applied="variable_identification"
                ))
            
            # Solve the equation
            solutions = solve(equation, solve_var)
            
            if include_steps:
                steps.append(self._create_step(
                    expression=str(solutions),
                    description=f"Found {len(solutions)} solution(s)",
                    rule_applied="equation_solving"
                ))
            
            return solutions
            
        except Exception as e:
            raise SolverError(f"Equation solving failed: {str(e)}")
    
    def _simplify_expression(
        self,
        expr: sp.Basic,
        steps: List[SolutionStep],
        include_steps: bool
    ) -> sp.Basic:
        """Simplify a mathematical expression with steps."""
        current_expr = expr
        
        # Step 1: Expand
        expanded = expand(current_expr)
        if expanded != current_expr and include_steps:
            steps.append(self._create_step(
                expression=str(expanded),
                description="Expanded the expression",
                rule_applied="expansion"
            ))
            current_expr = expanded
        
        # Step 2: Simplify
        simplified = simplify(current_expr)
        if simplified != current_expr and include_steps:
            steps.append(self._create_step(
                expression=str(simplified),
                description="Applied general simplification rules",
                rule_applied="simplification"
            ))
            current_expr = simplified
        
        # Step 3: Factor if possible
        try:
            factored = factor(current_expr)
            if factored != current_expr and include_steps:
                steps.append(self._create_step(
                    expression=str(factored),
                    description="Factored the expression",
                    rule_applied="factoring"
                ))
                current_expr = factored
        except:
            pass
        
        # Step 4: Cancel common factors in rational expressions
        try:
            cancelled = cancel(current_expr)
            if cancelled != current_expr and include_steps:
                steps.append(self._create_step(
                    expression=str(cancelled),
                    description="Cancelled common factors",
                    rule_applied="cancellation"
                ))
                current_expr = cancelled
        except:
            pass
        
        return current_expr
    
    def _create_step(
        self,
        expression: str,
        description: str,
        rule_applied: Optional[str] = None
    ) -> SolutionStep:
        """Create a solution step."""
        self.step_counter += 1
        return SolutionStep(
            step_number=self.step_counter,
            expression=expression,
            description=description,
            rule_applied=rule_applied
        )
    
    def _generate_visualization_data(self, expr: sp.Basic, result: Any) -> Dict[str, Any]:
        """Generate data for expression visualization."""
        try:
            return {
                "expression_tree": self._build_expression_tree(expr),
                "complexity_metrics": {
                    "node_count": len(list(sp.preorder_traversal(expr))),
                    "depth": self._calculate_depth(expr),
                    "operations": self._count_operations(expr),
                },
                "latex_representation": sp.latex(expr),
                "result_latex": sp.latex(result) if hasattr(result, '__iter__') == False else [sp.latex(r) for r in result],
            }
        except Exception:
            return {"error": "Visualization data generation failed"}
    
    def _build_expression_tree(self, expr: sp.Basic) -> Dict[str, Any]:
        """Build a tree structure for visualization."""
        def _build_node(node):
            if node.is_Atom:
                return {
                    "id": str(id(node)),
                    "label": str(node),
                    "type": "leaf",
                    "value": str(node),
                    "children": []
                }
            else:
                children = [_build_node(arg) for arg in node.args]
                return {
                    "id": str(id(node)),
                    "label": type(node).__name__,
                    "type": "operation",
                    "operator": type(node).__name__,
                    "children": children
                }
        
        return _build_node(expr)
    
    def _calculate_depth(self, expr: sp.Basic) -> int:
        """Calculate the depth of the expression tree."""
        if expr.is_Atom:
            return 1
        else:
            if not expr.args:
                return 1
            return 1 + max(self._calculate_depth(arg) for arg in expr.args)
    
    def _count_operations(self, expr: sp.Basic) -> Dict[str, int]:
        """Count different types of operations in the expression."""
        operations = {}
        
        def _count(node):
            if not node.is_Atom:
                op_type = type(node).__name__
                operations[op_type] = operations.get(op_type, 0) + 1
                for arg in node.args:
                    _count(arg)
        
        _count(expr)
        return operations
