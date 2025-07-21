"""
Math parser service for converting expressions to AST and SymPy objects.
"""
import re
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from app.core.exceptions import ParseError
from app.models.schemas import ExpressionFormat


class MathParser:
    """Mathematical expression parser."""
    
    def __init__(self):
        """Initialize the parser with transformations."""
        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application,)
        )
        
        # Setup the global and local namespaces for parsing
        self._setup_namespaces()
    
    def _setup_namespaces(self):
        """Setup global and local namespaces for parsing."""
        # Mathematical functions and constants
        self.local_dict = {
            # Trigonometric functions
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'sec': sp.sec,
            'csc': sp.csc,
            'cot': sp.cot,
            # Inverse trigonometric functions
            'asin': sp.asin,
            'arcsin': sp.asin,
            'acos': sp.acos,
            'arccos': sp.acos,
            'atan': sp.atan,
            'arctan': sp.atan,
            'asec': sp.asec,
            'arcsec': sp.asec,
            'acsc': sp.acsc,
            'arccsc': sp.acsc,
            'acot': sp.acot,
            'arccot': sp.acot,
            # Hyperbolic functions
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'asinh': sp.asinh,
            'acosh': sp.acosh,
            'atanh': sp.atanh,
            # Logarithmic and exponential functions
            'log': sp.log,
            'ln': sp.log,
            'log10': lambda x: sp.log(x, 10),
            'exp': sp.exp,
            # Root and power functions
            'sqrt': sp.sqrt,
            'cbrt': lambda x: sp.Pow(x, sp.Rational(1, 3)),
            'pow': sp.Pow,
            # Other mathematical functions
            'abs': sp.Abs,
            'floor': sp.floor,
            'ceil': sp.ceiling,
            'ceiling': sp.ceiling,
            'round': lambda x: sp.floor(x + sp.Rational(1, 2)),
            'sign': sp.sign,
            'factorial': sp.factorial,
            'gamma': sp.gamma,
            # Mathematical constants
            'pi': sp.pi,
            'e': sp.E,
            'I': sp.I,
            'oo': sp.oo,
            'inf': sp.oo,
            'infinity': sp.oo,
        }
        
        # SymPy core classes for unevaluated expressions
        self.global_dict = {
            'Add': sp.Add,
            'Mul': sp.Mul,
            'Pow': sp.Pow,
            'Integer': sp.Integer,
            'Rational': sp.Rational,
            'Float': sp.Float,
            'Symbol': sp.Symbol,
            'Function': sp.Function,
            'Eq': sp.Eq,
            'Ne': sp.Ne,
            'Lt': sp.Lt,
            'Le': sp.Le,
            'Gt': sp.Gt,
            'Ge': sp.Ge,
        }
    
    def parse_expression(
        self,
        expression: str,
        format_type: ExpressionFormat = ExpressionFormat.TEXT
    ) -> Tuple[sp.Basic, Dict[str, Any]]:
        """
        Parse a mathematical expression into SymPy object and AST.
        
        Args:
            expression: The mathematical expression to parse
            format_type: The format of the input expression
            
        Returns:
            Tuple of (SymPy expression, AST dictionary)
            
        Raises:
            ParseError: If parsing fails
        """
        try:
            # Clean and preprocess the expression
            cleaned_expr = self._preprocess_expression(expression, format_type)
            
            # Parse based on format
            if format_type == ExpressionFormat.LATEX:
                sympy_expr = self._parse_latex(cleaned_expr)
            elif format_type == ExpressionFormat.MATHML:
                raise ParseError("MathML parsing not yet implemented")
            else:  # TEXT or SYMPY
                sympy_expr = self._parse_text(cleaned_expr)
            
            # Generate AST
            ast_tree = self._generate_ast(sympy_expr)
            
            return sympy_expr, ast_tree
            
        except Exception as e:
            raise ParseError(f"Failed to parse expression '{expression}': {str(e)}")
    
    def _preprocess_expression(self, expression: str, format_type: ExpressionFormat) -> str:
        """Preprocess expression for parsing."""
        # Remove extra whitespace
        expression = expression.strip()
        
        if format_type == ExpressionFormat.TEXT:
            # Replace common mathematical notation
            replacements = {
                '^': '**',  # Exponentiation
                '÷': '/',   # Division
                '×': '*',   # Multiplication
                '√': 'sqrt',  # Square root
                '∞': 'oo',  # Infinity
                'π': 'pi',  # Pi
            }
            
            for old, new in replacements.items():
                expression = expression.replace(old, new)
            
            # Handle implicit multiplication more carefully
            # Match digit followed by variable/function, but not function calls
            expression = re.sub(r'(\d+)([a-zA-Z][a-zA-Z0-9_]*(?!\())', r'\1*\2', expression)
            # Match variable followed by digit
            expression = re.sub(r'([a-zA-Z][a-zA-Z0-9_]*)(\d+)', r'\1*\2', expression)
            # Match closing parenthesis followed by variable/number
            expression = re.sub(r'\)([\da-zA-Z])', r')*\1', expression)
            # Match number/variable followed by opening parenthesis (but not function calls)
            expression = re.sub(r'(\d+)\s*\(', r'\1*(', expression)
        
        return expression
    
    def _parse_latex(self, expression: str) -> sp.Basic:
        """Parse LaTeX expression."""
        try:
            return parse_latex(expression)
        except Exception as e:
            raise ParseError(f"LaTeX parsing failed: {str(e)}")
    
    def _parse_text(self, expression: str) -> sp.Basic:
        """Parse text expression."""
        try:
            # Check if this is an equation (contains = sign)
            if '=' in expression and not any(op in expression for op in ['==', '!=', '<=', '>=']):
                # Handle equation: split on = and create an Eq object
                parts = expression.split('=')
                if len(parts) == 2:
                    left_expr = parse_expr(
                        parts[0].strip(),
                        transformations=self.transformations,
                        local_dict=self.local_dict,
                        global_dict=self.global_dict,
                        evaluate=True
                    )
                    right_expr = parse_expr(
                        parts[1].strip(),
                        transformations=self.transformations,
                        local_dict=self.local_dict,
                        global_dict=self.global_dict,
                        evaluate=True
                    )
                    return sp.Eq(left_expr, right_expr)
                else:
                    raise ParseError("Invalid equation: equations must have exactly one '=' sign")
            
            # Regular expression parsing
            result = parse_expr(
                expression,
                transformations=self.transformations,
                local_dict=self.local_dict,
                global_dict=self.global_dict,
                evaluate=True  # Changed to True for better compatibility
            )
            return result
        except Exception as e:
            # Fallback: try without evaluate=False
            try:
                result = parse_expr(
                    expression,
                    transformations=self.transformations,
                    local_dict=self.local_dict
                )
                return result
            except Exception as e2:
                raise ParseError(f"Text parsing failed: {str(e2)}")
    
    def _generate_ast(self, expr: sp.Basic) -> Dict[str, Any]:
        """Generate AST representation of SymPy expression."""
        def _convert_node(node):
            if node.is_Atom:
                return {
                    "type": "atom",
                    "value": str(node),
                    "sympy_type": type(node).__name__,
                    "is_number": node.is_number,
                    "is_symbol": node.is_symbol,
                }
            else:
                # Handle equations specially
                if isinstance(node, sp.Eq):
                    return {
                        "type": "equation",
                        "operator": "Eq",
                        "left": _convert_node(node.lhs),
                        "right": _convert_node(node.rhs),
                        "sympy_type": "Eq",
                    }
                else:
                    return {
                        "type": "operation",
                        "operator": type(node).__name__,
                        "args": [_convert_node(arg) for arg in node.args],
                        "sympy_type": type(node).__name__,
                    }
        
        return {
            "root": _convert_node(expr),
            "expression": str(expr),
            "latex": sp.latex(expr),
            "variables": [str(var) for var in expr.free_symbols],
            "functions": list(set(str(f.func) for f in expr.atoms(sp.Function))),
            "complexity": self._calculate_complexity(expr),
            "is_equation": isinstance(expr, sp.Eq),
        }
    
    def _calculate_complexity(self, expr: sp.Basic) -> float:
        """Calculate expression complexity score."""
        # Simple complexity metric based on node count and operation types
        complexity = 0.0
        
        def _count_complexity(node):
            nonlocal complexity
            
            if node.is_Atom:
                complexity += 0.1
            else:
                # Different operations have different complexity weights
                op_weights = {
                    'Add': 0.2,
                    'Mul': 0.3,
                    'Pow': 0.5,
                    'log': 0.7,
                    'sin': 0.6,
                    'cos': 0.6,
                    'tan': 0.6,
                    'exp': 0.8,
                    'Integral': 2.0,
                    'Derivative': 1.5,
                }
                
                op_name = type(node).__name__
                complexity += op_weights.get(op_name, 0.4)
                
                for arg in node.args:
                    _count_complexity(arg)
        
        _count_complexity(expr)
        return round(complexity, 2)
    
    def extract_variables(self, expr: sp.Basic) -> List[str]:
        """Extract all variables from expression."""
        return [str(var) for var in expr.free_symbols]
    
    def extract_functions(self, expr: sp.Basic) -> List[str]:
        """Extract all functions from expression."""
        functions = set()
        for atom in expr.atoms(sp.Function):
            functions.add(str(atom.func))
        return list(functions)
    
    def extract_constants(self, expr: sp.Basic) -> List[str]:
        """Extract all constants from expression."""
        constants = []
        for atom in expr.atoms():
            if atom.is_number and not atom.is_Symbol:
                constants.append(str(atom))
            elif atom in [sp.pi, sp.E, sp.I, sp.oo]:
                constants.append(str(atom))
        return list(set(constants))
    
    def validate_syntax(self, expression: str, format_type: ExpressionFormat) -> Tuple[bool, Optional[str]]:
        """
        Validate expression syntax without full parsing.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.parse_expression(expression, format_type)
            return True, None
        except ParseError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
