"""
Expression parsing endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import ParseRequest, ParseResponse
from app.services.parser import MathParser
from app.core.exceptions import ParseError

router = APIRouter()
parser_service = MathParser()


@router.post("/", response_model=ParseResponse)
async def parse_expression(request: ParseRequest) -> ParseResponse:
    """
    Parse a mathematical expression into its components.
    
    This endpoint analyzes a mathematical expression and returns:
    - Parsed canonical form
    - Abstract Syntax Tree (AST)
    - Variables, functions, and constants found
    - Complexity score
    
    **Example request:**
    ```json
    {
        "expression": "x^2 + 2*x + 1",
        "format": "text",
        "generate_ast": true,
        "validate_syntax": true
    }
    ```
    """
    try:
        # Parse the expression
        sympy_expr, ast_tree = parser_service.parse_expression(
            request.expression, 
            request.format
        )
        
        # Extract components
        variables = parser_service.extract_variables(sympy_expr)
        functions = parser_service.extract_functions(sympy_expr)
        constants = parser_service.extract_constants(sympy_expr)
        
        return ParseResponse(
            success=True,
            parsed_expression=str(sympy_expr),
            ast_tree=ast_tree if request.generate_ast else None,
            variables=variables,
            functions=functions,
            constants=constants,
            complexity_score=ast_tree.get("complexity", 0.0) if ast_tree else None,
        )
        
    except ParseError as e:
        return ParseResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during parsing: {str(e)}"
        )


@router.post("/validate")
async def validate_expression(request: ParseRequest) -> JSONResponse:
    """
    Validate the syntax of a mathematical expression without full parsing.
    
    **Example request:**
    ```json
    {
        "expression": "sin(x) + cos(y)",
        "format": "text"
    }
    ```
    
    **Example response:**
    ```json
    {
        "valid": true,
        "message": "Expression is valid"
    }
    ```
    """
    try:
        is_valid, error_message = parser_service.validate_syntax(
            request.expression, 
            request.format
        )
        
        return JSONResponse(
            content={
                "valid": is_valid,
                "message": error_message if not is_valid else "Expression is valid"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "valid": False,
                "message": f"Validation failed: {str(e)}"
            }
        )
