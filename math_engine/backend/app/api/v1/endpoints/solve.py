"""
Equation solving endpoints.
"""
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import SolveRequest, SolveResponse
from app.services.solver import MathSolver
from app.services.llm import LLMExplanationService
from app.core.exceptions import SolverError, LLMError

router = APIRouter()
solver_service = MathSolver()
llm_service = LLMExplanationService()


@router.post("/", response_model=SolveResponse)
async def solve_expression(request: SolveRequest) -> SolveResponse:
    """
    Solve a mathematical equation or simplify an expression.
    
    This endpoint can:
    - Solve equations for specific variables
    - Simplify mathematical expressions
    - Provide step-by-step solutions
    - Generate AI-powered explanations
    - Compute numerical approximations
    
    **Example request (equation solving):**
    ```json
    {
        "expression": "x^2 + 2*x + 1 = 0",
        "variable": "x",
        "steps": true,
        "explanation": true,
        "numerical": false
    }
    ```
    
    **Example request (expression simplification):**
    ```json
    {
        "expression": "(x+1)^2",
        "simplify_only": true,
        "steps": true,
        "explanation": false
    }
    ```
    """
    try:
        # Solve the expression
        result = solver_service.solve_equation(
            expression=request.expression,
            variable=request.variable,
            include_steps=request.steps,
            numerical=request.numerical,
            precision=request.precision or 10
        )
        
        explanations = None
        if request.explanation and llm_service.is_enabled():
            try:
                explanations = llm_service.generate_step_explanations(
                    original_expression=request.expression,
                    steps=result.steps,
                    final_result=result.final_result
                )
            except LLMError as e:
                # Don't fail the entire request if explanation generation fails
                explanations = [f"Explanation generation failed: {str(e)}"]
        
        return SolveResponse(
            success=True,
            result=result,
            explanations=explanations,
            warnings=[]
        )
        
    except SolverError as e:
        return SolveResponse(
            success=False,
            error_message=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during solving: {str(e)}"
        )


@router.post("/batch")
async def solve_batch(expressions: list[SolveRequest]) -> JSONResponse:
    """
    Solve multiple expressions in a batch request.
    
    **Example request:**
    ```json
    [
        {
            "expression": "x^2 - 4 = 0",
            "variable": "x",
            "steps": false
        },
        {
            "expression": "sin(x) = 0.5",
            "variable": "x",
            "numerical": true
        }
    ]
    ```
    """
    if len(expressions) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 10 expressions"
        )
    
    results = []
    for i, request in enumerate(expressions):
        try:
            result = solver_service.solve_equation(
                expression=request.expression,
                variable=request.variable,
                include_steps=request.steps,
                numerical=request.numerical,
                precision=request.precision or 10
            )
            
            results.append({
                "index": i,
                "success": True,
                "result": result.dict()
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


@router.post("/complete")
async def generate_complete_solution(request: SolveRequest) -> JSONResponse:
    """
    Generate a complete step-by-step solution with AI explanations.
    
    This endpoint provides the most comprehensive solution format including:
    - Step-by-step mathematical solution
    - AI-generated explanations for each step
    - Key mathematical concepts identified
    - Educational context and reasoning
    
    **Example request:**
    ```json
    {
        "expression": "2x + 5 = 3x - 7",
        "variable": "x",
        "steps": true,
        "explanation": true,
        "numerical": false
    }
    ```
    """
    if not llm_service.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM explanation service is not enabled or configured. Please set up your API key."
        )
    
    try:
        # First solve to get the mathematical steps
        result = solver_service.solve_equation(
            expression=request.expression,
            variable=request.variable,
            include_steps=True,
            numerical=request.numerical,
            precision=request.precision or 10
        )
        
        # Generate comprehensive AI explanation
        complete_solution = llm_service.generate_complete_solution(
            original_expression=request.expression,
            final_result=result.final_result,
            steps=result.steps,
            context={
                "difficulty_level": "intermediate",
                "audience": "student",
                "include_reasoning": True
            }
        )
        
        return JSONResponse(content={
            "success": True,
            "original_expression": request.expression,
            "final_result": result.final_result,
            "solution_steps": [step.dict() for step in result.steps],
            "overall_explanation": complete_solution["overall_explanation"],
            "step_explanations": complete_solution["step_explanations"],
            "key_concepts": complete_solution["key_concepts"],
            "execution_time_ms": result.execution_time_ms,
            "is_equation": "=" in request.expression,
            "variable_solved": request.variable
        })
        
    except SolverError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_message": f"Solver error: {str(e)}",
                "error_type": "solver_error"
            }
        )
    except LLMError as e:
        # Return mathematical solution even if LLM fails
        return JSONResponse(content={
            "success": True,
            "original_expression": request.expression,
            "final_result": result.final_result,
            "solution_steps": [step.dict() for step in result.steps],
            "overall_explanation": f"Mathematical solution completed, but explanation generation failed: {str(e)}",
            "step_explanations": [],
            "key_concepts": [],
            "execution_time_ms": result.execution_time_ms,
            "warning": "LLM explanations failed but mathematical solution succeeded"
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate complete solution: {str(e)}"
        )


@router.post("/explain")
async def explain_solution(request: SolveRequest) -> JSONResponse:
    """
    Get a detailed explanation of how to solve an expression.
    
    This endpoint focuses on generating comprehensive explanations
    rather than just the solution steps.
    """
    if not llm_service.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM explanation service is not enabled or configured"
        )
    
    try:
        # First solve to get the steps
        result = solver_service.solve_equation(
            expression=request.expression,
            variable=request.variable,
            include_steps=True,
            numerical=request.numerical,
            precision=request.precision or 10
        )
        
        # Generate comprehensive explanation
        overall_explanation = llm_service.generate_overall_explanation(
            original_expression=request.expression,
            final_result=result.final_result,
            steps=result.steps,
            context={
                "difficulty_level": "intermediate",
                "audience": "student"
            }
        )
        
        step_explanations = llm_service.generate_step_explanations(
            original_expression=request.expression,
            steps=result.steps,
            final_result=result.final_result,
            context={
                "difficulty_level": "intermediate",
                "audience": "student"
            }
        )
        
        return JSONResponse(content={
            "success": True,
            "original_expression": request.expression,
            "final_result": result.final_result,
            "overall_explanation": overall_explanation,
            "step_explanations": step_explanations,
            "steps": [step.dict() for step in result.steps],
            "execution_time_ms": result.execution_time_ms
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate explanation: {str(e)}"
        )
