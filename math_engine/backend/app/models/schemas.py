"""
Pydantic models for API requests and responses.
"""
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class ExpressionFormat(str, Enum):
    """Supported expression formats."""
    LATEX = "latex"
    SYMPY = "sympy"
    MATHML = "mathml"
    TEXT = "text"


class SolutionStep(BaseModel):
    """Model for a single solution step."""
    step_number: int = Field(..., description="Step number in the solution process")
    expression: str = Field(..., description="Mathematical expression at this step")
    description: str = Field(..., description="Description of what was done in this step")
    rule_applied: Optional[str] = Field(None, description="Mathematical rule applied")
    ast_representation: Optional[Dict[str, Any]] = Field(None, description="AST representation")


class SolutionResult(BaseModel):
    """Model for complete solution result."""
    original_expression: str = Field(..., description="Original input expression")
    final_result: str = Field(..., description="Final simplified result")
    steps: List[SolutionStep] = Field(default_factory=list, description="Step-by-step solution")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    ast_tree: Optional[Dict[str, Any]] = Field(None, description="Abstract syntax tree")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Data for visualization")


class ParseRequest(BaseModel):
    """Request model for expression parsing."""
    expression: str = Field(..., min_length=1, max_length=1000, description="Mathematical expression to parse")
    format: ExpressionFormat = Field(ExpressionFormat.TEXT, description="Input format")
    generate_ast: bool = Field(True, description="Whether to generate AST")
    validate_syntax: bool = Field(True, description="Whether to validate syntax")

    @validator('expression')
    def validate_expression(cls, v):
        if not v.strip():
            raise ValueError('Expression cannot be empty or whitespace only')
        return v.strip()


class ParseResponse(BaseModel):
    """Response model for expression parsing."""
    success: bool = Field(..., description="Whether parsing was successful")
    parsed_expression: Optional[str] = Field(None, description="Parsed expression in canonical form")
    ast_tree: Optional[Dict[str, Any]] = Field(None, description="Abstract syntax tree")
    variables: List[str] = Field(default_factory=list, description="Variables found in expression")
    functions: List[str] = Field(default_factory=list, description="Functions found in expression")
    constants: List[str] = Field(default_factory=list, description="Constants found in expression")
    complexity_score: Optional[float] = Field(None, description="Expression complexity score")
    error_message: Optional[str] = Field(None, description="Error message if parsing failed")


class SolveRequest(BaseModel):
    """Request model for equation solving."""
    expression: str = Field(..., min_length=1, max_length=1000, description="Mathematical expression to solve")
    variable: Optional[str] = Field(None, description="Variable to solve for (if equation)")
    format: ExpressionFormat = Field(ExpressionFormat.TEXT, description="Input format")
    steps: bool = Field(True, description="Whether to include step-by-step solution")
    explanation: bool = Field(False, description="Whether to generate AI explanations")
    simplify_only: bool = Field(False, description="Whether to only simplify, not solve")
    numerical: bool = Field(False, description="Whether to compute numerical approximations")
    precision: Optional[int] = Field(10, ge=1, le=50, description="Numerical precision")

    @validator('expression')
    def validate_expression(cls, v):
        if not v.strip():
            raise ValueError('Expression cannot be empty or whitespace only')
        return v.strip()


class SolveResponse(BaseModel):
    """Response model for equation solving."""
    success: bool = Field(..., description="Whether solving was successful")
    result: Optional[SolutionResult] = Field(None, description="Solution result")
    explanations: Optional[List[str]] = Field(None, description="AI-generated explanations")
    error_message: Optional[str] = Field(None, description="Error message if solving failed")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during solving")


class VisualizationRequest(BaseModel):
    """Request model for expression visualization."""
    expression: str = Field(..., min_length=1, max_length=1000, description="Expression to visualize")
    visualization_type: str = Field("tree", description="Type of visualization (tree, graph, plot)")
    format: ExpressionFormat = Field(ExpressionFormat.TEXT, description="Input format")
    width: Optional[int] = Field(800, ge=100, le=2000, description="Visualization width")
    height: Optional[int] = Field(600, ge=100, le=2000, description="Visualization height")
    interactive: bool = Field(True, description="Whether to generate interactive visualization")


class VisualizationResponse(BaseModel):
    """Response model for expression visualization."""
    success: bool = Field(..., description="Whether visualization was successful")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")
    svg_content: Optional[str] = Field(None, description="SVG content for static visualization")
    d3_config: Optional[Dict[str, Any]] = Field(None, description="D3.js configuration")
    error_message: Optional[str] = Field(None, description="Error message if visualization failed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    status_code: int = Field(..., description="HTTP status code")
