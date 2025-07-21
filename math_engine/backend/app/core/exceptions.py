"""
Custom exceptions for the math engine.
"""


class MathEngineException(Exception):
    """Base exception for math engine errors."""
    pass


class ParseError(MathEngineException):
    """Exception raised when expression parsing fails."""
    pass


class SolverError(MathEngineException):
    """Exception raised when equation solving fails."""
    pass


class SimplificationError(MathEngineException):
    """Exception raised when expression simplification fails."""
    pass


class VisualizationError(MathEngineException):
    """Exception raised when visualization generation fails."""
    pass


class LLMError(MathEngineException):
    """Exception raised when LLM explanation generation fails."""
    pass
