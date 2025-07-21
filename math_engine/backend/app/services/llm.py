"""
LLM service for generating AI-powered explanations of mathematical steps.
"""
import os
from typing import List, Optional, Dict, Any
import openai
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import LLMError
from app.models.schemas import SolutionStep


class LLMExplanationService:
    """Service for generating AI explanations of mathematical solutions."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.client = None
        
        # Initialize based on provider
        if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
            self.client = OpenAI(
                api_key=settings.GEMINI_API_KEY,
                base_url=settings.GEMINI_BASE_URL
            )
        elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.model = settings.LLM_MODEL
        self.provider = settings.LLM_PROVIDER
        self.enabled = settings.ENABLE_LLM_EXPLANATIONS and self.client is not None
    
    def generate_step_explanations(
        self,
        original_expression: str,
        steps: List[SolutionStep],
        final_result: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate step-by-step explanations directly from the equation and solution.
        
        Args:
            original_expression: The original mathematical expression
            steps: List of solution steps (not used, kept for compatibility)
            final_result: Final result of the solution
            context: Additional context for explanation generation
            
        Returns:
            List of step explanations showing actual algebraic operations
            
        Raises:
            LLMError: If explanation generation fails
        """
        if not self.enabled:
            return []
        
        try:
            # Generate step explanations directly from equation and solution
            prompt = f"""Original equation: {original_expression}
Final solution: {final_result}

Generate a list of concise algebraic steps to solve this equation. Each step should show only the actual operation performed and the resulting equation.

Format each step as: "Operation description: resulting equation"

Examples:
- "Subtract 3 from both sides: 2x = 1"
- "Divide both sides by 2: x = 1/2"
- "Add 5 to both sides: x = 8"

Return only the step descriptions as a simple list, one per line. Do not include explanations or additional text."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematics tutor. Generate concise step-by-step algebraic operations. Each step should be one line showing the operation and result."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse the response into individual steps
            response_text = response.choices[0].message.content.strip()
            steps_list = [step.strip() for step in response_text.split('\n') if step.strip()]
            
            return steps_list
            
        except Exception as e:
            return [f"Could not generate step explanations: {str(e)}"]
    
    def generate_overall_explanation(
        self,
        original_expression: str,
        final_result: str,
        steps: List[SolutionStep],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an overall explanation of the entire solution process.
        
        Args:
            original_expression: The original mathematical expression
            final_result: Final result of the solution
            steps: List of all solution steps
            context: Additional context for explanation generation
            
        Returns:
            Overall explanation of the solution
            
        Raises:
            LLMError: If explanation generation fails
        """
        if not self.enabled:
            return "LLM explanations are not enabled or configured."
        
        try:
            prompt = self._build_overall_explanation_prompt(
                original_expression, final_result, steps, context
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert mathematics tutor who provides clear, step-by-step explanations. Your explanations should be educational, easy to follow, and help students understand both the process and the reasoning behind each step."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise LLMError(f"Failed to generate overall explanation: {str(e)}")
    
    def _generate_single_explanation(
        self,
        original_expression: str,
        current_step: SolutionStep,
        previous_steps: List[SolutionStep],
        final_result: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate explanation for a single step."""
        try:
            prompt = self._build_step_explanation_prompt(
                original_expression, current_step, previous_steps, context
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematics tutor. Explain the actual algebraic steps taken to solve equations. Be concise and focus on what operation was performed and why. Use simple HTML formatting. Example: 'Subtract 3 from both sides: <strong>2x = 1</strong>'"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=150,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Could not generate explanation for step {current_step.step_number}: {str(e)}"
    
    def _build_step_explanation_prompt(
        self,
        original_expression: str,
        current_step: SolutionStep,
        previous_steps: List[SolutionStep],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for single step explanation."""
        prompt_parts = [
            f"Original expression: {original_expression}"
        ]
        
        if previous_steps:
            prompt_parts.append("Previous steps:")
            for step in previous_steps:
                prompt_parts.append(f"Step {step.step_number}: {step.expression} - {step.description}")
        
        prompt_parts.append(f"Current step {current_step.step_number}: {current_step.expression}")
        
        if current_step.description:
            prompt_parts.append(f"Description: {current_step.description}")
        
        if current_step.rule_applied:
            prompt_parts.append(f"Rule applied: {current_step.rule_applied}")
        
        if context and context.get('difficulty_level'):
            difficulty_level = context.get('difficulty_level', 'intermediate')
            prompt_parts.append(f"Explanation difficulty level: {difficulty_level}")
        
        prompt_parts.append("If this step involves an actual algebraic operation (like adding, subtracting, multiplying, or dividing), explain what was done. Use this format: 'Operation: result'. For example: 'Subtract 3 from both sides: <strong>2x = 1</strong>'. If this step is just setup or identification (like 'Original equation' or 'Solving for x'), respond with 'Setup step - no operation performed.' Use simple HTML formatting.")
        
        return "\n".join(prompt_parts)
    
    def _build_overall_explanation_prompt(
        self,
        original_expression: str,
        final_result: str,
        steps: List[SolutionStep],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for overall explanation."""
        prompt_parts = [
            f"Original problem: {original_expression}",
            f"Final answer: {final_result}",
            "Solution process:"
        ]
        
        for step in steps:
            step_info = f"Step {step.step_number}: {step.expression}"
            if step.description:
                step_info += f" ({step.description})"
            if step.rule_applied:
                step_info += f" Rule: {step.rule_applied}"
            prompt_parts.append(step_info)
        
        if context:
            difficulty_level = context.get('difficulty_level', 'intermediate')
            audience = context.get('audience', 'student')
            prompt_parts.append(f"Explain this for a {audience} at {difficulty_level} level.")
        
        prompt_parts.append("Provide a concise step-by-step solution showing only the actual algebraic operations. Use this exact format for each operation:<br><strong>Operation description: resulting equation</strong><br>For example:<br><strong>Subtract 3 from both sides: 2x = 1</strong><br><strong>Divide both sides by 2: x = 1/2</strong><br>Keep it brief and practical.")
        
        return "\n".join(prompt_parts)
    
    def generate_complete_solution(
        self,
        original_expression: str,
        final_result: str,
        steps: List[SolutionStep],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete step-by-step solution with explanations.
        
        Args:
            original_expression: The original mathematical expression
            final_result: Final result of the solution
            steps: List of all solution steps
            context: Additional context for explanation generation
            
        Returns:
            Dictionary containing comprehensive solution explanation
            
        Raises:
            LLMError: If explanation generation fails
        """
        if not self.enabled:
            return {
                "overall_explanation": "LLM explanations are not enabled or configured.",
                "step_explanations": [],
                "key_concepts": []
            }
        
        try:
            # Generate overall explanation
            overall_explanation = self.generate_overall_explanation(
                original_expression, final_result, steps, context
            )
            
            # Generate step-by-step explanations
            step_explanations = self.generate_step_explanations(
                original_expression, steps, final_result, context
            )
            
            # Generate key concepts
            key_concepts = self._identify_key_concepts(
                original_expression, steps, context
            )
            
            return {
                "overall_explanation": overall_explanation,
                "step_explanations": step_explanations,
                "key_concepts": key_concepts
            }
            
        except Exception as e:
            raise LLMError(f"Failed to generate complete solution: {str(e)}")
    
    def _identify_key_concepts(
        self,
        original_expression: str,
        steps: List[SolutionStep],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Identify key mathematical concepts used in the solution."""
        try:
            prompt_parts = [
                f"Mathematical problem: {original_expression}",
                "",
                "Solution steps:"
            ]
            
            for step in steps:
                prompt_parts.append(f"Step {step.step_number}: {step.expression}")
                if step.rule_applied:
                    prompt_parts.append(f"  Rule: {step.rule_applied}")
            
            prompt_parts.extend([
                "",
                "List the key mathematical concepts, rules, or techniques used in this solution.",
                "Return as a comma-separated list (e.g., 'quadratic formula, factoring, square roots').",
                "Focus on the most important concepts a student should understand."
            ])
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematics curriculum expert identifying key concepts."
                    },
                    {
                        "role": "user",
                        "content": "\n".join(prompt_parts)
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            concepts_text = response.choices[0].message.content.strip()
            # Parse the comma-separated list
            concepts = [concept.strip() for concept in concepts_text.split(',')]
            return [concept for concept in concepts if concept]
            
        except Exception as e:
            return [f"Could not identify key concepts: {str(e)}"]
        """Check if LLM explanations are enabled and configured."""
        return self.enabled
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the LLM service."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "gemini_api_key_configured": bool(settings.GEMINI_API_KEY),
            "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
            "client_initialized": self.client is not None,
        }
