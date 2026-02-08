"""Async Anthropic client with refinement capabilities."""

import asyncio
import time
from typing import List, Optional, Tuple

import numpy as np
from anthropic import AsyncAnthropic

from models import (
    Formulation,
    FormulationFeedback,
    FormulationSchema,
    RefinedFormulation,
    SingleFormulationFeedback,
    SolverResult,
)
from prompts import (
    FORMULATION_EVALUATION_PROMPT,
    FORMULATION_GENERATION_PROMPT,
    FORMULATION_REFINEMENT_PROMPT,
)


class RefinementClient:
    """Async Anthropic client with self-refinement capabilities."""

    def __init__(self, api_key: str, max_rpm: int = 50):
        self.client = AsyncAnthropic(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_rpm)
        self.model = "claude-sonnet-4-5"
        self.request_times: List[float] = []

    async def _rate_limit(self):
        """Rate limit requests to max_rpm per minute."""
        current_time = time.time()
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        if len(self.request_times) >= 50:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        self.request_times.append(current_time)

    async def generate_formulation(self, nl_description: str, temperature: float = 0.7) -> Optional[Formulation]:
        """Generate a single optimization formulation from natural language."""
        async with self.semaphore:
            await self._rate_limit()
            try:
                prompt = FORMULATION_GENERATION_PROMPT.format(nl_description=nl_description)
                
                response = await self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=8192,
                    temperature=temperature,
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": prompt}],
                    output_format=FormulationSchema,
                )
                
                if response.parsed_output:
                    return Formulation.from_pydantic(response.parsed_output, raw=str(response.content[0].text))
                return None
            except Exception as e:
                print(f"  ⚠️  Generation error at temp={temperature}: {e}")
                return None

    async def generate_multiple(self, nl_description: str, k: int = 5) -> List[Formulation]:
        """Generate multiple formulations with different temperatures."""
        temperatures = np.linspace(0.3, 1.0, k)
        tasks = [self.generate_formulation(nl_description, float(temp)) for temp in temperatures]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Formulation)]

    async def evaluate_single_formulation(
        self,
        nl_description: str,
        formulation: Formulation,
        solver_result: SolverResult,
        formulation_index: int,
        best_score_so_far: Optional[float] = None,
    ) -> FormulationFeedback:
        """Evaluate a single formulation with structured scores."""
        async with self.semaphore:
            await self._rate_limit()
            try:
                # Build compact formulation summary
                vars_str = ", ".join([f"{v['name']}({v['var_type']})" for v in formulation.variables[:10]])
                if len(formulation.variables) > 10:
                    vars_str += f"... ({len(formulation.variables)} total)"
                
                cons_str = "\n".join([f"  - {c['expression']}" for c in formulation.constraints[:8]])
                if len(formulation.constraints) > 8:
                    cons_str += f"\n  ... ({len(formulation.constraints)} total)"
                
                status_str = f"Status: {solver_result.status}"
                if solver_result.optimal_value is not None:
                    status_str += f", Optimal Value: {solver_result.optimal_value:.2f}"
                elif solver_result.status != "OPTIMAL":
                    status_str += f"\nSolver Message: {solver_result.solver_message}"
                
                best_context = ""
                if best_score_so_far is not None:
                    best_context = f"\n\nNote: Best score seen so far in this iteration is {best_score_so_far}/20. Use this as a reference point."
                
                prompt = FORMULATION_EVALUATION_PROMPT.format(
                    nl_description=nl_description,
                    status_str=status_str,
                    objective_sense=formulation.objective.get('sense', 'Not provided'),
                    objective_expression=formulation.objective.get('expression', 'Not provided'),
                    objective_description=formulation.objective.get('description', 'Not provided'),
                    num_variables=len(formulation.variables),
                    vars_str=vars_str,
                    num_constraints=len(formulation.constraints),
                    cons_str=cons_str,
                    best_context=best_context,
                )

                response = await self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=2048,
                    temperature=0.0,
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": prompt}],
                    output_format=SingleFormulationFeedback,
                )
                
                if response.parsed_output:
                    # Convert to FormulationFeedback with the correct index
                    single_fb = response.parsed_output
                    return FormulationFeedback(
                        formulation_index=formulation_index,
                        completeness_score=single_fb.completeness_score,
                        constraint_validity_score=single_fb.constraint_validity_score,
                        objective_alignment_score=single_fb.objective_alignment_score,
                        variable_definition_score=single_fb.variable_definition_score,
                        completeness_explanation=single_fb.completeness_explanation,
                        constraint_validity_explanation=single_fb.constraint_validity_explanation,
                        objective_alignment_explanation=single_fb.objective_alignment_explanation,
                        variable_definition_explanation=single_fb.variable_definition_explanation,
                        total_score=single_fb.total_score,
                        improvement_suggestions=single_fb.improvement_suggestions,
                    )
                else:
                    raise ValueError("No parsed output from evaluation")
            except Exception as e:
                print(f"  ❌ ERROR: Failed to evaluate formulation {formulation_index}: {e}")
                raise

    async def evaluate_formulations(
        self,
        nl_description: str,
        formulations: List[Formulation],
        solver_results: List[SolverResult],
    ) -> List[FormulationFeedback]:
        """Evaluate formulations in parallel with structured scores."""
        # Create parallel evaluation tasks
        tasks = [
            self.evaluate_single_formulation(
                nl_description,
                formulation,
                solver_result,
                i,
                best_score_so_far=None,
            )
            for i, (formulation, solver_result) in enumerate(zip(formulations, solver_results))
        ]
        
        # Execute all evaluations in parallel
        feedbacks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any exceptions and log them
        valid_feedbacks = []
        for i, fb in enumerate(feedbacks):
            if isinstance(fb, Exception):
                print(f"  ⚠️  Formulation {i} evaluation failed: {fb}")
            else:
                valid_feedbacks.append(fb)
        
        if not valid_feedbacks:
            raise RuntimeError("All formulation evaluations failed")
        
        return valid_feedbacks

    async def refine_formulation(
        self,
        nl_description: str,
        original_formulation: Formulation,
        feedback: FormulationFeedback,
        solver_result: SolverResult,
        best_formulation_summary: Optional[str] = None,
    ) -> Tuple[Formulation, str]:
        """Refine a formulation based on structured feedback."""
        async with self.semaphore:
            await self._rate_limit()
            try:
                original_schema = original_formulation.to_pydantic()
                
                # Build comparison text if available
                comparison_text = ""
                if best_formulation_summary:
                    comparison_text = f"\n\nFor comparison, here's a higher-scoring formulation:\n{best_formulation_summary}\n"
                
                variables_str = "\n".join([f"  - {v.name} ({v.var_type}): {v.description}" for v in original_schema.variables])
                constraints_str = "\n".join([f"  - {c.expression} | {c.description}" for c in original_schema.constraints])
                
                solver_details = ""
                if solver_result.optimal_value:
                    solver_details = f"Optimal Value: {solver_result.optimal_value}"
                else:
                    solver_details = f"Message: {solver_result.solver_message}"
                
                prompt = FORMULATION_REFINEMENT_PROMPT.format(
                    nl_description=nl_description,
                    objective_sense=original_schema.objective.sense,
                    objective_expression=original_schema.objective.expression,
                    objective_description=original_schema.objective.description,
                    num_variables=len(original_schema.variables),
                    variables_str=variables_str,
                    num_constraints=len(original_schema.constraints),
                    constraints_str=constraints_str,
                    solver_status=solver_result.status,
                    solver_details=solver_details,
                    total_score=feedback.total_score,
                    completeness_score=feedback.completeness_score,
                    completeness_explanation=feedback.completeness_explanation,
                    constraint_validity_score=feedback.constraint_validity_score,
                    constraint_validity_explanation=feedback.constraint_validity_explanation,
                    objective_alignment_score=feedback.objective_alignment_score,
                    objective_alignment_explanation=feedback.objective_alignment_explanation,
                    variable_definition_score=feedback.variable_definition_score,
                    variable_definition_explanation=feedback.variable_definition_explanation,
                    improvement_suggestions=feedback.improvement_suggestions,
                    comparison_text=comparison_text,
                )

                response = await self.client.beta.messages.parse(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.0,
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": prompt}],
                    output_format=RefinedFormulation,
                )
                
                if response.parsed_output:
                    refined = Formulation.from_pydantic(
                        response.parsed_output.refined,
                        raw=f"Refined based on feedback (score: {feedback.total_score}/20)"
                    )
                    return refined, response.parsed_output.explanation
                else:
                    # Fallback
                    return original_formulation, "Failed to parse refinement response - kept original"
                    
            except Exception as e:
                print(f"  ❌ ERROR: Failed to refine formulation {feedback.formulation_index}: {e}")
                print(f"     Keeping original formulation for this candidate")
                raise
