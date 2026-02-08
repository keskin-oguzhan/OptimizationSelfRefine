"""Self-refinement engine for iteratively improving formulations."""

from typing import List, Optional

import numpy as np

from client import RefinementClient
from models import Formulation, RefinementIteration, RefinementResult, SolverResult
from solver import SolverVerifier


class SelfRefinementEngine:
    """Self-refinement engine for iteratively improving formulations.
    
    Follows the self-refine approach:
    1. Generate initial formulations
    2. Evaluate with structured scores
    3. Refine based on feedback
    4. Re-evaluate and check for convergence
    5. Repeat until quality plateaus or threshold met
    """

    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 3, score_threshold: float = 18.0):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

    def _count_unique_values(self, values: List[float]) -> int:
        """Count unique values within tolerance."""
        unique = []
        for v in values:
            if not any(abs(v - u) < self.tolerance for u in unique):
                unique.append(v)
        return len(unique)

    async def refine(
        self,
        nl: str,
        formulations: List[Formulation],
        solver_results: List[SolverResult],
        client: RefinementClient,
        verifier: SolverVerifier,
        ground_truth: Optional[float] = None,
    ) -> RefinementResult:
        """Run self-refinement on formulations."""
        
        # Check if we have any formulations to refine
        if len(formulations) == 0:
            return RefinementResult(
                attempted=False,
                reason="No formulations to refine"
            )
        
        # Get initial optimal values
        valid_results = [r for r in solver_results if r.status == "OPTIMAL"]
        initial_values = [r.optimal_value for r in valid_results if r.optimal_value is not None]
        initial_divergence = float(np.std(initial_values)) if len(initial_values) > 1 else 0.0
        initial_num_unique = self._count_unique_values(initial_values) if initial_values else 0
        
        # Calculate initial error metrics vs ground truth
        initial_best_error_pct = None
        initial_mean_error_pct = None
        if ground_truth is not None and ground_truth != 0 and initial_values:
            errors_pct = [abs(v - ground_truth) / abs(ground_truth) * 100 for v in initial_values]
            initial_best_error_pct = min(errors_pct)
            initial_mean_error_pct = float(np.mean(errors_pct))
        
        result = RefinementResult(
            attempted=True,
            initial_optimal_values=initial_values,
            initial_divergence=initial_divergence,
            initial_num_unique=initial_num_unique,
            initial_best_error_pct=initial_best_error_pct,
            initial_mean_error_pct=initial_mean_error_pct,
        )
        
        # Current state for iteration
        current_formulations = formulations
        current_results = solver_results
        previous_avg_score = 0.0
        
        for iter_num in range(self.max_iterations):
            print(f"  → Refinement iteration {iter_num + 1}/{self.max_iterations}")
            
            # Step 1: Evaluate current formulations with structured scores
            print(f"     - Evaluating {len(current_formulations)} formulations...")
            feedbacks = await client.evaluate_formulations(nl, current_formulations, current_results)
            print(f"     - Received {len(feedbacks)} feedbacks")
            
            if not feedbacks:
                print("     - No feedback generated, stopping")
                break
            
            # Calculate quality metrics
            optimal_feedbacks = [f for f in feedbacks if current_results[f.formulation_index].status == "OPTIMAL"]
            problematic_feedbacks = [f for f in feedbacks if current_results[f.formulation_index].status in ["ERROR", "UNBOUNDED", "INFEASIBLE"]]
            
            score_feedbacks = optimal_feedbacks if optimal_feedbacks else feedbacks
            
            if not score_feedbacks:
                print("     - No formulations to evaluate, stopping")
                break
            
            avg_score = float(np.mean([f.total_score for f in score_feedbacks]))
            max_score = max([f.total_score for f in score_feedbacks])
            min_score = min([f.total_score for f in score_feedbacks])
            
            print(f"     - Quality scores: min={min_score}/20, avg={avg_score:.1f}/20, max={max_score}/20")
            print(f"     - Formulations: {len(optimal_feedbacks)} optimal, {len(problematic_feedbacks)} problematic")
            
            # Store initial avg score
            if iter_num == 0:
                result.initial_avg_score = avg_score
            
            # Step 2: Check convergence criteria
            current_values = [r.optimal_value for r in current_results if r.status == "OPTIMAL" and r.optimal_value is not None]
            current_divergence = float(np.std(current_values)) if len(current_values) > 1 else 0.0
            current_num_unique = self._count_unique_values(current_values) if current_values else 0
            num_problematic = len([r for r in current_results if r.status in ["ERROR", "UNBOUNDED", "INFEASIBLE"]])
            
            min_score = min([f.total_score for f in score_feedbacks]) if score_feedbacks else 0
            all_meet_threshold = all(f.total_score >= self.score_threshold for f in score_feedbacks)
            values_aligned = current_num_unique == 1 if current_values else False
            no_errors = num_problematic == 0
            
            converged = all_meet_threshold and (values_aligned or no_errors)
            
            # Record this iteration
            iteration = RefinementIteration(
                iteration_number=iter_num + 1,
                formulations=[f.to_dict() for f in current_formulations],
                feedbacks=[{
                    "formulation_index": fb.formulation_index,
                    "scores": {
                        "completeness": fb.completeness_score,
                        "constraint_validity": fb.constraint_validity_score,
                        "objective_alignment": fb.objective_alignment_score,
                        "variable_definition": fb.variable_definition_score,
                        "total": fb.total_score,
                    },
                    "explanations": {
                        "completeness": fb.completeness_explanation,
                        "constraint_validity": fb.constraint_validity_explanation,
                        "objective_alignment": fb.objective_alignment_explanation,
                        "variable_definition": fb.variable_definition_explanation,
                    },
                    "improvement_suggestions": fb.improvement_suggestions,
                } for fb in feedbacks],
                solver_results=[r.to_dict() for r in current_results],
                optimal_values=current_values,
                average_score=avg_score,
                min_score=min_score,
                max_score=max_score,
                all_meet_threshold=all_meet_threshold,
                value_divergence=current_divergence,
                num_unique_values=current_num_unique,
                converged=converged,
            )
            result.iterations.append(iteration)
            
            # Check if we should stop
            if converged:
                print(f"     ✓ Convergence achieved!")
                result.converged = True
                break
            
            # Step 3: Refine formulations
            print("     - Refining formulations...")
            
            # Find best formulation for comparison
            feedbacks_for_best = optimal_feedbacks if optimal_feedbacks else feedbacks
            best_feedback = max(feedbacks_for_best, key=lambda f: f.total_score)
            best_formulation = current_formulations[best_feedback.formulation_index]
            best_summary = f"""Best formulation (score: {best_feedback.total_score}/20):
  Objective: {best_formulation.objective.get('sense', 'Not provided')} {best_formulation.objective.get('expression', 'Not provided')}
  Variables: {len(best_formulation.variables)}
  Constraints: {len(best_formulation.constraints)}"""
            
            # Refine each formulation (only those below threshold)
            refinement_tasks = []
            for fb in feedbacks:
                form_result = current_results[fb.formulation_index]
                should_refine = (
                    fb.total_score < self.score_threshold or 
                    form_result.status != "OPTIMAL"
                )
                
                if should_refine:
                    task = client.refine_formulation(
                        nl,
                        current_formulations[fb.formulation_index],
                        fb,
                        form_result,
                        best_summary if fb.formulation_index != best_feedback.formulation_index else None,
                    )
                    refinement_tasks.append((fb.formulation_index, task))
                else:
                    refinement_tasks.append((fb.formulation_index, None))
            
            # Execute refinements
            new_formulations = list(current_formulations)
            
            for idx, task in refinement_tasks:
                if task is not None:
                    refined, explanation = await task
                    new_formulations[idx] = refined
                    print(f"       * Formulation {idx}: REFINED - {explanation[:60]}...")
                else:
                    print(f"       * Formulation {idx}: KEPT (quality threshold met)")
            
            # Step 4: Re-solve refined formulations
            print("     - Re-solving...")
            new_results = []
            for i, form in enumerate(new_formulations):
                result_obj = verifier.verify_and_solve(form, formulation_index=i)
                new_results.append(result_obj)
            
            new_optimal = [r for r in new_results if r.status == "OPTIMAL"]
            new_values = [r.optimal_value for r in new_optimal if r.optimal_value is not None]
            new_num_unique = self._count_unique_values(new_values)
            
            print(f"     - Results: {len(new_optimal)} optimal, {new_num_unique} unique values")
            
            # Prepare for next iteration
            current_formulations = new_formulations
            current_results = new_results
            previous_avg_score = avg_score
        
        # Final metrics
        if result.iterations:
            last_iteration = result.iterations[-1]
            result.final_optimal_values = last_iteration.optimal_values
            result.final_divergence = last_iteration.value_divergence
            result.final_num_unique = last_iteration.num_unique_values
            result.final_avg_score = last_iteration.average_score
        
        result.score_improvement = result.final_avg_score - result.initial_avg_score
        result.divergence_reduction = initial_divergence - result.final_divergence
        result.agreement_improvement = float(initial_num_unique - result.final_num_unique)
        
        # Calculate final error metrics
        if ground_truth is not None and ground_truth != 0 and result.final_optimal_values:
            errors_pct = [abs(v - ground_truth) / abs(ground_truth) * 100 for v in result.final_optimal_values]
            result.final_best_error_pct = min(errors_pct)
            result.final_mean_error_pct = float(np.mean(errors_pct))
        
        # Determine final status
        if result.converged:
            result.reason = f"Converged after {len(result.iterations)} iteration(s)."
        else:
            result.reason = f"Completed {len(result.iterations)} iteration(s) with score improvement: {result.score_improvement:.1f}"
        
        return result
