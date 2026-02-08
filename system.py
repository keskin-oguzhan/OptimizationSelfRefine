"""Main uncertainty resolution system."""

import time
from typing import List, Optional

import numpy as np

from client import RefinementClient
from models import RefinementResult, SolverResult, SystemOutput
from refinement import SelfRefinementEngine
from solver import SolverVerifier


class UncertaintyResolutionSystem:
    """Complete system for optimization formulation and refinement."""
    
    def __init__(
        self,
        api_key: str,
        solver_executable: Optional[str] = None,
        max_rpm: int = 45,
        enable_refinement: bool = True,
        max_refinement_iterations: int = 3,
        quality_threshold: float = 18.0,
    ):
        self.client = RefinementClient(api_key, max_rpm)
        self.verifier = SolverVerifier(executable_path=solver_executable)
        self.refiner = SelfRefinementEngine(
            max_iterations=max_refinement_iterations,
            score_threshold=quality_threshold
        )
        self.enable_refinement = enable_refinement

    async def run_single(
        self,
        problem_id: str,
        nl_description: str,
        difficulty: str = "Unknown",
        ground_truth_answer: Optional[float] = None,
        k: int = 5,
    ) -> SystemOutput:
        """Process a single optimization problem."""
        start_time = time.time()
        try:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {problem_id} (Difficulty: {difficulty})")
            print(f"{'='*70}")

            # Step 1: Generate formulations
            print(f"[1/4] Generating {k} formulations...")
            formulations = await self.client.generate_multiple(nl_description, k)
            n_initial = len(formulations)
            print(f"  ✓ Generated {n_initial} formulations")
            if n_initial == 0:
                raise RuntimeError("No formulations generated")

            # Step 2: Solve formulations
            print("[2/4] Solving formulations with optimizer...")
            solver_results: List[SolverResult] = []
            for i, formulation in enumerate(formulations):
                result = self.verifier.verify_and_solve(formulation, formulation_index=i)
                solver_results.append(result)
                status_icon = "✓" if result.is_valid else "✗"
                status_msg = f"  {status_icon} Candidate {i}: {result.status}"
                if result.optimal_value is not None:
                    status_msg += f" (obj={result.optimal_value:.2f})"
                print(status_msg)

            optimal_count = sum(1 for r in solver_results if r.is_valid)
            print(f"  → {optimal_count}/{n_initial} formulations are optimal")

            # Step 3: Analyze uncertainty
            print("[3/4] Analyzing uncertainty...")
            optimal_values = [r.optimal_value for r in solver_results if r.is_valid and r.optimal_value is not None]
            value_divergence = float(np.std(optimal_values)) if len(optimal_values) > 1 else 0.0
            num_unique = len(set([round(v, 6) for v in optimal_values])) if optimal_values else 0
            
            # Count problematic formulations
            error_count = sum(1 for r in solver_results if r.status == "ERROR")
            unbounded_count = sum(1 for r in solver_results if r.status == "UNBOUNDED")
            infeasible_count = sum(1 for r in solver_results if r.status == "INFEASIBLE")
            problematic_count = error_count + unbounded_count + infeasible_count
            
            value_uncertainty = num_unique > 1 if optimal_values else False
            has_problems = problematic_count > 0
            needs_refinement = value_uncertainty or has_problems
            
            if value_uncertainty:
                print("  🚨 MODEL UNCERTAINTY DETECTED!")
                print(f"    - {num_unique} unique optimal values")
                print(f"    - Optimal values: {[f'{v:.2f}' for v in optimal_values]}")
            elif has_problems:
                print("  ⚠️  FORMULATION ISSUES DETECTED!")
                print(f"    - {optimal_count} optimal, {error_count} errors, {unbounded_count} unbounded, {infeasible_count} infeasible")
            else:
                print("  ✓ No uncertainty detected")

            # Step 4: Self-refinement
            refinement_result = RefinementResult(attempted=False)
            print("[4/4] Self-refinement to improve quality...")
            
            if self.enable_refinement and needs_refinement and n_initial > 0:
                try:
                    refinement_result = await self.refiner.refine(
                        nl_description, formulations, solver_results, self.client, self.verifier,
                        ground_truth=ground_truth_answer
                    )
                    
                    if refinement_result.converged:
                        print(f"  ✓ CONVERGED: {refinement_result.reason}")
                    else:
                        print(f"  ⚙️  IMPROVED: {refinement_result.reason}")
                    
                    # Print metrics
                    if refinement_result.iterations:
                        print(f"  Quality: {refinement_result.initial_avg_score:.1f} → {refinement_result.final_avg_score:.1f}")
                        print(f"  Unique values: {refinement_result.initial_num_unique} → {refinement_result.final_num_unique}")
                        
                except Exception as e:
                    print(f"  ❌ REFINEMENT FAILED: {e}")
                    refinement_result = RefinementResult(
                        attempted=True,
                        reason=f"Refinement failed: {str(e)[:200]}"
                    )
            elif not needs_refinement:
                print("  → Skipped (no issues detected)")
            else:
                print("  → Skipped (refinement disabled)")

            execution_time = time.time() - start_time
            
            # Determine ambiguity type
            ambiguity_type = None
            if value_uncertainty and has_problems:
                ambiguity_type = "MODEL_UNCERTAINTY_WITH_ERRORS"
            elif value_uncertainty:
                ambiguity_type = "MODEL_UNCERTAINTY"
            elif has_problems:
                ambiguity_type = "FORMULATION_ERRORS"

            return SystemOutput(
                problem_id=problem_id,
                nl_description=nl_description,
                difficulty=difficulty,
                ground_truth_answer=ground_truth_answer,
                initial_candidates=n_initial,
                solver_verified=optimal_count,
                optimal_solutions=optimal_count,
                ambiguity_detected=needs_refinement,
                ambiguity_type=ambiguity_type,
                optimal_values=optimal_values,
                value_divergence=value_divergence,
                execution_time=execution_time,
                formulations=[f.to_dict() for f in formulations],
                solver_results=[r.to_dict() for r in solver_results],
                refinement=refinement_result.to_dict(),
            )
        except Exception as e:
            return SystemOutput(
                problem_id=problem_id,
                nl_description=nl_description,
                difficulty=difficulty,
                ground_truth_answer=ground_truth_answer,
                execution_time=time.time() - start_time,
                error=str(e),
            )
