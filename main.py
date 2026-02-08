"""Main entry point for the uncertainty resolution system.

This script processes optimization problems from the IndustryOR dataset,
generates multiple formulations, and uses self-refinement to improve quality.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from data_loader import load_industryor_problems
from system import UncertaintyResolutionSystem

load_dotenv()


async def main():
    """Run the uncertainty resolution system."""
    # Load API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in environment")
        return

    # Get solver path if specified
    solver_executable = os.getenv("GUROBI_EXECUTABLE")
    if solver_executable and not os.path.exists(solver_executable):
        solver_executable = None
    
    print(f"Using Gurobi solver{f' at {solver_executable}' if solver_executable else ''}")
    
    # Initialize system
    system = UncertaintyResolutionSystem(
        api_key=api_key,
        solver_executable=solver_executable,
        max_rpm=45,
        enable_refinement=True,
        max_refinement_iterations=3,
        quality_threshold=18.0,
    )

    # Load problems
    # Configure problem filters here
    difficulty_filter = ["Hard"]  # Options: ["Easy", "Medium", "Hard"]
    subset_ids = [47]  # Specific problem IDs, or None for all
    
    problems = load_industryor_problems(
        difficulty_filter=difficulty_filter,
        subset_ids=subset_ids
    )
    print(f"✓ Loaded {len(problems)} problems")

    if not problems:
        print("No problems to process")
        return

    # Process problems sequentially
    results = []
    for problem in problems:
        result = await system.run_single(
            problem_id=problem["id"],
            nl_description=problem["nl_description"],
            difficulty=problem.get("difficulty", "Unknown"),
            ground_truth_answer=problem.get("ground_truth_answer"),
        )
        results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"self_refinement_{timestamp}.json"
    
    with open(out_file, "w") as f:
        json.dump({"results": [r.to_dict() for r in results]}, f, indent=2)
    print(f"\n✓ Results saved to {out_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"\n{r.problem_id}:")
        print(f"  Uncertainty detected: {r.ambiguity_detected}")
        if r.refinement and r.refinement.get('attempted'):
            ref = r.refinement
            print(f"  Converged: {ref.get('converged', False)}")
            print(f"  Iterations: {len(ref.get('iterations', []))}")
            print(f"  Score: {ref.get('initial_avg_score', 0):.1f} → {ref.get('final_avg_score', 0):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
