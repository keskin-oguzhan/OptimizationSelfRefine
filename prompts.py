"""Prompt templates for optimization formulation generation and refinement."""

# Generation prompt template
FORMULATION_GENERATION_PROMPT = """Convert this optimization problem into a mathematical formulation.

Problem:
{nl_description}

Provide a structured formulation with:
1. Decision variables (with types and bounds)
2. Objective function (with optimization direction)
3. Constraints (with descriptions)

Be precise about variable types (continuous/integer/binary) and provide actual mathematical expressions.
"""

# Evaluation prompt template
FORMULATION_EVALUATION_PROMPT = """You are evaluating a single optimization formulation for quality and correctness.

Natural Language Problem:
{nl_description}

Formulation to Evaluate:
{status_str}

Objective: {objective_sense} {objective_expression}
Description: {objective_description}

Variables ({num_variables}):
{vars_str}

Constraints ({num_constraints}):
{cons_str}
{best_context}

Task: Evaluate this formulation on four criteria using a 1-5 scale:

1. COMPLETENESS (1-5):
   - 1: Missing major problem elements (constraints, variables, or objective components)
   - 2: Missing some important elements
   - 3: Most elements present, but some gaps or ambiguities
   - 4: Nearly complete with minor gaps
   - 5: All problem aspects fully captured and well-defined

2. CONSTRAINT VALIDITY (1-5):
   - 1: Contradictory, nonsensical, or fundamentally flawed constraints
   - 2: Major logical issues or missing critical constraints (e.g., unbounded, infeasible)
   - 3: Mostly sound but with some edge cases or potential issues
   - 4: Sound constraints with minor refinement opportunities
   - 5: All constraints logically sound, complete, and properly bounded
   Note: UNBOUNDED/INFEASIBLE/ERROR status indicates serious constraint issues (likely score 1-2)

3. OBJECTIVE ALIGNMENT (1-5):
   - 1: Objective contradicts problem intent (wrong direction or wrong quantity)
   - 2: Objective is partially aligned but missing key components
   - 3: Objective captures main intent but misses some aspects
   - 4: Objective well-aligned with minor improvements possible
   - 5: Objective perfectly captures what should be optimized

4. VARIABLE DEFINITION (1-5):
   - 1: Variables poorly defined, wrong types, or missing critical bounds
   - 2: Variables defined but with significant type/bound issues
   - 3: Variables adequately defined with some improvements needed
   - 4: Variables well-defined with minor refinements possible
   - 5: All variables perfectly defined with appropriate types and bounds

Provide:
- Score for each criterion (1-5)
- Detailed explanation for each score
- Total score (sum of 4 criteria, out of 20)
- Specific, actionable improvement suggestions (e.g., "Add constraint: x <= M*y" not just "missing constraint")
"""

# Refinement prompt template
FORMULATION_REFINEMENT_PROMPT = """You are improving an optimization formulation based on structured feedback.

Natural Language Problem:
{nl_description}

Your Current Formulation:
Objective: {objective_sense} {objective_expression}
  ({objective_description})

Variables ({num_variables}):
{variables_str}

Constraints ({num_constraints}):
{constraints_str}

Solver Result: {solver_status}
{solver_details}

FEEDBACK SCORES (out of 5 each, total {total_score}/20):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Completeness: {completeness_score}/5
   {completeness_explanation}

2. Constraint Validity: {constraint_validity_score}/5
   {constraint_validity_explanation}

3. Objective Alignment: {objective_alignment_score}/5
   {objective_alignment_explanation}

4. Variable Definition: {variable_definition_score}/5
   {variable_definition_explanation}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}
{comparison_text}

Task: Create an IMPROVED formulation that addresses all the feedback above. 

Requirements:
- Fix all issues identified in the low-scoring criteria
- Implement the specific suggestions provided
- Ensure the formulation accurately represents the natural language problem
- Provide a complete formulation (all variables, objective, all constraints)
- Explain what you changed and why
"""
