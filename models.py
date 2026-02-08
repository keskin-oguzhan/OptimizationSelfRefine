"""Data models and schemas for optimization formulations and refinement."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class VariableSchema(BaseModel):
    """Schema for optimization variables."""
    name: str = Field(description="Variable name")
    var_type: str = Field(description="continuous, integer, binary")
    lower_bound: Optional[float] = Field(default=None)
    upper_bound: Optional[float] = Field(default=None)
    description: str = Field(description="What this variable represents")


class ObjectiveSchema(BaseModel):
    """Schema for objective function."""
    sense: str = Field(description="minimize or maximize")
    expression: str = Field(description="Mathematical expression")
    description: str = Field(description="Natural language description")


class ConstraintSchema(BaseModel):
    """Schema for constraints."""
    expression: str = Field(description="Constraint expression")
    constraint_type: str = Field(description="linear/bounds/logical/quadratic")
    description: str = Field(description="Constraint meaning")


class FormulationSchema(BaseModel):
    """Complete optimization formulation schema."""
    variables: List[VariableSchema]
    objective: ObjectiveSchema
    constraints: List[ConstraintSchema]


class SingleFormulationFeedback(BaseModel):
    """Structured feedback on a single formulation."""
    
    # Scores (1-5 scale, where 5 is best)
    completeness_score: int = Field(description="Are all problem aspects captured? (1-5)")
    constraint_validity_score: int = Field(description="Are constraints logically sound? (1-5)")
    objective_alignment_score: int = Field(description="Does objective match problem intent? (1-5)")
    variable_definition_score: int = Field(description="Are variables well-defined with proper bounds? (1-5)")
    
    # Detailed explanations for each score
    completeness_explanation: str = Field(description="Why this completeness score?")
    constraint_validity_explanation: str = Field(description="Why this constraint validity score?")
    objective_alignment_explanation: str = Field(description="Why this objective alignment score?")
    variable_definition_explanation: str = Field(description="Why this variable definition score?")
    
    # Overall assessment
    total_score: int = Field(description="Sum of all scores (out of 20)")
    improvement_suggestions: str = Field(description="Concrete suggestions for improvement")


class FormulationFeedback(BaseModel):
    """Feedback with formulation index."""
    formulation_index: int = Field(description="Index of the formulation being evaluated")
    
    # Scores (1-5 scale, where 5 is best)
    completeness_score: int = Field(description="Are all problem aspects captured? (1-5)")
    constraint_validity_score: int = Field(description="Are constraints logically sound? (1-5)")
    objective_alignment_score: int = Field(description="Does objective match problem intent? (1-5)")
    variable_definition_score: int = Field(description="Are variables well-defined with proper bounds? (1-5)")
    
    # Detailed explanations for each score
    completeness_explanation: str = Field(description="Why this completeness score?")
    constraint_validity_explanation: str = Field(description="Why this constraint validity score?")
    objective_alignment_explanation: str = Field(description="Why this objective alignment score?")
    variable_definition_explanation: str = Field(description="Why this variable definition score?")
    
    # Overall assessment
    total_score: int = Field(description="Sum of all scores (out of 20)")
    improvement_suggestions: str = Field(description="Concrete suggestions for improvement")


class RefinedFormulation(BaseModel):
    """A refined formulation based on feedback."""
    explanation: str = Field(description="What changes were made and why")
    refined: FormulationSchema = Field(description="The improved formulation")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Formulation:
    """Optimization formulation with variables, objective, and constraints."""
    variables: List[Dict[str, Any]]
    objective: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    raw_output: str = ""

    @classmethod
    def from_pydantic(cls, schema: FormulationSchema, raw: str = ""):
        """Create Formulation from Pydantic schema."""
        return cls(
            variables=[v.model_dump() for v in schema.variables],
            objective=schema.objective.model_dump(),
            constraints=[c.model_dump() for c in schema.constraints],
            raw_output=raw,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variables": self.variables,
            "objective": self.objective,
            "constraints": self.constraints,
        }
    
    def to_pydantic(self) -> FormulationSchema:
        """Convert back to Pydantic schema for LLM input."""
        return FormulationSchema(
            variables=[VariableSchema(**v) for v in self.variables],
            objective=ObjectiveSchema(**self.objective),
            constraints=[ConstraintSchema(**c) for c in self.constraints],
        )


@dataclass
class SolverResult:
    """Result from solving an optimization formulation."""
    formulation_index: int
    status: str
    optimal_value: Optional[float] = None
    solve_time: float = 0.0
    solver_message: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if solution is optimal."""
        return self.status == "OPTIMAL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formulation_index": self.formulation_index,
            "status": self.status,
            "optimal_value": self.optimal_value,
            "solve_time": self.solve_time,
            "solver_message": self.solver_message,
        }


@dataclass
class RefinementIteration:
    """Records one iteration of the refinement loop."""
    iteration_number: int
    formulations: List[Dict[str, Any]] = field(default_factory=list)
    feedbacks: List[Dict[str, Any]] = field(default_factory=list)
    solver_results: List[Dict[str, Any]] = field(default_factory=list)
    optimal_values: List[float] = field(default_factory=list)
    average_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    all_meet_threshold: bool = False
    value_divergence: float = 0.0
    num_unique_values: int = 0
    converged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RefinementResult:
    """Final result of the self-refinement process."""
    attempted: bool = False
    converged: bool = False
    
    # Initial state
    initial_optimal_values: List[float] = field(default_factory=list)
    initial_divergence: float = 0.0
    initial_num_unique: int = 0
    initial_avg_score: float = 0.0
    
    # Iterations
    iterations: List[RefinementIteration] = field(default_factory=list)
    
    # Final state
    final_optimal_values: List[float] = field(default_factory=list)
    final_divergence: float = 0.0
    final_num_unique: int = 0
    final_avg_score: float = 0.0
    
    # Improvement metrics
    score_improvement: float = 0.0
    divergence_reduction: float = 0.0
    agreement_improvement: float = 0.0
    
    # Error metrics (vs ground truth)
    initial_best_error_pct: Optional[float] = None
    initial_mean_error_pct: Optional[float] = None
    final_best_error_pct: Optional[float] = None
    final_mean_error_pct: Optional[float] = None
    
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemOutput:
    """Complete system output for a single problem."""
    problem_id: str
    nl_description: str
    difficulty: str
    ground_truth_answer: Optional[float] = None

    initial_candidates: int = 0
    solver_verified: int = 0
    optimal_solutions: int = 0

    ambiguity_detected: bool = False
    ambiguity_type: Optional[str] = None
    optimal_values: List[float] = field(default_factory=list)
    value_divergence: float = 0.0

    execution_time: float = 0.0

    formulations: List[Dict[str, Any]] = field(default_factory=list)
    solver_results: List[Dict[str, Any]] = field(default_factory=list)

    # Self-refinement results
    refinement: Dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
