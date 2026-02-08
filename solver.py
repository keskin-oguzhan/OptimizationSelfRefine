"""Gurobi solver for optimization formulations."""

import re
import time
from typing import Any, Dict, List, Optional

from pyomo.environ import Binary, Constraint, ConcreteModel, Integers, Objective, Reals, Var, maximize, minimize, value
from pyomo.opt import SolverFactory, TerminationCondition

from models import Formulation, SolverResult


class SolverVerifier:
    """Solver-based verification using Pyomo with Gurobi."""
    
    def __init__(self, executable_path: Optional[str] = None):
        """Initialize Gurobi solver.
        
        Args:
            executable_path: Optional path to Gurobi executable
        """
        if executable_path:
            self.solver = SolverFactory("gurobi", executable=executable_path)
            print(f"Using Gurobi solver at: {executable_path}")
        else:
            self.solver = SolverFactory("gurobi")
        
        if not self.solver.available():
            raise RuntimeError("Gurobi solver not available. Ensure Gurobi is installed with a valid license.")

    def verify_and_solve(self, formulation: Formulation, formulation_index: int = 0) -> SolverResult:
        """Verify and solve an optimization formulation.
        
        Args:
            formulation: The formulation to solve
            formulation_index: Index of the formulation
            
        Returns:
            SolverResult with status and optimal value
        """
        start_time = time.time()
        try:
            model = self._build_model(formulation)
            result = self.solver.solve(model, tee=False)
            solve_time = time.time() - start_time

            tc = result.solver.termination_condition
            if tc == TerminationCondition.optimal:
                return SolverResult(formulation_index, "OPTIMAL", float(value(model.obj)), solve_time, "Solved")
            elif tc == TerminationCondition.infeasible:
                return SolverResult(formulation_index, "INFEASIBLE", None, solve_time, "Infeasible")
            elif tc == TerminationCondition.unbounded:
                return SolverResult(formulation_index, "UNBOUNDED", None, solve_time, "Unbounded")
            elif tc == TerminationCondition.infeasibleOrUnbounded:
                return SolverResult(formulation_index, "UNBOUNDED", None, solve_time, "Infeasible or unbounded")
            else:
                return SolverResult(formulation_index, "ERROR", None, solve_time, f"Unexpected termination: {tc}")
        except KeyError as ke:
            return SolverResult(formulation_index, "ERROR", None, time.time() - start_time, f"Variable/expression error: {ke}")
        except SyntaxError as se:
            return SolverResult(formulation_index, "ERROR", None, time.time() - start_time, f"Syntax error in constraint: {se}")
        except Exception as e:
            return SolverResult(formulation_index, "ERROR", None, time.time() - start_time, f"Build/solve error: {str(e)[:100]}")

    @staticmethod
    def _safe_replace_vars(expr: str, var_names: List[str]) -> str:
        """Replace variable tokens with 'model.<var>' using word boundaries."""
        for v in sorted(var_names, key=len, reverse=True):
            expr = re.sub(rf"\b{re.escape(v)}\b", f"model.{v}", expr)
        return expr

    def _build_model(self, formulation: Formulation) -> ConcreteModel:
        """Build Pyomo model from formulation.
        
        Args:
            formulation: The formulation to build
            
        Returns:
            ConcreteModel ready to solve
        """
        model = ConcreteModel()
        var_objects: Dict[str, Any] = {}

        # Build variables
        for var in formulation.variables:
            name = var["name"]
            var_type = var.get("var_type", "continuous")
            lb = var.get("lower_bound", 0)
            ub = var.get("upper_bound", None)

            if var_type == "continuous":
                var_objects[name] = Var(bounds=(lb, ub), within=Reals)
            elif var_type == "integer":
                var_objects[name] = Var(bounds=(lb, ub), within=Integers)
            elif var_type == "binary":
                var_objects[name] = Var(within=Binary)
            else:
                var_objects[name] = Var(bounds=(lb, ub), within=Reals)
            setattr(model, name, var_objects[name])

        # Build objective
        obj_expr = formulation.objective.get("expression")
        obj_sense = formulation.objective.get("sense")

        if not obj_expr:
            raise ValueError("Objective expression is missing or empty")
        if not obj_sense:
            raise ValueError("Objective sense (minimize/maximize) is missing")

        try:
            obj_expr = self._safe_replace_vars(obj_expr, list(var_objects.keys()))
            obj_value = eval(obj_expr, {"model": model, "sum": sum})
            model.obj = Objective(expr=obj_value, sense=minimize if obj_sense == "minimize" else maximize)
        except Exception as e:
            raise ValueError(f"Failed to build objective: {e}")

        # Build constraints
        for i, constraint in enumerate(formulation.constraints):
            expr = constraint.get("expression")
            if not expr:
                continue
            try:
                expr = self._safe_replace_vars(expr, list(var_objects.keys()))

                if "<=" in expr:
                    lhs, rhs = expr.split("<=", 1)
                    lhs_val = eval(lhs.strip(), {"model": model, "sum": sum})
                    rhs_val = eval(rhs.strip(), {"model": model, "sum": sum})
                    setattr(model, f"con_{i}", Constraint(expr=lhs_val <= rhs_val))
                elif ">=" in expr:
                    lhs, rhs = expr.split(">=", 1)
                    lhs_val = eval(lhs.strip(), {"model": model, "sum": sum})
                    rhs_val = eval(rhs.strip(), {"model": model, "sum": sum})
                    setattr(model, f"con_{i}", Constraint(expr=lhs_val >= rhs_val))
                elif "==" in expr or "=" in expr:
                    if "==" in expr:
                        lhs, rhs = expr.split("==", 1)
                    else:
                        lhs, rhs = expr.split("=", 1)
                    lhs_val = eval(lhs.strip(), {"model": model, "sum": sum})
                    rhs_val = eval(rhs.strip(), {"model": model, "sum": sum})
                    setattr(model, f"con_{i}", Constraint(expr=lhs_val == rhs_val))
            except Exception:
                continue
        return model
