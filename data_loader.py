"""Data loading utilities for optimization problems."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def load_industryor_problems(
    dataset_path: Optional[str] = None,
    subset_ids: Optional[List[int]] = None,
    difficulty_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load problems from IndustryOR dataset.
    
    Args:
        dataset_path: Optional path to local JSON file
        subset_ids: Optional list of problem IDs to include
        difficulty_filter: Optional list of difficulties to include (e.g., ["Easy", "Medium", "Hard"])
        
    Returns:
        List of problem dictionaries
    """
    try:
        if dataset_path and Path(dataset_path).exists():
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            dataset = load_dataset("CardinalOperations/IndustryOR", split="test")
            data = [item for item in dataset]

        problems: List[Dict[str, Any]] = []
        for item in data:
            pid = int(item["id"])
            difficulty = item.get("difficulty", "Unknown")
            
            # Apply filters
            if subset_ids and pid not in subset_ids:
                continue
            if difficulty_filter and difficulty not in difficulty_filter:
                continue
            
            problems.append({
                "id": f"IndustryOR_{pid}",
                "nl_description": item["en_question"],
                "ground_truth_answer": float(item["en_answer"]),
                "difficulty": difficulty,
            })
        
        return problems
    except Exception as e:
        print(f"Error loading IndustryOR: {e}")
        return []
