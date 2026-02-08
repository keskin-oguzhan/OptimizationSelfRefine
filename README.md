# Uncertainty Resolution with Self-Refinement

Self-refinement approach for resolving model uncertainty in optimization formulations. The system generates multiple candidate formulations, evaluates them using structured feedback, and iteratively refines them until quality converges.

## Requirements

- Python 3.8+
- Anthropic API key
- Gurobi solver with valid license

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
GUROBI_EXECUTABLE=/path/to/gurobi_cl  # Optional
```

## Running the Code

### Basic Usage

```bash
python main.py
```

### Setting Problem Filters

Edit `main.py` to configure which problems to process:

```python
# Filter by difficulty
difficulty_filter = ["Easy", "Medium", "Hard"]  # or None for all

# Select specific problem IDs
subset_ids = [1, 2, 3]  # or None for all
```

### Configuration Options

In `main.py`, adjust system parameters:

```python
system = UncertaintyResolutionSystem(
    api_key=api_key,
    solver_executable=solver_executable,
    max_rpm=45,                      # API rate limit
    enable_refinement=True,          # Enable/disable refinement
    max_refinement_iterations=3,     # Maximum refinement iterations
    quality_threshold=18.0,          # Quality score threshold (out of 20)
)
```

## Output

Results are saved to `results/self_refinement_TIMESTAMP.json` with:
- Generated formulations
- Solver results
- Refinement iterations and feedback
- Final quality metrics

## Module Structure

- `main.py` - Entry point and configuration
- `system.py` - Main uncertainty resolution system
- `client.py` - Anthropic API client with refinement
- `solver.py` - Gurobi solver integration
- `refinement.py` - Self-refinement engine
- `models.py` - Data structures and schemas
- `prompts.py` - LLM prompt templates
- `data_loader.py` - Dataset loading utilities
