# Utility Function Module

This module provides a comprehensive framework for constructing utility functions using DCE (Discrete Choice Experiment) data and SEM (Structural Equation Modeling) results.

## Module Structure

- `data_loader/`: Data loading and preprocessing modules
- `components/`: Individual utility function components
- `sem_integration/`: SEM results integration modules  
- `engine/`: Main utility function calculation engine
- `results/`: Results storage and analysis modules
- `tests/`: Testing and validation modules
- `config/`: Configuration files
- `outputs/`: Generated results and outputs

## Key Features

1. **Modular Design**: Each component is independently implemented for maximum reusability
2. **Extensibility**: Easy to add new utility function components
3. **Maintainability**: Clear separation of concerns and single responsibility principle
4. **Readability**: Well-documented code with clear naming conventions
5. **Independence**: Completely separated from other project modules

## Utility Function Components

### DCE Variables
- Sugar presence/absence (설탕 유무)
- Health label presence/absence (건강라벨 유무)  
- Price variables (가격 변수)

### SEM Results Integration
- Health benefits perception (건강유익성)
- Nutrition knowledge level (영양지식 수준)
- Perceived price impact (인지된 가격의 영향)

### Error Term
- Random utility component

## Quick Start

### Method 1: Using the Main Interface (Recommended)

```python
from utility_function.main import UtilityFunctionMain

# Initialize the system
system = UtilityFunctionMain(random_seed=42)

# Run complete analysis
experiment_id = system.run_complete_analysis("my_experiment")

# Or run quick analysis
results = system.run_quick_analysis("quick_test")
```

### Method 2: Using the Calculator Directly

```python
from utility_function.engine import UtilityCalculator

# Initialize calculator
calculator = UtilityCalculator()

# Load data
calculator.load_data()

# Integrate SEM factors
calculator.integrate_sem_factors()

# Setup and fit components
calculator.setup_components()
calculator.fit_components()

# Setup aggregator
calculator.setup_aggregator()

# Calculate utility
total_utility = calculator.calculate_utility()

# Get decomposition
decomposition = calculator.get_utility_decomposition()
```

### Method 3: Command Line Interface

```bash
# Run complete analysis
python -m utility_function.main --mode analysis --experiment-name "my_analysis"

# Run quick analysis
python -m utility_function.main --mode quick --experiment-name "quick_test"

# Run tests
python -m utility_function.main --mode test --test-type all

# Validate data
python -m utility_function.main --mode validate

# Benchmark performance
python -m utility_function.main --mode benchmark
```

## Testing and Validation

The module includes comprehensive testing and validation capabilities:

```python
# Run all tests
system = UtilityFunctionMain()
test_results = system.run_tests("all")

# Validate data quality
validation_report = system.validate_data()

# Benchmark performance
benchmark_results = system.benchmark_performance()
```

## Results Management

Results are automatically saved and can be managed through the results system:

```python
from utility_function.results import ResultsManager, ResultsAnalyzer, ResultsExporter

# Manage results
manager = ResultsManager()
experiments = manager.list_experiments()

# Analyze results
analyzer = ResultsAnalyzer()
analysis = analyzer.generate_comprehensive_report(results)

# Export results
exporter = ResultsExporter()
exporter.export_to_excel(results, "my_analysis")
```
