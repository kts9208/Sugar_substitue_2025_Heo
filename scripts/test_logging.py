"""
Test gradient check logging
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.analysis.hybrid_choice_model.iclv_models.iclv_config import create_iclv_config
from src.analysis.hybrid_choice_model.iclv_models.measurement_equations import OrderedProbitMeasurement
from src.analysis.hybrid_choice_model.iclv_models.structural_equations import LatentVariableRegression
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice
from src.analysis.hybrid_choice_model.iclv_models.simultaneous_estimator_fixed import SimultaneousEstimator

print("=" * 70)
print("ICLV Gradient Check Logging Test")
print("=" * 70)

# Load data
print("\n1. Loading data...")
data = pd.read_csv('data/processed/iclv/integrated_data.csv')
print(f"   Data shape: {data.shape}")

# Small sample
print("\n2. Extracting small sample...")
test_ids = data['respondent_id'].unique()[:10]  # Only 10 individuals
test_data = data[data['respondent_id'].isin(test_ids)].copy()
print(f"   Test individuals: {len(test_ids)}")
print(f"   Test data shape: {test_data.shape}")

# Configuration
print("\n3. Creating ICLV configuration...")
config = create_iclv_config(
    latent_variable='health_concern',
    indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
    sociodemographics=['age_std', 'gender', 'income_std'],
    choice_attributes=['price_diff', 'label_organic'],
    price_variable='price_diff',
    individual_id_column='respondent_id',
    choice_column='choice',
    n_draws=50,  # Reduced for speed
    max_iterations=100,  # Reduced for speed
    scramble_halton=True
)
print("   Configuration complete")

# Create models
print("\n4. Creating models...")
measurement_model = OrderedProbitMeasurement(config.measurement)
print("   - Measurement model created")

structural_model = LatentVariableRegression(config.structural)
print("   - Structural model created")

choice_model = BinaryProbitChoice(config.choice)
print("   - Choice model created")

# Run estimation
print("\n5. Running ICLV simultaneous estimation...")
print("   (Watch for gradient check logging)")
print()

estimator = SimultaneousEstimator(config)
results = estimator.estimate(
    test_data,
    measurement_model,
    structural_model,
    choice_model
)

print("\n" + "=" * 70)
print("ESTIMATION COMPLETE")
print("=" * 70)
print(f"\nFinal log-likelihood: {results['log_likelihood']:.4f}")
print(f"Number of iterations: {results.get('n_iterations', 'N/A')}")
print(f"Convergence: {results.get('converged', 'N/A')}")

