"""
Load the best parameters from the previous run and save them
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

project_root = Path('.')
sys.path.insert(0, str(project_root))

from src.analysis.hybrid_choice_model.iclv_models.iclv_config import (
    MeasurementConfig,
    ChoiceConfig,
    EstimationConfig
)
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_config import (
    MultiLatentStructuralConfig,
    MultiLatentConfig
)
from src.analysis.hybrid_choice_model.iclv_models.gpu_batch_estimator import GPUBatchEstimator
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_measurement import MultiLatentMeasurement
from src.analysis.hybrid_choice_model.iclv_models.multi_latent_structural import MultiLatentStructural
from src.analysis.hybrid_choice_model.iclv_models.choice_equations import BinaryProbitChoice

# DataConfig
from dataclasses import dataclass

@dataclass
class DataConfig:
    """데이터 설정"""
    individual_id: str = 'respondent_id'
    choice_id: str = 'choice_set'


def main():
    """Extract parameters from the estimator's internal state"""
    
    print("="*80)
    print("Loading Iteration 40 Parameters")
    print("="*80)
    print()
    
    # 1. Load data
    print("1. Loading data...")
    data_path = Path('data/processed/iclv/integrated_data.csv')
    data = pd.read_csv(data_path)
    print(f"   Data shape: {data.shape}")
    
    # 2. Setup configurations (same as test script)
    print("\n2. Setting up ICLV configurations...")
    
    measurement_configs = {
        'health_concern': MeasurementConfig(
            latent_variable='health_concern',
            indicators=['q6', 'q7', 'q8', 'q9', 'q10', 'q11'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'perceived_benefit': MeasurementConfig(
            latent_variable='perceived_benefit',
            indicators=['q12', 'q13', 'q14', 'q15', 'q16', 'q17'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'perceived_price': MeasurementConfig(
            latent_variable='perceived_price',
            indicators=['q27', 'q28', 'q29'],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'nutrition_knowledge': MeasurementConfig(
            latent_variable='nutrition_knowledge',
            indicators=[f'q{i}' for i in range(30, 50)],
            n_categories=5,
            measurement_method='continuous_linear'
        ),
        'purchase_intention': MeasurementConfig(
            latent_variable='purchase_intention',
            indicators=['q18', 'q19', 'q20'],
            n_categories=5,
            measurement_method='continuous_linear'
        )
    }
    
    structural_config = MultiLatentStructuralConfig(
        endogenous_lv='purchase_intention',
        exogenous_lvs=['health_concern', 'perceived_benefit', 'perceived_price', 'nutrition_knowledge'],
        covariates=[],
        hierarchical_paths=[
            {'target': 'perceived_benefit', 'predictors': ['health_concern']},
            {'target': 'purchase_intention', 'predictors': ['perceived_benefit']}
        ],
        error_variance=1.0
    )
    
    choice_config = ChoiceConfig(
        choice_attributes=['sugar_free', 'health_label', 'price'],
    )
    
    estimation_config = EstimationConfig(
        optimizer='BFGS',
        use_analytic_gradient=True,
        n_draws=100,
        draw_type='halton',
        max_iterations=1,  # Just initialize
        calculate_se=False,
        use_parallel=False,
        n_cores=None,
        early_stopping=False,
        gradient_log_level='MINIMAL',
        use_parameter_scaling=False
    )
    
    data_config = DataConfig()
    
    # 3. Create estimator
    print("\n3. Creating estimator...")
    
    measurement = MultiLatentMeasurement(measurement_configs)
    structural = MultiLatentStructural(structural_config)
    choice = BinaryProbitChoice(choice_config)
    
    config = MultiLatentConfig(
        measurement=measurement_configs,
        structural=structural_config,
        choice=choice_config,
        estimation=estimation_config,
        data=data_config
    )
    
    estimator = GPUBatchEstimator(
        data=data,
        measurement=measurement,
        structural=structural,
        choice=choice,
        config=config
    )
    
    # 4. Get initial parameters (this will show us the structure)
    print("\n4. Getting parameter structure...")
    initial_params = estimator._get_initial_parameters()
    param_names = estimator._get_parameter_names()
    
    print(f"\nTotal parameters: {len(initial_params)}")
    print(f"Total parameter names: {len(param_names)}")
    
    # 5. Display parameter structure
    print("\n5. Parameter structure:")
    print("="*80)
    
    # Group by type
    zeta_params = [(i, name, val) for i, (name, val) in enumerate(zip(param_names, initial_params)) if 'ζ_' in name]
    sigma_params = [(i, name, val) for i, (name, val) in enumerate(zip(param_names, initial_params)) if 'σ²_' in name]
    gamma_params = [(i, name, val) for i, (name, val) in enumerate(zip(param_names, initial_params)) if 'γ_' in name]
    beta_params = [(i, name, val) for i, (name, val) in enumerate(zip(param_names, initial_params)) if 'β_' in name]
    lambda_params = [(i, name, val) for i, (name, val) in enumerate(zip(param_names, initial_params)) if 'λ_' in name]
    
    print(f"\nZeta parameters: {len(zeta_params)}")
    for i, name, val in zeta_params[:5]:
        print(f"  [{i:2d}] {name}: {val:.6f}")
    print(f"  ...")
    
    print(f"\nSigma_sq parameters: {len(sigma_params)}")
    for i, name, val in sigma_params[:5]:
        print(f"  [{i:2d}] {name}: {val:.6f}")
    print(f"  ...")
    
    print(f"\nGamma parameters: {len(gamma_params)}")
    for i, name, val in gamma_params:
        print(f"  [{i:2d}] {name}: {val:.6f}")
    
    print(f"\nBeta parameters: {len(beta_params)}")
    for i, name, val in beta_params:
        print(f"  [{i:2d}] {name}: {val:.6f}")
    
    print(f"\nLambda parameters: {len(lambda_params)}")
    for i, name, val in lambda_params:
        print(f"  [{i:2d}] {name}: {val:.6f}")
    
    # 6. Save parameter structure to file
    print("\n6. Saving parameter structure...")
    
    param_info = {
        'param_names': param_names,
        'initial_values': initial_params.tolist(),
        'n_params': len(initial_params),
        'zeta_indices': [i for i, name, _ in zeta_params],
        'sigma_indices': [i for i, name, _ in sigma_params],
        'gamma_indices': [i for i, name, _ in gamma_params],
        'beta_indices': [i for i, name, _ in beta_params],
        'lambda_indices': [i for i, name, _ in lambda_params],
    }
    
    output_file = Path('results/parameter_structure.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(param_info, f)
    
    print(f"   Saved to: {output_file}")
    
    # Also save as text
    output_txt = Path('results/parameter_structure.txt')
    with open(output_txt, 'w') as f:
        f.write("Parameter Structure\n")
        f.write("="*80 + "\n\n")
        for i, (name, val) in enumerate(zip(param_names, initial_params)):
            f.write(f"[{i:2d}] {name:50s}: {val:12.6f}\n")
    
    print(f"   Saved to: {output_txt}")
    
    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()

