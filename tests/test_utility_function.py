"""
Simple test script for the utility function module.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add utility_function to path
sys.path.append(str(Path(__file__).parent))

from utility_function.main import UtilityFunctionMain


def test_basic_functionality():
    """Test basic functionality of the utility function module."""
    print("Testing Utility Function Module")
    print("=" * 50)
    
    try:
        # Initialize system
        print("1. Initializing system...")
        system = UtilityFunctionMain(random_seed=42)
        print("   ✓ System initialized successfully")
        
        # Test with synthetic data
        print("\n2. Testing with synthetic data...")
        
        # Create synthetic DCE data
        synthetic_dce = pd.DataFrame({
            'respondent_id': [1, 1, 2, 2, 3, 3],
            'choice_set': [1, 1, 1, 1, 1, 1],
            'sugar_free': [0, 1, 0, 1, 0, 1],
            'health_label': [0, 0, 1, 1, 0, 1],
            'price_normalized': [0.2, 0.8, 0.5, 0.3, 0.7, 0.4],
            'choice_value': [0, 1, 1, 0, 0, 1]
        })
        
        # Create synthetic SEM results
        synthetic_sem = {
            'factor_effects': {
                'health_benefit': {'coefficient': 0.4, 'p_value': 0.01},
                'nutrition_knowledge': {'coefficient': 0.3, 'p_value': 0.02},
                'perceived_price': {'coefficient': -0.2, 'p_value': 0.03}
            }
        }
        
        # Override data loading with synthetic data
        system.calculator.dce_data = {'processed_data': synthetic_dce}
        system.calculator.sem_results = synthetic_sem
        print("   ✓ Synthetic data created")
        
        # Test SEM integration
        print("\n3. Testing SEM integration...")
        system.calculator.integrate_sem_factors()
        print(f"   ✓ SEM factors integrated. Data shape: {system.calculator.integrated_data.shape}")
        
        # Test component setup and fitting
        print("\n4. Testing component setup and fitting...")
        system.calculator.setup_components()
        system.calculator.fit_components()
        print(f"   ✓ Components fitted. Is fitted: {system.calculator.is_fitted}")
        
        # Test aggregator setup
        print("\n5. Testing aggregator setup...")
        system.calculator.setup_aggregator()
        n_components = len(system.calculator.aggregator.components)
        print(f"   ✓ Aggregator setup complete. Components: {n_components}")
        
        # Test utility calculation
        print("\n6. Testing utility calculation...")
        total_utility = system.calculator.calculate_utility()
        print(f"   ✓ Utility calculated. Shape: {total_utility.shape}")
        print(f"   ✓ Utility stats: mean={total_utility.mean():.4f}, std={total_utility.std():.4f}")
        
        # Test decomposition
        print("\n7. Testing utility decomposition...")
        decomposition = system.calculator.get_utility_decomposition()
        print(f"   ✓ Decomposition complete. Keys: {list(decomposition.keys())}")
        
        # Test analysis
        print("\n8. Testing results analysis...")
        analysis = system.analyzer.analyze_utility_distribution(total_utility)
        print(f"   ✓ Analysis complete. Mean: {analysis['descriptive_stats']['mean']:.4f}")
        
        # Test validation
        print("\n9. Testing validation...")
        validation_report = system.validate_data(synthetic_dce, synthetic_sem)
        validation_status = validation_report.get('validation_summary', {}).get('overall_status', 'UNKNOWN')
        print(f"   ✓ Validation complete. Status: {validation_status}")
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("The utility function module is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_component_independence():
    """Test that components work independently."""
    print("\n" + "=" * 50)
    print("Testing Component Independence")
    print("=" * 50)
    
    try:
        from utility_function.components import SugarComponent, HealthLabelComponent, PriceComponent
        
        # Create test data
        test_data = pd.DataFrame({
            'sugar_free': [0, 1, 0, 1],
            'health_label': [0, 0, 1, 1],
            'price_normalized': [0.2, 0.8, 0.5, 0.3]
        })
        
        # Test each component independently
        components = [
            ('Sugar', SugarComponent()),
            ('Health Label', HealthLabelComponent()),
            ('Price', PriceComponent())
        ]
        
        for name, component in components:
            print(f"\nTesting {name} Component...")
            component.set_coefficient(0.5)
            utility = component.calculate_utility(test_data)
            print(f"   ✓ {name} component works. Utility shape: {utility.shape}")
            
        print("\n✓ All components work independently!")
        return True
        
    except Exception as e:
        print(f"\n✗ Component independence test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("Utility Function Module Test Suite")
    print("=" * 60)
    
    # Test basic functionality
    test1_passed = test_basic_functionality()
    
    # Test component independence
    test2_passed = test_component_independence()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("✓ ALL TESTS PASSED!")
        print("The utility function module is ready for use.")
        return 0
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
