"""
Test runner for utility function module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from ..engine.utility_calculator import UtilityCalculator
from ..data_loader import DCEDataLoader, SEMResultsLoader
from ..components import SugarComponent, HealthLabelComponent, PriceComponent
from ..results import ResultsManager, ResultsAnalyzer

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Main test runner for the utility function module.
    
    Coordinates and executes all tests including unit tests,
    integration tests, and validation tests.
    """
    
    def __init__(self, test_data_dir: Optional[Path] = None):
        """
        Initialize test runner.
        
        Args:
            test_data_dir: Directory containing test data
        """
        self.test_data_dir = test_data_dir or Path("utility_function/testing/test_data")
        self.test_results = {}
        self.test_summary = {}
        
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all tests in the utility function module.
        
        Args:
            verbose: Whether to print detailed test output
            
        Returns:
            Dictionary with test results
        """
        logger.info("Starting comprehensive test suite...")
        
        test_results = {
            'start_time': datetime.now(),
            'unit_tests': {},
            'integration_tests': {},
            'validation_tests': {},
            'performance_tests': {}
        }
        
        # Run unit tests
        logger.info("Running unit tests...")
        test_results['unit_tests'] = self._run_unit_tests(verbose)
        
        # Run integration tests
        logger.info("Running integration tests...")
        test_results['integration_tests'] = self._run_integration_tests(verbose)
        
        # Run validation tests
        logger.info("Running validation tests...")
        test_results['validation_tests'] = self._run_validation_tests(verbose)
        
        # Run performance tests
        logger.info("Running performance tests...")
        test_results['performance_tests'] = self._run_performance_tests(verbose)
        
        test_results['end_time'] = datetime.now()
        test_results['total_duration'] = (test_results['end_time'] - test_results['start_time']).total_seconds()
        
        # Generate summary
        test_results['summary'] = self._generate_test_summary(test_results)
        
        self.test_results = test_results
        logger.info("Test suite completed")
        
        if verbose:
            self._print_test_summary(test_results)
            
        return test_results
    
    def _run_unit_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run unit tests for individual components.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            Unit test results
        """
        unit_results = {}
        
        # Test data loaders
        unit_results['data_loaders'] = self._test_data_loaders()
        
        # Test utility components
        unit_results['utility_components'] = self._test_utility_components()
        
        # Test SEM integration
        unit_results['sem_integration'] = self._test_sem_integration()
        
        # Test results management
        unit_results['results_management'] = self._test_results_management()
        
        return unit_results
    
    def _test_data_loaders(self) -> Dict[str, bool]:
        """Test data loader components."""
        results = {}
        
        try:
            # Test DCE data loader
            dce_loader = DCEDataLoader()
            # Create minimal test data
            test_dce_data = pd.DataFrame({
                'respondent_id': [1, 2, 3],
                'choice_set': [1, 1, 1],
                'alternative': [1, 2, 3],
                'sugar_type': ['sugar', 'sugar_free', 'sugar'],
                'health_label': [0, 1, 0],
                'price': [100, 120, 110],
                'chosen': [1, 0, 0]
            })
            
            processed_data = dce_loader._process_dce_data(test_dce_data)
            results['dce_loader'] = isinstance(processed_data, dict) and 'processed_data' in processed_data
            
        except Exception as e:
            logger.error(f"DCE loader test failed: {str(e)}")
            results['dce_loader'] = False
            
        try:
            # Test SEM results loader
            sem_loader = SEMResultsLoader()
            # Create minimal test SEM data
            test_sem_data = pd.DataFrame({
                'Path': ['Health_Benefit -> Choice', 'Nutrition_Knowledge -> Choice'],
                'Estimate': [0.5, 0.3],
                'P_Value': [0.01, 0.05]
            })
            
            processed_sem = sem_loader._process_sem_results(test_sem_data)
            results['sem_loader'] = isinstance(processed_sem, dict)
            
        except Exception as e:
            logger.error(f"SEM loader test failed: {str(e)}")
            results['sem_loader'] = False
            
        return results
    
    def _test_utility_components(self) -> Dict[str, bool]:
        """Test utility component classes."""
        results = {}
        
        # Create test data
        test_data = pd.DataFrame({
            'sugar_free': [0, 1, 0, 1],
            'health_label': [0, 0, 1, 1],
            'price_normalized': [0.2, 0.8, 0.5, 0.3],
            'choice_value': [0, 1, 1, 0]
        })
        
        try:
            # Test Sugar Component
            sugar_comp = SugarComponent()
            sugar_comp.set_coefficient(0.5)
            utility = sugar_comp.calculate_utility(test_data)
            results['sugar_component'] = isinstance(utility, pd.Series) and len(utility) == len(test_data)
            
        except Exception as e:
            logger.error(f"Sugar component test failed: {str(e)}")
            results['sugar_component'] = False
            
        try:
            # Test Health Label Component
            health_comp = HealthLabelComponent()
            health_comp.set_coefficient(0.3)
            utility = health_comp.calculate_utility(test_data)
            results['health_label_component'] = isinstance(utility, pd.Series) and len(utility) == len(test_data)
            
        except Exception as e:
            logger.error(f"Health label component test failed: {str(e)}")
            results['health_label_component'] = False
            
        try:
            # Test Price Component
            price_comp = PriceComponent()
            price_comp.set_coefficient(-0.2)
            utility = price_comp.calculate_utility(test_data)
            results['price_component'] = isinstance(utility, pd.Series) and len(utility) == len(test_data)
            
        except Exception as e:
            logger.error(f"Price component test failed: {str(e)}")
            results['price_component'] = False
            
        return results
    
    def _test_sem_integration(self) -> Dict[str, bool]:
        """Test SEM integration components."""
        results = {}
        
        # Create test data
        test_dce_data = pd.DataFrame({
            'respondent_id': [1, 2, 3, 4],
            'sugar_free': [0, 1, 0, 1],
            'health_label': [0, 0, 1, 1],
            'price_normalized': [0.2, 0.8, 0.5, 0.3]
        })
        
        test_sem_results = {
            'factor_effects': {
                'health_benefit': {'coefficient': 0.4, 'p_value': 0.01},
                'nutrition_knowledge': {'coefficient': 0.3, 'p_value': 0.02},
                'perceived_price': {'coefficient': -0.2, 'p_value': 0.03}
            }
        }
        
        try:
            from ..sem_integration import SEMUtilityComponent
            sem_comp = SEMUtilityComponent()
            
            # Test integration
            integrated_data = sem_comp.integrate_all_sem_factors(test_dce_data, test_sem_results)
            results['sem_integration'] = isinstance(integrated_data, pd.DataFrame) and len(integrated_data) == len(test_dce_data)
            
        except Exception as e:
            logger.error(f"SEM integration test failed: {str(e)}")
            results['sem_integration'] = False
            
        return results
    
    def _test_results_management(self) -> Dict[str, bool]:
        """Test results management components."""
        results = {}
        
        try:
            # Test results manager
            results_manager = ResultsManager()
            
            # Create test results
            test_results = {
                'total_utility': pd.Series([1.0, 2.0, 1.5, 0.8]),
                'data': pd.DataFrame({'test_col': [1, 2, 3, 4]}),
                'parameters': {'test_param': 'test_value'}
            }
            
            # Test saving
            experiment_id = results_manager.save_utility_results(test_results, "test_experiment")
            results['save_results'] = isinstance(experiment_id, str)
            
            # Test loading
            loaded_results = results_manager.load_utility_results(experiment_id)
            results['load_results'] = 'total_utility' in loaded_results
            
            # Clean up
            results_manager.delete_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"Results management test failed: {str(e)}")
            results['save_results'] = False
            results['load_results'] = False
            
        try:
            # Test results analyzer
            analyzer = ResultsAnalyzer()
            test_utility = pd.Series([1.0, 2.0, 1.5, 0.8, 2.2])
            
            analysis = analyzer.analyze_utility_distribution(test_utility)
            results['results_analyzer'] = 'descriptive_stats' in analysis
            
        except Exception as e:
            logger.error(f"Results analyzer test failed: {str(e)}")
            results['results_analyzer'] = False
            
        return results
    
    def _run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run integration tests for the complete system.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            Integration test results
        """
        integration_results = {}
        
        try:
            # Test complete utility calculation workflow
            calculator = UtilityCalculator(random_seed=42)
            
            # Create synthetic test data
            test_dce_data = self._create_synthetic_dce_data()
            test_sem_results = self._create_synthetic_sem_results()
            
            # Mock the data loading
            calculator.dce_data = {'processed_data': test_dce_data}
            calculator.sem_results = test_sem_results
            
            # Test integration and fitting
            calculator.integrate_sem_factors()
            integration_results['sem_integration'] = calculator.integrated_data is not None
            
            # Test component setup and fitting
            calculator.setup_components()
            calculator.fit_components()
            integration_results['component_fitting'] = calculator.is_fitted
            
            # Test aggregator setup
            calculator.setup_aggregator()
            integration_results['aggregator_setup'] = len(calculator.aggregator.components) > 0
            
            # Test utility calculation
            total_utility = calculator.calculate_utility()
            integration_results['utility_calculation'] = isinstance(total_utility, pd.Series) and len(total_utility) > 0
            
            # Test decomposition
            decomposition = calculator.get_utility_decomposition()
            integration_results['utility_decomposition'] = 'contributions' in decomposition
            
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
            integration_results['error'] = str(e)
            
        return integration_results
    
    def _run_validation_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run validation tests to check data consistency and model validity.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            Validation test results
        """
        validation_results = {}
        
        try:
            # Test data validation
            test_data = self._create_synthetic_dce_data()
            
            # Check data types
            validation_results['data_types'] = all(
                test_data[col].dtype in [np.int64, np.float64, np.bool_] 
                for col in ['sugar_free', 'health_label', 'price_normalized']
            )
            
            # Check data ranges
            validation_results['data_ranges'] = (
                test_data['sugar_free'].isin([0, 1]).all() and
                test_data['health_label'].isin([0, 1]).all() and
                (test_data['price_normalized'] >= 0).all() and
                (test_data['price_normalized'] <= 1).all()
            )
            
            # Test model consistency
            calculator = UtilityCalculator(random_seed=42)
            calculator.dce_data = {'processed_data': test_data}
            calculator.sem_results = self._create_synthetic_sem_results()
            
            calculator.integrate_sem_factors()
            calculator.setup_components()
            calculator.fit_components()
            calculator.setup_aggregator()
            
            # Calculate utility twice to check consistency
            utility1 = calculator.calculate_utility()
            utility2 = calculator.calculate_utility()
            
            validation_results['model_consistency'] = np.allclose(utility1, utility2)
            
        except Exception as e:
            logger.error(f"Validation test failed: {str(e)}")
            validation_results['error'] = str(e)
            
        return validation_results
    
    def _run_performance_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run performance tests to check computational efficiency.
        
        Args:
            verbose: Whether to print detailed output
            
        Returns:
            Performance test results
        """
        performance_results = {}
        
        try:
            import time
            
            # Test with different data sizes
            data_sizes = [100, 500, 1000]
            
            for size in data_sizes:
                start_time = time.time()
                
                # Create larger synthetic dataset
                test_data = self._create_synthetic_dce_data(n_observations=size)
                
                calculator = UtilityCalculator(random_seed=42)
                calculator.dce_data = {'processed_data': test_data}
                calculator.sem_results = self._create_synthetic_sem_results()
                
                calculator.integrate_sem_factors()
                calculator.setup_components()
                calculator.fit_components()
                calculator.setup_aggregator()
                
                utility = calculator.calculate_utility()
                
                end_time = time.time()
                duration = end_time - start_time
                
                performance_results[f'size_{size}'] = {
                    'duration_seconds': duration,
                    'observations_per_second': size / duration if duration > 0 else float('inf')
                }
                
        except Exception as e:
            logger.error(f"Performance test failed: {str(e)}")
            performance_results['error'] = str(e)
            
        return performance_results
    
    def _create_synthetic_dce_data(self, n_observations: int = 100) -> pd.DataFrame:
        """Create synthetic DCE data for testing."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'respondent_id': np.repeat(range(1, n_observations//4 + 1), 4)[:n_observations],
            'choice_set': np.tile(range(1, 5), n_observations//4)[:n_observations],
            'sugar_free': np.random.choice([0, 1], n_observations),
            'health_label': np.random.choice([0, 1], n_observations),
            'price_normalized': np.random.uniform(0, 1, n_observations),
            'choice_value': np.random.choice([0, 1], n_observations)
        })
        
        return data
    
    def _create_synthetic_sem_results(self) -> Dict[str, Any]:
        """Create synthetic SEM results for testing."""
        return {
            'factor_effects': {
                'health_benefit': {'coefficient': 0.4, 'p_value': 0.01},
                'nutrition_knowledge': {'coefficient': 0.3, 'p_value': 0.02},
                'perceived_price': {'coefficient': -0.2, 'p_value': 0.03}
            },
            'factor_scores': pd.DataFrame({
                'respondent_id': range(1, 26),
                'health_benefit_score': np.random.normal(0, 1, 25),
                'nutrition_knowledge_score': np.random.normal(0, 1, 25),
                'perceived_price_score': np.random.normal(0, 1, 25)
            })
        }
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        summary = {
            'total_duration': test_results['total_duration'],
            'test_categories': {}
        }
        
        for category in ['unit_tests', 'integration_tests', 'validation_tests', 'performance_tests']:
            if category in test_results:
                category_results = test_results[category]
                if isinstance(category_results, dict):
                    total_tests = 0
                    passed_tests = 0
                    
                    for test_name, result in category_results.items():
                        if isinstance(result, dict):
                            for sub_test, sub_result in result.items():
                                total_tests += 1
                                if sub_result is True:
                                    passed_tests += 1
                        elif isinstance(result, bool):
                            total_tests += 1
                            if result is True:
                                passed_tests += 1
                                
                    summary['test_categories'][category] = {
                        'total_tests': total_tests,
                        'passed_tests': passed_tests,
                        'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0
                    }
                    
        return summary
    
    def _print_test_summary(self, test_results: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "="*60)
        print("UTILITY FUNCTION MODULE TEST SUMMARY")
        print("="*60)
        
        summary = test_results['summary']
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        print()
        
        for category, stats in summary['test_categories'].items():
            print(f"{category.replace('_', ' ').title()}:")
            print(f"  Tests: {stats['passed_tests']}/{stats['total_tests']} passed")
            print(f"  Pass Rate: {stats['pass_rate']:.1%}")
            print()
            
        overall_passed = sum(stats['passed_tests'] for stats in summary['test_categories'].values())
        overall_total = sum(stats['total_tests'] for stats in summary['test_categories'].values())
        overall_rate = overall_passed / overall_total if overall_total > 0 else 0.0
        
        print(f"Overall: {overall_passed}/{overall_total} tests passed ({overall_rate:.1%})")
        print("="*60)
