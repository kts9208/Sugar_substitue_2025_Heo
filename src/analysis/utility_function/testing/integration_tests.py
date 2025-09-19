"""
Integration tests for utility function module.
"""

import pandas as pd
import numpy as np
import unittest
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from ..engine.utility_calculator import UtilityCalculator
from ..data_loader import DCEDataLoader, SEMResultsLoader
from ..results import ResultsManager, ResultsAnalyzer, ResultsExporter
from ..testing.validation_suite import ValidationSuite

logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """
    Integration test suite for the complete utility function system.
    
    Tests end-to-end workflows and component interactions.
    """
    
    def __init__(self):
        """Initialize integration test suite."""
        self.test_results = {}
        
    def run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Args:
            verbose: Whether to print detailed test output
            
        Returns:
            Dictionary with integration test results
        """
        logger.info("Starting integration tests...")
        
        test_results = {
            'end_to_end_workflow': {},
            'data_pipeline': {},
            'results_pipeline': {},
            'error_handling': {},
            'edge_cases': {}
        }
        
        # Test end-to-end workflow
        test_results['end_to_end_workflow'] = self._test_end_to_end_workflow()
        
        # Test data pipeline integration
        test_results['data_pipeline'] = self._test_data_pipeline_integration()
        
        # Test results pipeline integration
        test_results['results_pipeline'] = self._test_results_pipeline_integration()
        
        # Test error handling
        test_results['error_handling'] = self._test_error_handling()
        
        # Test edge cases
        test_results['edge_cases'] = self._test_edge_cases()
        
        # Generate integration summary
        test_results['integration_summary'] = self._generate_integration_summary(test_results)
        
        self.test_results = test_results
        
        if verbose:
            self._print_integration_summary(test_results)
            
        logger.info("Integration tests completed")
        return test_results
    
    def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow."""
        logger.info("Testing end-to-end workflow...")
        
        workflow_results = {}
        
        try:
            # Step 1: Create synthetic data
            dce_data = self._create_synthetic_dce_data()
            sem_results = self._create_synthetic_sem_results()
            
            workflow_results['data_creation'] = {
                'success': True,
                'dce_data_shape': dce_data.shape,
                'sem_results_keys': list(sem_results.keys())
            }
            
            # Step 2: Initialize calculator
            calculator = UtilityCalculator(random_seed=42)
            calculator.dce_data = {'processed_data': dce_data}
            calculator.sem_results = sem_results
            
            workflow_results['calculator_initialization'] = {'success': True}
            
            # Step 3: Integrate SEM factors
            calculator.integrate_sem_factors()
            workflow_results['sem_integration'] = {
                'success': True,
                'integrated_data_shape': calculator.integrated_data.shape,
                'new_columns': len(calculator.integrated_data.columns) - len(dce_data.columns)
            }
            
            # Step 4: Setup and fit components
            calculator.setup_components()
            calculator.fit_components()
            workflow_results['component_fitting'] = {
                'success': True,
                'is_fitted': calculator.is_fitted
            }
            
            # Step 5: Setup aggregator
            calculator.setup_aggregator()
            workflow_results['aggregator_setup'] = {
                'success': True,
                'n_components': len(calculator.aggregator.components)
            }
            
            # Step 6: Calculate utility
            total_utility = calculator.calculate_utility()
            workflow_results['utility_calculation'] = {
                'success': True,
                'utility_shape': total_utility.shape,
                'utility_stats': {
                    'mean': float(total_utility.mean()),
                    'std': float(total_utility.std()),
                    'min': float(total_utility.min()),
                    'max': float(total_utility.max())
                }
            }
            
            # Step 7: Get decomposition
            decomposition = calculator.get_utility_decomposition()
            workflow_results['utility_decomposition'] = {
                'success': True,
                'has_contributions': 'contributions' in decomposition,
                'has_importance': 'component_importance' in decomposition
            }
            
            # Step 8: Analyze results
            analyzer = ResultsAnalyzer()
            analysis = analyzer.generate_comprehensive_report({
                'total_utility': total_utility,
                'decomposition': decomposition,
                'data': calculator.integrated_data
            })
            workflow_results['results_analysis'] = {
                'success': True,
                'analysis_sections': list(analysis.keys())
            }
            
            # Step 9: Save and load results
            results_manager = ResultsManager()
            experiment_id = results_manager.save_utility_results({
                'total_utility': total_utility,
                'decomposition': decomposition,
                'data': calculator.integrated_data
            }, "integration_test")
            
            loaded_results = results_manager.load_utility_results(experiment_id)
            workflow_results['results_persistence'] = {
                'success': True,
                'experiment_id': experiment_id,
                'loaded_keys': list(loaded_results.keys())
            }
            
            # Clean up
            results_manager.delete_experiment(experiment_id)
            
        except Exception as e:
            logger.error(f"End-to-end workflow test failed: {str(e)}")
            workflow_results['error'] = str(e)
            workflow_results['success'] = False
            
        return workflow_results
    
    def _test_data_pipeline_integration(self) -> Dict[str, Any]:
        """Test data pipeline integration."""
        logger.info("Testing data pipeline integration...")
        
        pipeline_results = {}
        
        try:
            # Test DCE data loader integration
            dce_loader = DCEDataLoader()
            raw_dce_data = pd.DataFrame({
                'respondent_id': [1, 1, 2, 2],
                'choice_set': [1, 1, 1, 1],
                'alternative': [1, 2, 1, 2],
                'sugar_type': ['sugar', 'sugar_free', 'sugar', 'sugar_free'],
                'health_label': [0, 1, 1, 0],
                'price': [100, 120, 110, 115],
                'chosen': [1, 0, 0, 1]
            })
            
            processed_dce = dce_loader._process_dce_data(raw_dce_data)
            pipeline_results['dce_processing'] = {
                'success': True,
                'input_shape': raw_dce_data.shape,
                'output_keys': list(processed_dce.keys()),
                'processed_shape': processed_dce['processed_data'].shape
            }
            
            # Test SEM results loader integration
            sem_loader = SEMResultsLoader()
            raw_sem_data = pd.DataFrame({
                'Path': ['Health_Benefit -> Choice', 'Nutrition_Knowledge -> Choice', 'Perceived_Price -> Choice'],
                'Estimate': [0.45, 0.32, -0.18],
                'P_Value': [0.001, 0.015, 0.045]
            })
            
            processed_sem = sem_loader._process_sem_results(raw_sem_data)
            pipeline_results['sem_processing'] = {
                'success': True,
                'input_shape': raw_sem_data.shape,
                'output_keys': list(processed_sem.keys()),
                'n_factors': len(processed_sem.get('factor_effects', {}))
            }
            
            # Test integration between DCE and SEM data
            from ..sem_integration import SEMUtilityComponent
            sem_component = SEMUtilityComponent()
            
            integrated_data = sem_component.integrate_all_sem_factors(
                processed_dce['processed_data'], processed_sem
            )
            
            pipeline_results['data_integration'] = {
                'success': True,
                'original_columns': len(processed_dce['processed_data'].columns),
                'integrated_columns': len(integrated_data.columns),
                'new_columns_added': len(integrated_data.columns) - len(processed_dce['processed_data'].columns)
            }
            
        except Exception as e:
            logger.error(f"Data pipeline integration test failed: {str(e)}")
            pipeline_results['error'] = str(e)
            pipeline_results['success'] = False
            
        return pipeline_results
    
    def _test_results_pipeline_integration(self) -> Dict[str, Any]:
        """Test results pipeline integration."""
        logger.info("Testing results pipeline integration...")
        
        results_pipeline = {}
        
        try:
            # Create test results
            test_utility = pd.Series(np.random.normal(0, 1, 100))
            test_data = self._create_synthetic_dce_data(100)
            test_contributions = pd.DataFrame({
                'sugar_component': np.random.normal(0, 0.5, 100),
                'health_label_component': np.random.normal(0, 0.3, 100),
                'price_component': np.random.normal(0, 0.4, 100),
                'total_utility': test_utility
            })
            
            test_results = {
                'total_utility': test_utility,
                'data': test_data,
                'decomposition': {'contributions': test_contributions}
            }
            
            # Test results manager
            results_manager = ResultsManager()
            experiment_id = results_manager.save_utility_results(test_results, "pipeline_test")
            loaded_results = results_manager.load_utility_results(experiment_id)
            
            results_pipeline['results_manager'] = {
                'success': True,
                'save_successful': experiment_id is not None,
                'load_successful': 'total_utility' in loaded_results,
                'data_integrity': np.allclose(test_utility, loaded_results['total_utility'])
            }
            
            # Test results analyzer
            analyzer = ResultsAnalyzer()
            analysis = analyzer.generate_comprehensive_report(test_results)
            
            results_pipeline['results_analyzer'] = {
                'success': True,
                'has_utility_distribution': 'utility_distribution' in analysis,
                'has_component_effects': 'component_effects' in analysis,
                'has_summary': 'summary' in analysis
            }
            
            # Test results exporter
            exporter = ResultsExporter()
            csv_files = exporter.export_to_csv(test_results, "pipeline_test")
            json_file = exporter.export_to_json(test_results, "pipeline_test")
            
            results_pipeline['results_exporter'] = {
                'success': True,
                'csv_files_created': len(csv_files),
                'json_file_created': json_file.exists(),
                'export_formats': ['csv', 'json']
            }
            
            # Test validation suite
            validator = ValidationSuite()
            validation_report = validator.generate_validation_report(
                test_data, 
                {'factor_effects': {'health_benefit': {'coefficient': 0.4, 'p_value': 0.01}}},
                test_utility,
                test_contributions
            )
            
            results_pipeline['validation_suite'] = {
                'success': True,
                'has_validation_summary': 'validation_summary' in validation_report,
                'validation_sections': list(validation_report.keys())
            }
            
            # Clean up
            results_manager.delete_experiment(experiment_id)
            for file_path in csv_files + [json_file]:
                if file_path.exists():
                    file_path.unlink()
                    
        except Exception as e:
            logger.error(f"Results pipeline integration test failed: {str(e)}")
            results_pipeline['error'] = str(e)
            results_pipeline['success'] = False
            
        return results_pipeline
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling across the system."""
        logger.info("Testing error handling...")
        
        error_handling = {}
        
        # Test with invalid DCE data
        try:
            calculator = UtilityCalculator()
            invalid_dce_data = pd.DataFrame({
                'invalid_column': [1, 2, 3]
            })
            calculator.dce_data = {'processed_data': invalid_dce_data}
            calculator.sem_results = self._create_synthetic_sem_results()
            
            # This should handle the error gracefully
            try:
                calculator.integrate_sem_factors()
                error_handling['invalid_dce_data'] = {'handled_gracefully': False}
            except Exception:
                error_handling['invalid_dce_data'] = {'handled_gracefully': True}
                
        except Exception as e:
            error_handling['invalid_dce_data'] = {'error': str(e)}
            
        # Test with missing SEM results
        try:
            calculator = UtilityCalculator()
            calculator.dce_data = {'processed_data': self._create_synthetic_dce_data()}
            calculator.sem_results = {}  # Empty SEM results
            
            try:
                calculator.integrate_sem_factors()
                error_handling['missing_sem_results'] = {'handled_gracefully': False}
            except Exception:
                error_handling['missing_sem_results'] = {'handled_gracefully': True}
                
        except Exception as e:
            error_handling['missing_sem_results'] = {'error': str(e)}
            
        # Test with incompatible data sizes
        try:
            from ..components import SugarComponent
            component = SugarComponent()
            
            # Try to calculate utility with mismatched data
            small_data = pd.DataFrame({'sugar_free': [0, 1]})
            
            try:
                utility = component.calculate_utility(small_data)
                error_handling['data_size_mismatch'] = {
                    'handled_gracefully': True,
                    'result_size': len(utility)
                }
            except Exception:
                error_handling['data_size_mismatch'] = {'handled_gracefully': False}
                
        except Exception as e:
            error_handling['data_size_mismatch'] = {'error': str(e)}
            
        return error_handling
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions."""
        logger.info("Testing edge cases...")
        
        edge_cases = {}
        
        # Test with minimal data
        try:
            minimal_data = pd.DataFrame({
                'respondent_id': [1],
                'sugar_free': [0],
                'health_label': [1],
                'price_normalized': [0.5],
                'choice_value': [1]
            })
            
            calculator = UtilityCalculator(random_seed=42)
            calculator.dce_data = {'processed_data': minimal_data}
            calculator.sem_results = self._create_synthetic_sem_results()
            
            calculator.integrate_sem_factors()
            calculator.setup_components()
            calculator.fit_components()
            calculator.setup_aggregator()
            utility = calculator.calculate_utility()
            
            edge_cases['minimal_data'] = {
                'success': True,
                'utility_calculated': len(utility) == 1,
                'utility_value': float(utility.iloc[0])
            }
            
        except Exception as e:
            edge_cases['minimal_data'] = {'success': False, 'error': str(e)}
            
        # Test with extreme values
        try:
            extreme_data = pd.DataFrame({
                'respondent_id': [1, 2, 3],
                'sugar_free': [0, 1, 0],
                'health_label': [0, 1, 1],
                'price_normalized': [0.0, 1.0, 0.5],  # Extreme price values
                'choice_value': [0, 1, 1]
            })
            
            calculator = UtilityCalculator(random_seed=42)
            calculator.dce_data = {'processed_data': extreme_data}
            calculator.sem_results = self._create_synthetic_sem_results()
            
            calculator.integrate_sem_factors()
            calculator.setup_components()
            calculator.fit_components()
            calculator.setup_aggregator()
            utility = calculator.calculate_utility()
            
            edge_cases['extreme_values'] = {
                'success': True,
                'all_finite': np.isfinite(utility).all(),
                'utility_range': float(utility.max() - utility.min())
            }
            
        except Exception as e:
            edge_cases['extreme_values'] = {'success': False, 'error': str(e)}
            
        # Test with all same choices
        try:
            same_choice_data = pd.DataFrame({
                'respondent_id': [1, 2, 3, 4],
                'sugar_free': [0, 1, 0, 1],
                'health_label': [0, 0, 1, 1],
                'price_normalized': [0.2, 0.4, 0.6, 0.8],
                'choice_value': [1, 1, 1, 1]  # All chosen
            })
            
            calculator = UtilityCalculator(random_seed=42)
            calculator.dce_data = {'processed_data': same_choice_data}
            calculator.sem_results = self._create_synthetic_sem_results()
            
            calculator.integrate_sem_factors()
            calculator.setup_components()
            calculator.fit_components()
            calculator.setup_aggregator()
            utility = calculator.calculate_utility()
            
            edge_cases['same_choices'] = {
                'success': True,
                'utility_calculated': len(utility) == 4,
                'has_variation': utility.nunique() > 1
            }
            
        except Exception as e:
            edge_cases['same_choices'] = {'success': False, 'error': str(e)}
            
        return edge_cases
    
    def _create_synthetic_dce_data(self, n_observations: int = 50) -> pd.DataFrame:
        """Create synthetic DCE data for testing."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'respondent_id': np.repeat(range(1, n_observations//4 + 1), 4)[:n_observations],
            'choice_set': np.tile(range(1, 5), n_observations//4)[:n_observations],
            'sugar_free': np.random.choice([0, 1], n_observations),
            'health_label': np.random.choice([0, 1], n_observations),
            'price_normalized': np.random.uniform(0, 1, n_observations),
            'choice_value': np.random.choice([0, 1], n_observations)
        })
    
    def _create_synthetic_sem_results(self) -> Dict[str, Any]:
        """Create synthetic SEM results for testing."""
        return {
            'factor_effects': {
                'health_benefit': {'coefficient': 0.4, 'p_value': 0.01},
                'nutrition_knowledge': {'coefficient': 0.3, 'p_value': 0.02},
                'perceived_price': {'coefficient': -0.2, 'p_value': 0.03}
            }
        }
    
    def _generate_integration_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integration test summary."""
        summary = {
            'overall_status': 'PASS',
            'test_categories': {},
            'critical_failures': [],
            'warnings': []
        }
        
        for category, results in test_results.items():
            if category == 'integration_summary':
                continue
                
            if isinstance(results, dict):
                success_count = 0
                total_count = 0
                
                for test_name, test_result in results.items():
                    if isinstance(test_result, dict):
                        total_count += 1
                        if test_result.get('success', False):
                            success_count += 1
                        elif 'error' in test_result:
                            summary['critical_failures'].append(f"{category}.{test_name}: {test_result['error']}")
                            
                summary['test_categories'][category] = {
                    'passed': success_count,
                    'total': total_count,
                    'pass_rate': success_count / total_count if total_count > 0 else 0.0
                }
                
                if success_count < total_count:
                    summary['overall_status'] = 'FAIL'
                    
        return summary
    
    def _print_integration_summary(self, test_results: Dict[str, Any]):
        """Print formatted integration test summary."""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        summary = test_results.get('integration_summary', {})
        
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print()
        
        for category, stats in summary.get('test_categories', {}).items():
            print(f"{category.replace('_', ' ').title()}:")
            print(f"  Tests: {stats['passed']}/{stats['total']} passed")
            print(f"  Pass Rate: {stats['pass_rate']:.1%}")
            print()
            
        if summary.get('critical_failures'):
            print("Critical Failures:")
            for failure in summary['critical_failures']:
                print(f"  - {failure}")
            print()
            
        print("="*60)
