"""
Main execution script for utility function module.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .engine.utility_calculator import UtilityCalculator
from .results import ResultsManager, ResultsAnalyzer, ResultsExporter, UtilityVisualizer
from .testing import TestRunner, ValidationSuite, PerformanceTester, IntegrationTestSuite
from .config.settings import OUTPUT_DIR, DEFAULT_RANDOM_SEED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'utility_function.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class UtilityFunctionMain:
    """
    Main class for running utility function calculations.
    
    Provides high-level interface for the complete utility function system.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize main utility function system.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or DEFAULT_RANDOM_SEED
        self.calculator = UtilityCalculator(random_seed=self.random_seed)
        self.results_manager = ResultsManager()
        self.analyzer = ResultsAnalyzer()
        self.exporter = ResultsExporter()
        self.visualizer = UtilityVisualizer()
        
        logger.info("Utility Function System initialized")
        
    def run_complete_analysis(self, experiment_name: str = "utility_analysis") -> str:
        """
        Run complete utility function analysis workflow.
        
        Args:
            experiment_name: Name for this analysis experiment
            
        Returns:
            Experiment ID for saved results
        """
        logger.info(f"Starting complete analysis: {experiment_name}")
        
        try:
            # Step 1: Load and validate data
            logger.info("Step 1: Loading and validating data...")
            self.calculator.load_data()
            
            # Step 2: Integrate SEM factors
            logger.info("Step 2: Integrating SEM factors...")
            self.calculator.integrate_sem_factors()
            
            # Step 3: Setup and fit components
            logger.info("Step 3: Setting up and fitting components...")
            self.calculator.setup_components()
            self.calculator.fit_components()
            
            # Step 4: Setup aggregator
            logger.info("Step 4: Setting up utility aggregator...")
            self.calculator.setup_aggregator()
            
            # Step 5: Calculate utility
            logger.info("Step 5: Calculating utility...")
            total_utility = self.calculator.calculate_utility()
            
            # Step 6: Get decomposition and analysis
            logger.info("Step 6: Analyzing results...")
            decomposition = self.calculator.get_utility_decomposition()
            importance = self.calculator.analyze_component_importance()
            
            # Step 7: Comprehensive analysis
            logger.info("Step 7: Generating comprehensive analysis...")
            results = {
                'total_utility': total_utility,
                'decomposition': decomposition,
                'importance': importance,
                'data': self.calculator.integrated_data,
                'parameters': {
                    'random_seed': self.random_seed,
                    'experiment_name': experiment_name
                }
            }
            
            analysis = self.analyzer.generate_comprehensive_report(results)
            results['analysis'] = analysis
            
            # Step 8: Save results
            logger.info("Step 8: Saving results...")
            experiment_id = self.results_manager.save_utility_results(results, experiment_name)
            
            # Step 9: Generate exports and visualizations
            logger.info("Step 9: Generating exports and visualizations...")
            self._generate_outputs(results, experiment_id)
            
            logger.info(f"Complete analysis finished successfully: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {str(e)}")
            raise
    
    def run_quick_analysis(self, experiment_name: str = "quick_analysis") -> Dict[str, Any]:
        """
        Run quick utility function analysis (no file outputs).
        
        Args:
            experiment_name: Name for this analysis
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting quick analysis: {experiment_name}")
        
        try:
            # Load data and run basic analysis
            self.calculator.load_data()
            self.calculator.integrate_sem_factors()
            self.calculator.setup_components()
            self.calculator.fit_components()
            self.calculator.setup_aggregator()
            
            total_utility = self.calculator.calculate_utility()
            decomposition = self.calculator.get_utility_decomposition()
            
            # Basic analysis
            results = {
                'total_utility': total_utility,
                'decomposition': decomposition,
                'data': self.calculator.integrated_data
            }
            
            analysis = self.analyzer.analyze_utility_distribution(total_utility)
            results['analysis'] = analysis
            
            logger.info("Quick analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Quick analysis failed: {str(e)}")
            raise
    
    def run_tests(self, test_type: str = "all") -> Dict[str, Any]:
        """
        Run tests for the utility function system.
        
        Args:
            test_type: Type of tests to run ('all', 'unit', 'integration', 'performance', 'validation')
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Running tests: {test_type}")
        
        test_results = {}
        
        if test_type in ['all', 'unit', 'integration']:
            # Run main test suite
            test_runner = TestRunner()
            test_results['main_tests'] = test_runner.run_all_tests()
            
        if test_type in ['all', 'integration']:
            # Run integration tests
            integration_suite = IntegrationTestSuite()
            test_results['integration_tests'] = integration_suite.run_integration_tests()
            
        if test_type in ['all', 'performance']:
            # Run performance tests
            performance_tester = PerformanceTester()
            test_results['performance_tests'] = performance_tester.run_performance_tests()
            
        if test_type in ['all', 'validation']:
            # Run validation tests with synthetic data
            validator = ValidationSuite()
            
            # Create synthetic data for validation
            synthetic_dce = self._create_synthetic_dce_data()
            synthetic_sem = self._create_synthetic_sem_results()
            synthetic_utility = pd.Series(np.random.normal(0, 1, len(synthetic_dce)))
            
            validation_report = validator.generate_validation_report(
                synthetic_dce, synthetic_sem, synthetic_utility
            )
            test_results['validation_tests'] = validation_report
            
        logger.info("Tests completed")
        return test_results
    
    def validate_data(self, dce_data: Optional[pd.DataFrame] = None,
                     sem_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate input data quality.
        
        Args:
            dce_data: DCE data to validate (loads from config if None)
            sem_results: SEM results to validate (loads from config if None)
            
        Returns:
            Validation report
        """
        logger.info("Validating input data...")
        
        validator = ValidationSuite()
        
        # Load data if not provided
        if dce_data is None or sem_results is None:
            self.calculator.load_data()
            dce_data = self.calculator.dce_data['processed_data']
            sem_results = self.calculator.sem_results
            
        # Run validation
        validation_report = validator.generate_validation_report(
            dce_data, sem_results, pd.Series(np.random.normal(0, 1, len(dce_data)))
        )
        
        logger.info("Data validation completed")
        return validation_report
    
    def benchmark_performance(self, data_sizes: Optional[list] = None) -> Dict[str, Any]:
        """
        Benchmark system performance.
        
        Args:
            data_sizes: List of data sizes to test
            
        Returns:
            Performance benchmark results
        """
        logger.info("Running performance benchmark...")
        
        performance_tester = PerformanceTester()
        benchmark_results = performance_tester.run_performance_tests(data_sizes)
        
        logger.info("Performance benchmark completed")
        return benchmark_results
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive system summary.
        
        Returns:
            System summary information
        """
        summary = {
            'system_info': {
                'random_seed': self.random_seed,
                'output_directory': str(OUTPUT_DIR),
                'calculator_fitted': self.calculator.is_fitted
            },
            'data_info': {},
            'results_info': {}
        }
        
        # Add data information if available
        if self.calculator.dce_data:
            summary['data_info']['dce_data'] = {
                'shape': self.calculator.dce_data['processed_data'].shape,
                'columns': list(self.calculator.dce_data['processed_data'].columns)
            }
            
        if self.calculator.sem_results:
            summary['data_info']['sem_results'] = {
                'factor_effects': list(self.calculator.sem_results.get('factor_effects', {}).keys())
            }
            
        # Add results information
        experiments = self.results_manager.list_experiments()
        summary['results_info'] = {
            'n_experiments': len(experiments),
            'recent_experiments': [exp['experiment_id'] for exp in experiments[:5]]
        }
        
        return summary
    
    def _generate_outputs(self, results: Dict[str, Any], experiment_id: str):
        """Generate output files and visualizations."""
        try:
            # Export to multiple formats
            self.exporter.export_to_csv(results, experiment_id)
            self.exporter.export_to_excel(results, experiment_id)
            self.exporter.export_to_json(results, experiment_id)
            
            # Generate summary report
            self.exporter.generate_summary_report(results, results.get('analysis'))
            
            # Create visualizations
            if 'total_utility' in results:
                self.visualizer.plot_utility_distribution(results['total_utility'])
                
            if 'decomposition' in results and 'contributions' in results['decomposition']:
                self.visualizer.plot_component_contributions(results['decomposition']['contributions'])
                
            if 'data' in results and 'total_utility' in results:
                self.visualizer.plot_attribute_effects(results['data'], results['total_utility'])
                
                # Model performance if choice data available
                if 'choice_value' in results['data'].columns:
                    self.visualizer.plot_model_performance(
                        results['total_utility'], results['data']['choice_value']
                    )
                    
            # Create comprehensive dashboard
            self.visualizer.create_comprehensive_dashboard(results, results.get('analysis'))
            
        except Exception as e:
            logger.warning(f"Could not generate all outputs: {str(e)}")
    
    def _create_synthetic_dce_data(self, n_observations: int = 100) -> pd.DataFrame:
        """Create synthetic DCE data for testing."""
        np.random.seed(self.random_seed)
        
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


def main():
    """Main entry point for the utility function system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility Function Analysis System')
    parser.add_argument('--mode', choices=['analysis', 'quick', 'test', 'validate', 'benchmark'], 
                       default='analysis', help='Mode to run')
    parser.add_argument('--experiment-name', default='utility_analysis', 
                       help='Name for the experiment')
    parser.add_argument('--test-type', choices=['all', 'unit', 'integration', 'performance', 'validation'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--random-seed', type=int, default=DEFAULT_RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize system
    system = UtilityFunctionMain(random_seed=args.random_seed)
    
    try:
        if args.mode == 'analysis':
            experiment_id = system.run_complete_analysis(args.experiment_name)
            print(f"Complete analysis finished. Experiment ID: {experiment_id}")
            
        elif args.mode == 'quick':
            results = system.run_quick_analysis(args.experiment_name)
            print("Quick analysis completed successfully")
            print(f"Utility statistics: mean={results['total_utility'].mean():.4f}, "
                  f"std={results['total_utility'].std():.4f}")
            
        elif args.mode == 'test':
            test_results = system.run_tests(args.test_type)
            print("Tests completed. Check logs for detailed results.")
            
        elif args.mode == 'validate':
            validation_report = system.validate_data()
            print("Data validation completed")
            print(f"Validation status: {validation_report.get('validation_summary', {}).get('overall_status', 'UNKNOWN')}")
            
        elif args.mode == 'benchmark':
            benchmark_results = system.benchmark_performance()
            print("Performance benchmark completed. Check logs for detailed results.")
            
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
