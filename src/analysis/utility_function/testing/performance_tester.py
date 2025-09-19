"""
Performance testing module for utility function calculations.
"""

import pandas as pd
import numpy as np
import time
import gc
from typing import Dict, Any, List, Optional, Callable
import logging
from datetime import datetime
from pathlib import Path

# Try to import psutil, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

    # Fallback class for basic memory monitoring
    class MockProcess:
        def memory_info(self):
            return type('obj', (object,), {'rss': 0})()

    class MockPsutil:
        @staticmethod
        def Process():
            return MockProcess()

        @staticmethod
        def cpu_count():
            return 1

        @staticmethod
        def virtual_memory():
            return type('obj', (object,), {
                'total': 8 * 1024 * 1024 * 1024,  # 8GB
                'available': 4 * 1024 * 1024 * 1024  # 4GB
            })()

        sys = type('obj', (object,), {
            'version_info': type('obj', (object,), {
                'major': 3, 'minor': 8, 'micro': 0
            })()
        })()

    psutil = MockPsutil()

from ..engine.utility_calculator import UtilityCalculator

logger = logging.getLogger(__name__)


class PerformanceTester:
    """
    Performance testing suite for utility function calculations.
    
    Tests computational efficiency, memory usage, and scalability.
    """
    
    def __init__(self):
        """Initialize performance tester."""
        self.test_results = {}
        self.baseline_metrics = {}
        
    def run_performance_tests(self, test_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run comprehensive performance tests.
        
        Args:
            test_sizes: List of data sizes to test (default: [100, 500, 1000, 5000])
            
        Returns:
            Dictionary with performance test results
        """
        if test_sizes is None:
            test_sizes = [100, 500, 1000, 5000]
            
        logger.info("Starting performance tests...")
        
        results = {
            'test_timestamp': datetime.now().isoformat(),
            'test_sizes': test_sizes,
            'system_info': self._get_system_info(),
            'baseline_performance': {},
            'scalability_analysis': {},
            'memory_usage': {},
            'component_performance': {}
        }
        
        # Run baseline performance tests
        results['baseline_performance'] = self._test_baseline_performance(test_sizes)
        
        # Run scalability analysis
        results['scalability_analysis'] = self._test_scalability(test_sizes)
        
        # Run memory usage tests
        results['memory_usage'] = self._test_memory_usage(test_sizes)
        
        # Run component-specific performance tests
        results['component_performance'] = self._test_component_performance()
        
        # Generate performance summary
        results['performance_summary'] = self._generate_performance_summary(results)
        
        self.test_results = results
        logger.info("Performance tests completed")
        
        return results
    
    def _test_baseline_performance(self, test_sizes: List[int]) -> Dict[str, Any]:
        """
        Test baseline performance across different data sizes.
        
        Args:
            test_sizes: List of data sizes to test
            
        Returns:
            Baseline performance results
        """
        logger.info("Testing baseline performance...")
        
        baseline_results = {}
        
        for size in test_sizes:
            logger.info(f"Testing with {size} observations...")
            
            # Create test data
            test_data = self._create_test_data(size)
            test_sem_results = self._create_test_sem_results()
            
            # Measure total execution time
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0  # MB
            
            try:
                # Run complete utility calculation
                calculator = UtilityCalculator(random_seed=42)
                calculator.dce_data = {'processed_data': test_data}
                calculator.sem_results = test_sem_results
                
                calculator.integrate_sem_factors()
                calculator.setup_components()
                calculator.fit_components()
                calculator.setup_aggregator()
                utility = calculator.calculate_utility()
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0  # MB
                
                baseline_results[f'size_{size}'] = {
                    'total_time_seconds': end_time - start_time,
                    'observations_per_second': size / (end_time - start_time),
                    'memory_usage_mb': end_memory - start_memory,
                    'memory_per_observation_kb': (end_memory - start_memory) * 1024 / size,
                    'success': True,
                    'utility_stats': {
                        'mean': float(utility.mean()),
                        'std': float(utility.std()),
                        'n_observations': len(utility)
                    }
                }
                
            except Exception as e:
                logger.error(f"Performance test failed for size {size}: {str(e)}")
                baseline_results[f'size_{size}'] = {
                    'success': False,
                    'error': str(e),
                    'total_time_seconds': time.time() - start_time
                }
                
            # Clean up memory
            gc.collect()
            
        return baseline_results
    
    def _test_scalability(self, test_sizes: List[int]) -> Dict[str, Any]:
        """
        Test scalability characteristics.
        
        Args:
            test_sizes: List of data sizes to test
            
        Returns:
            Scalability analysis results
        """
        logger.info("Testing scalability...")
        
        scalability_results = {
            'time_complexity': {},
            'memory_complexity': {},
            'efficiency_metrics': {}
        }
        
        times = []
        memories = []
        sizes = []
        
        for size in test_sizes:
            test_data = self._create_test_data(size)
            test_sem_results = self._create_test_sem_results()
            
            # Measure time and memory
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0
            
            try:
                calculator = UtilityCalculator(random_seed=42)
                calculator.dce_data = {'processed_data': test_data}
                calculator.sem_results = test_sem_results
                
                calculator.integrate_sem_factors()
                calculator.setup_components()
                calculator.fit_components()
                calculator.setup_aggregator()
                calculator.calculate_utility()
                
                execution_time = time.time() - start_time
                memory_usage = (psutil.Process().memory_info().rss / 1024 / 1024 - start_memory) if PSUTIL_AVAILABLE else 0
                
                times.append(execution_time)
                memories.append(memory_usage)
                sizes.append(size)
                
            except Exception as e:
                logger.error(f"Scalability test failed for size {size}: {str(e)}")
                
            gc.collect()
            
        # Analyze time complexity
        if len(times) >= 2:
            # Calculate growth rates
            time_growth_rates = []
            memory_growth_rates = []
            
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[i-1]
                time_ratio = times[i] / times[i-1]
                memory_ratio = memories[i] / memories[i-1] if memories[i-1] > 0 else 1
                
                time_growth_rates.append(time_ratio / size_ratio)
                memory_growth_rates.append(memory_ratio / size_ratio)
                
            scalability_results['time_complexity'] = {
                'execution_times': times,
                'growth_rates': time_growth_rates,
                'average_growth_rate': np.mean(time_growth_rates) if time_growth_rates else 1.0,
                'complexity_estimate': self._estimate_complexity(sizes, times)
            }
            
            scalability_results['memory_complexity'] = {
                'memory_usage': memories,
                'growth_rates': memory_growth_rates,
                'average_growth_rate': np.mean(memory_growth_rates) if memory_growth_rates else 1.0,
                'complexity_estimate': self._estimate_complexity(sizes, memories)
            }
            
        # Efficiency metrics
        if times and sizes:
            max_throughput = max(s/t for s, t in zip(sizes, times) if t > 0)
            min_throughput = min(s/t for s, t in zip(sizes, times) if t > 0)
            
            scalability_results['efficiency_metrics'] = {
                'max_throughput_obs_per_sec': max_throughput,
                'min_throughput_obs_per_sec': min_throughput,
                'throughput_degradation': (max_throughput - min_throughput) / max_throughput if max_throughput > 0 else 0
            }
            
        return scalability_results
    
    def _test_memory_usage(self, test_sizes: List[int]) -> Dict[str, Any]:
        """
        Test memory usage patterns.
        
        Args:
            test_sizes: List of data sizes to test
            
        Returns:
            Memory usage analysis results
        """
        logger.info("Testing memory usage...")
        
        memory_results = {
            'peak_memory_usage': {},
            'memory_efficiency': {},
            'garbage_collection': {}
        }
        
        for size in test_sizes:
            # Monitor memory throughout execution
            memory_snapshots = []
            
            def memory_monitor():
                return psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0
                
            initial_memory = memory_monitor()
            memory_snapshots.append(('start', initial_memory))
            
            test_data = self._create_test_data(size)
            memory_snapshots.append(('data_created', memory_monitor()))
            
            test_sem_results = self._create_test_sem_results()
            memory_snapshots.append(('sem_created', memory_monitor()))
            
            try:
                calculator = UtilityCalculator(random_seed=42)
                memory_snapshots.append(('calculator_created', memory_monitor()))
                
                calculator.dce_data = {'processed_data': test_data}
                calculator.sem_results = test_sem_results
                memory_snapshots.append(('data_loaded', memory_monitor()))
                
                calculator.integrate_sem_factors()
                memory_snapshots.append(('sem_integrated', memory_monitor()))
                
                calculator.setup_components()
                memory_snapshots.append(('components_setup', memory_monitor()))
                
                calculator.fit_components()
                memory_snapshots.append(('components_fitted', memory_monitor()))
                
                calculator.setup_aggregator()
                memory_snapshots.append(('aggregator_setup', memory_monitor()))
                
                utility = calculator.calculate_utility()
                peak_memory = memory_monitor()
                memory_snapshots.append(('utility_calculated', peak_memory))
                
                # Calculate memory metrics
                memory_usage = peak_memory - initial_memory
                memory_per_observation = memory_usage * 1024 / size  # KB per observation
                
                memory_results['peak_memory_usage'][f'size_{size}'] = {
                    'initial_memory_mb': initial_memory,
                    'peak_memory_mb': peak_memory,
                    'memory_usage_mb': memory_usage,
                    'memory_per_observation_kb': memory_per_observation,
                    'memory_snapshots': memory_snapshots
                }
                
                # Test garbage collection effectiveness
                gc_before = memory_monitor()
                del calculator, utility, test_data, test_sem_results
                gc.collect()
                gc_after = memory_monitor()
                
                memory_results['garbage_collection'][f'size_{size}'] = {
                    'memory_before_gc_mb': gc_before,
                    'memory_after_gc_mb': gc_after,
                    'memory_freed_mb': gc_before - gc_after,
                    'gc_effectiveness': (gc_before - gc_after) / memory_usage if memory_usage > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Memory test failed for size {size}: {str(e)}")
                memory_results['peak_memory_usage'][f'size_{size}'] = {
                    'success': False,
                    'error': str(e)
                }
                
        return memory_results
    
    def _test_component_performance(self) -> Dict[str, Any]:
        """
        Test performance of individual components.
        
        Returns:
            Component performance results
        """
        logger.info("Testing component performance...")
        
        component_results = {}
        test_data = self._create_test_data(1000)  # Standard size for component testing
        
        # Test individual components
        components_to_test = [
            ('data_loading', self._test_data_loading_performance),
            ('sem_integration', self._test_sem_integration_performance),
            ('component_fitting', self._test_component_fitting_performance),
            ('utility_calculation', self._test_utility_calculation_performance),
            ('aggregation', self._test_aggregation_performance)
        ]
        
        for component_name, test_function in components_to_test:
            try:
                start_time = time.time()
                result = test_function(test_data)
                execution_time = time.time() - start_time
                
                component_results[component_name] = {
                    'execution_time_seconds': execution_time,
                    'success': True,
                    'details': result
                }
                
            except Exception as e:
                logger.error(f"Component performance test failed for {component_name}: {str(e)}")
                component_results[component_name] = {
                    'success': False,
                    'error': str(e)
                }
                
        return component_results
    
    def _test_data_loading_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test data loading performance."""
        from ..data_loader import DCEDataLoader, SEMResultsLoader
        
        # Test DCE data processing
        dce_loader = DCEDataLoader()
        start_time = time.time()
        processed_dce = dce_loader._process_dce_data(test_data)
        dce_time = time.time() - start_time
        
        # Test SEM data processing
        sem_loader = SEMResultsLoader()
        test_sem_data = pd.DataFrame({
            'Path': ['Health_Benefit -> Choice', 'Nutrition_Knowledge -> Choice'],
            'Estimate': [0.5, 0.3],
            'P_Value': [0.01, 0.05]
        })
        start_time = time.time()
        processed_sem = sem_loader._process_sem_results(test_sem_data)
        sem_time = time.time() - start_time
        
        return {
            'dce_processing_time': dce_time,
            'sem_processing_time': sem_time,
            'total_time': dce_time + sem_time
        }
    
    def _test_sem_integration_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test SEM integration performance."""
        from ..sem_integration import SEMUtilityComponent
        
        sem_component = SEMUtilityComponent()
        test_sem_results = self._create_test_sem_results()
        
        start_time = time.time()
        integrated_data = sem_component.integrate_all_sem_factors(test_data, test_sem_results)
        integration_time = time.time() - start_time
        
        return {
            'integration_time': integration_time,
            'data_size_increase': len(integrated_data.columns) - len(test_data.columns)
        }
    
    def _test_component_fitting_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test component fitting performance."""
        from ..components import SugarComponent, HealthLabelComponent, PriceComponent
        
        # Add choice column for fitting
        test_data_with_choice = test_data.copy()
        test_data_with_choice['choice_value'] = np.random.choice([0, 1], len(test_data))
        
        components = [
            ('sugar', SugarComponent()),
            ('health_label', HealthLabelComponent()),
            ('price', PriceComponent())
        ]
        
        fitting_times = {}
        
        for name, component in components:
            start_time = time.time()
            component.fit(test_data_with_choice)
            fitting_times[f'{name}_fitting_time'] = time.time() - start_time
            
        return fitting_times
    
    def _test_utility_calculation_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test utility calculation performance."""
        from ..components import SugarComponent, HealthLabelComponent, PriceComponent
        
        components = [
            SugarComponent(),
            HealthLabelComponent(), 
            PriceComponent()
        ]
        
        calculation_times = {}
        
        for i, component in enumerate(components):
            component.set_coefficient(0.5)
            start_time = time.time()
            utility = component.calculate_utility(test_data)
            calculation_times[f'component_{i}_calculation_time'] = time.time() - start_time
            
        return calculation_times
    
    def _test_aggregation_performance(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test aggregation performance."""
        from ..engine.utility_aggregator import UtilityAggregator
        from ..components import SugarComponent, HealthLabelComponent, PriceComponent
        
        aggregator = UtilityAggregator()
        
        # Add components
        components = [SugarComponent(), HealthLabelComponent(), PriceComponent()]
        for component in components:
            component.set_coefficient(0.5)
            aggregator.add_component(component)
            
        start_time = time.time()
        total_utility = aggregator.calculate_total_utility(test_data)
        aggregation_time = time.time() - start_time
        
        return {
            'aggregation_time': aggregation_time,
            'n_components': len(components)
        }
    
    def _create_test_data(self, n_observations: int) -> pd.DataFrame:
        """Create test data for performance testing."""
        np.random.seed(42)
        
        return pd.DataFrame({
            'respondent_id': np.repeat(range(1, n_observations//4 + 1), 4)[:n_observations],
            'choice_set': np.tile(range(1, 5), n_observations//4)[:n_observations],
            'sugar_free': np.random.choice([0, 1], n_observations),
            'health_label': np.random.choice([0, 1], n_observations),
            'price_normalized': np.random.uniform(0, 1, n_observations),
            'choice_value': np.random.choice([0, 1], n_observations)
        })
    
    def _create_test_sem_results(self) -> Dict[str, Any]:
        """Create test SEM results for performance testing."""
        return {
            'factor_effects': {
                'health_benefit': {'coefficient': 0.4, 'p_value': 0.01},
                'nutrition_knowledge': {'coefficient': 0.3, 'p_value': 0.02},
                'perceived_price': {'coefficient': -0.2, 'p_value': 0.03}
            }
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
        }
    
    def _estimate_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate computational complexity from timing data."""
        if len(sizes) < 2 or len(times) < 2:
            return "insufficient_data"
            
        # Calculate growth ratios
        growth_ratios = []
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            growth_ratios.append(time_ratio / size_ratio)
            
        avg_growth = np.mean(growth_ratios)
        
        if avg_growth < 1.2:
            return "O(n) - Linear"
        elif avg_growth < 2.0:
            return "O(n log n) - Linearithmic"
        elif avg_growth < 3.0:
            return "O(n²) - Quadratic"
        else:
            return "O(n³+) - Polynomial or worse"
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            'overall_performance': 'GOOD',
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Analyze baseline performance
        baseline = results.get('baseline_performance', {})
        if baseline:
            throughputs = []
            for size_key, metrics in baseline.items():
                if metrics.get('success', False):
                    throughputs.append(metrics.get('observations_per_second', 0))
                    
            if throughputs:
                avg_throughput = np.mean(throughputs)
                if avg_throughput < 100:
                    summary['overall_performance'] = 'POOR'
                    summary['bottlenecks'].append('Low throughput (< 100 obs/sec)')
                elif avg_throughput < 500:
                    summary['overall_performance'] = 'FAIR'
                    
        # Analyze scalability
        scalability = results.get('scalability_analysis', {})
        time_complexity = scalability.get('time_complexity', {})
        if time_complexity.get('complexity_estimate', '').startswith('O(n²)'):
            summary['bottlenecks'].append('Quadratic time complexity detected')
            summary['recommendations'].append('Consider algorithm optimization for large datasets')
            
        # Analyze memory usage
        memory_usage = results.get('memory_usage', {})
        if memory_usage:
            high_memory_usage = False
            for size_key, metrics in memory_usage.get('peak_memory_usage', {}).items():
                if metrics.get('memory_per_observation_kb', 0) > 100:  # > 100KB per observation
                    high_memory_usage = True
                    break
                    
            if high_memory_usage:
                summary['bottlenecks'].append('High memory usage per observation')
                summary['recommendations'].append('Consider memory optimization techniques')
                
        return summary
