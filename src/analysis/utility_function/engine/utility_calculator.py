"""
Main utility function calculator engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

from ..data_loader import DCEDataLoader, SEMResultsLoader
from ..components import SugarComponent, HealthLabelComponent, PriceComponent, InteractionComponent
from ..sem_integration import SEMUtilityComponent
from .error_component import ErrorComponent
from .utility_aggregator import UtilityAggregator
from ..config.settings import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


class UtilityCalculator:
    """
    Main utility function calculator that integrates all components.
    
    Combines DCE attributes, SEM factors, interactions, and error terms
    to calculate comprehensive utility functions.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize utility calculator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or DEFAULT_RANDOM_SEED
        np.random.seed(self.random_seed)
        
        # Initialize data loaders
        self.dce_loader = DCEDataLoader()
        self.sem_loader = SEMResultsLoader()
        
        # Initialize components
        self.sugar_component = SugarComponent()
        self.health_label_component = HealthLabelComponent()
        self.price_component = PriceComponent()
        self.interaction_component = None  # Will be initialized based on requirements
        self.sem_component = SEMUtilityComponent()
        self.error_component = ErrorComponent(random_seed=self.random_seed)
        
        # Initialize aggregator
        self.aggregator = UtilityAggregator()
        
        # Data storage
        self.dce_data = None
        self.sem_results = None
        self.integrated_data = None
        self.utility_results = None
        
        self.is_fitted = False
        
    def load_data(self) -> 'UtilityCalculator':
        """
        Load DCE and SEM data.
        
        Returns:
            Self for method chaining
        """
        logger.info("Loading data for utility calculation...")
        
        # Load DCE data
        self.dce_data = self.dce_loader.load_data()
        if not self.dce_loader.validate_data(self.dce_data):
            raise ValueError("DCE data validation failed")
            
        # Load SEM results
        self.sem_results = self.sem_loader.load_data()
        if not self.sem_loader.validate_data(self.sem_results):
            raise ValueError("SEM results validation failed")
            
        logger.info("Data loading completed successfully")
        return self
    
    def setup_components(self, component_config: Optional[Dict[str, Any]] = None) -> 'UtilityCalculator':
        """
        Setup and configure utility components.
        
        Args:
            component_config: Configuration for components
            
        Returns:
            Self for method chaining
        """
        logger.info("Setting up utility components...")
        
        config = component_config or {}
        
        # Configure DCE attribute components
        sugar_config = config.get('sugar', {})
        self.sugar_component.set_coefficient(sugar_config.get('coefficient', 0.0))
        
        health_label_config = config.get('health_label', {})
        self.health_label_component.set_coefficient(health_label_config.get('coefficient', 0.0))
        
        price_config = config.get('price', {})
        self.price_component.set_coefficient(price_config.get('coefficient', 0.0))
        
        # Setup interaction components if specified
        interaction_config = config.get('interactions', {})
        if interaction_config.get('enabled', True):
            interaction_types = interaction_config.get('types', ['sugar_health', 'sugar_price', 'health_price'])
            self.interaction_component = InteractionComponent('sugar_health')  # Primary interaction
            
        # Configure error component
        error_config = config.get('error', {})
        self.error_component = ErrorComponent(
            error_type=error_config.get('type', 'gumbel'),
            scale=error_config.get('scale', 1.0),
            random_seed=self.random_seed
        )
        
        logger.info("Component setup completed")
        return self
    
    def integrate_sem_factors(self) -> 'UtilityCalculator':
        """
        Integrate SEM factors into DCE data.
        
        Returns:
            Self for method chaining
        """
        if self.dce_data is None or self.sem_results is None:
            raise ValueError("Data must be loaded before integrating SEM factors")
            
        logger.info("Integrating SEM factors...")
        
        # Use processed DCE data
        dce_processed = self.dce_data['processed_data']
        
        # Integrate all SEM factors
        self.integrated_data = self.sem_component.integrate_all_sem_factors(
            dce_processed, self.sem_results
        )
        
        logger.info("SEM factor integration completed")
        return self
    
    def fit_components(self, **kwargs) -> 'UtilityCalculator':
        """
        Fit all utility components to data.
        
        Args:
            **kwargs: Additional parameters for fitting
            
        Returns:
            Self for method chaining
        """
        if self.integrated_data is None:
            raise ValueError("SEM factors must be integrated before fitting components")
            
        logger.info("Fitting utility components...")
        
        fitting_method = kwargs.get('method', 'logit')
        choice_column = kwargs.get('choice_column', 'choice_value')
        
        # Fit DCE attribute components
        self.sugar_component.fit(self.integrated_data, method=fitting_method, choice_column=choice_column)
        self.health_label_component.fit(self.integrated_data, method=fitting_method, choice_column=choice_column)
        self.price_component.fit(self.integrated_data, method=fitting_method, choice_column=choice_column)
        
        # Fit interaction component if available
        if self.interaction_component:
            self.interaction_component.fit(self.integrated_data, method=fitting_method, choice_column=choice_column)
            
        # Fit SEM component
        self.sem_component.fit(self.integrated_data, **kwargs)
        
        # Fit error component
        self.error_component.fit(self.integrated_data, **kwargs)
        
        self.is_fitted = True
        logger.info("Component fitting completed")
        return self
    
    def setup_aggregator(self, aggregation_config: Optional[Dict[str, Any]] = None) -> 'UtilityCalculator':
        """
        Setup utility aggregator with components.
        
        Args:
            aggregation_config: Configuration for aggregation
            
        Returns:
            Self for method chaining
        """
        config = aggregation_config or {}
        
        # Set aggregation method
        method = config.get('method', 'additive')
        self.aggregator = UtilityAggregator(aggregation_method=method)
        
        # Add components with weights
        weights = config.get('weights', {})
        
        self.aggregator.add_component(self.sugar_component, weights.get('sugar', 1.0))
        self.aggregator.add_component(self.health_label_component, weights.get('health_label', 1.0))
        self.aggregator.add_component(self.price_component, weights.get('price', 1.0))
        
        if self.interaction_component:
            self.aggregator.add_component(self.interaction_component, weights.get('interaction', 1.0))
            
        self.aggregator.add_component(self.sem_component, weights.get('sem', 1.0))
        self.aggregator.add_component(self.error_component, weights.get('error', 1.0))
        
        logger.info("Aggregator setup completed")
        return self
    
    def calculate_utility(self, data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Calculate total utility for given data.
        
        Args:
            data: Input data (uses integrated_data if None)
            **kwargs: Additional parameters
            
        Returns:
            Series with total utility values
        """
        if not self.is_fitted:
            raise ValueError("Components must be fitted before calculating utility")
            
        input_data = data if data is not None else self.integrated_data
        
        if input_data is None:
            raise ValueError("No data available for utility calculation")
            
        logger.info(f"Calculating utility for {len(input_data)} observations...")
        
        # Calculate total utility using aggregator
        total_utility = self.aggregator.calculate_total_utility(input_data, **kwargs)
        
        # Store results
        self.utility_results = {
            'total_utility': total_utility,
            'data': input_data,
            'calculation_time': datetime.now(),
            'parameters': kwargs
        }
        
        logger.info(f"Utility calculation completed: mean={total_utility.mean():.4f}, std={total_utility.std():.4f}")
        return total_utility
    
    def get_utility_decomposition(self, data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Get detailed utility decomposition.
        
        Args:
            data: Input data (uses integrated_data if None)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with utility decomposition
        """
        input_data = data if data is not None else self.integrated_data
        
        if input_data is None:
            raise ValueError("No data available for utility decomposition")
            
        return self.aggregator.get_utility_decomposition(input_data, **kwargs)
    
    def analyze_component_importance(self, data: Optional[pd.DataFrame] = None, **kwargs) -> Dict[str, Any]:
        """
        Analyze the importance of each utility component.
        
        Args:
            data: Input data (uses integrated_data if None)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with component importance analysis
        """
        input_data = data if data is not None else self.integrated_data
        
        if input_data is None:
            raise ValueError("No data available for importance analysis")
            
        return self.aggregator.analyze_component_importance(input_data, **kwargs)
    
    def predict_choice_probabilities(self, data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """
        Predict choice probabilities using calculated utilities.
        
        Args:
            data: Input data (uses integrated_data if None)
            **kwargs: Additional parameters
            
        Returns:
            Series with choice probabilities
        """
        utilities = self.calculate_utility(data, **kwargs)
        
        # Convert utilities to probabilities using logistic function
        probabilities = 1 / (1 + np.exp(-utilities))
        
        return probabilities
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'calculator_info': {
                'is_fitted': self.is_fitted,
                'random_seed': self.random_seed,
                'has_data': self.integrated_data is not None,
                'has_results': self.utility_results is not None
            }
        }
        
        # Add data summaries
        if self.dce_data:
            summary['dce_summary'] = self.dce_loader.get_summary_statistics(self.dce_data)
            
        if self.sem_results:
            summary['sem_summary'] = self.sem_loader.get_summary_statistics(self.sem_results)
            
        # Add component information
        summary['components'] = {
            'sugar': self.sugar_component.get_component_info(),
            'health_label': self.health_label_component.get_component_info(),
            'price': self.price_component.get_component_info(),
            'sem': self.sem_component.get_component_info(),
            'error': self.error_component.get_component_info()
        }
        
        if self.interaction_component:
            summary['components']['interaction'] = self.interaction_component.get_component_info()
            
        # Add aggregator information
        summary['aggregator'] = self.aggregator.get_aggregator_info()
        
        # Add utility results summary if available
        if self.utility_results:
            utility = self.utility_results['total_utility']
            summary['utility_results'] = {
                'n_observations': len(utility),
                'mean': utility.mean(),
                'std': utility.std(),
                'min': utility.min(),
                'max': utility.max(),
                'calculation_time': self.utility_results['calculation_time'].isoformat()
            }
            
        return summary
    
    def reset(self) -> 'UtilityCalculator':
        """
        Reset the calculator to initial state.
        
        Returns:
            Self for method chaining
        """
        self.dce_data = None
        self.sem_results = None
        self.integrated_data = None
        self.utility_results = None
        self.is_fitted = False
        
        # Reset components
        self.sugar_component.reset()
        self.health_label_component.reset()
        self.price_component.reset()
        if self.interaction_component:
            self.interaction_component.reset()
        self.sem_component = SEMUtilityComponent()
        self.error_component.reset_errors()
        
        # Reset aggregator
        self.aggregator = UtilityAggregator()
        
        logger.info("Calculator reset completed")
        return self
