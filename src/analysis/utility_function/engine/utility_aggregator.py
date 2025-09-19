"""
Utility aggregator for combining multiple utility components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

from ..components.base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class UtilityAggregator:
    """
    Aggregates utility contributions from multiple components.
    
    Combines DCE attribute effects, SEM factor effects, interaction effects,
    and error terms into a total utility value.
    """
    
    def __init__(self, aggregation_method: str = 'additive'):
        """
        Initialize utility aggregator.
        
        Args:
            aggregation_method: Method for aggregating utilities ('additive', 'multiplicative')
        """
        self.aggregation_method = aggregation_method
        self.components = {}
        self.component_weights = {}
        self.aggregation_history = []
        
    def add_component(self, component: BaseUtilityComponent, weight: float = 1.0):
        """
        Add a utility component to the aggregator.
        
        Args:
            component: Utility component to add
            weight: Weight for this component in aggregation
        """
        self.components[component.name] = component
        self.component_weights[component.name] = weight
        logger.info(f"Added component '{component.name}' with weight {weight}")
        
    def remove_component(self, component_name: str):
        """
        Remove a utility component from the aggregator.
        
        Args:
            component_name: Name of component to remove
        """
        if component_name in self.components:
            del self.components[component_name]
            del self.component_weights[component_name]
            logger.info(f"Removed component '{component_name}'")
        else:
            logger.warning(f"Component '{component_name}' not found")
            
    def set_component_weight(self, component_name: str, weight: float):
        """
        Set weight for a specific component.
        
        Args:
            component_name: Name of component
            weight: New weight value
        """
        if component_name in self.component_weights:
            self.component_weights[component_name] = weight
            logger.info(f"Set weight for '{component_name}' to {weight}")
        else:
            logger.warning(f"Component '{component_name}' not found")
            
    def calculate_total_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate total utility by aggregating all components.
        
        Args:
            data: Input data for utility calculation
            **kwargs: Additional parameters passed to components
            
        Returns:
            Series with total utility values
        """
        if not self.components:
            logger.warning("No components added to aggregator")
            return pd.Series(0.0, index=data.index)
            
        component_utilities = {}
        
        # Calculate utility for each component
        for name, component in self.components.items():
            try:
                utility = component.calculate_utility(data, **kwargs)
                weight = self.component_weights[name]
                weighted_utility = utility * weight
                component_utilities[name] = weighted_utility
                
                logger.debug(f"Calculated utility for component '{name}': "
                           f"mean={weighted_utility.mean():.4f}, std={weighted_utility.std():.4f}")
                           
            except Exception as e:
                logger.error(f"Error calculating utility for component '{name}': {str(e)}")
                component_utilities[name] = pd.Series(0.0, index=data.index)
                
        # Aggregate utilities
        total_utility = self._aggregate_utilities(component_utilities, data.index)
        
        # Store aggregation history
        self.aggregation_history.append({
            'n_observations': len(data),
            'components_used': list(component_utilities.keys()),
            'total_utility_stats': {
                'mean': total_utility.mean(),
                'std': total_utility.std(),
                'min': total_utility.min(),
                'max': total_utility.max()
            }
        })
        
        logger.info(f"Total utility calculated for {len(data)} observations: "
                   f"mean={total_utility.mean():.4f}, std={total_utility.std():.4f}")
        
        return total_utility
    
    def _aggregate_utilities(self, component_utilities: Dict[str, pd.Series], 
                           index: pd.Index) -> pd.Series:
        """
        Aggregate component utilities using specified method.
        
        Args:
            component_utilities: Dictionary of component utilities
            index: Index for result series
            
        Returns:
            Aggregated utility series
        """
        if self.aggregation_method == 'additive':
            # Simple additive aggregation
            total_utility = pd.Series(0.0, index=index)
            for utility in component_utilities.values():
                total_utility += utility
                
        elif self.aggregation_method == 'multiplicative':
            # Multiplicative aggregation (for positive utilities)
            total_utility = pd.Series(1.0, index=index)
            for utility in component_utilities.values():
                # Add 1 to avoid multiplication by zero/negative values
                total_utility *= (1 + utility)
            # Subtract 1 to return to original scale
            total_utility -= 1
            
        else:
            logger.warning(f"Unknown aggregation method '{self.aggregation_method}', using additive")
            total_utility = pd.Series(0.0, index=index)
            for utility in component_utilities.values():
                total_utility += utility
                
        return total_utility
    
    def get_component_contributions(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Get individual component contributions to total utility.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with utility contributions from each component
        """
        contributions = pd.DataFrame(index=data.index)
        
        for name, component in self.components.items():
            try:
                utility = component.calculate_utility(data, **kwargs)
                weight = self.component_weights[name]
                contributions[name] = utility * weight
            except Exception as e:
                logger.error(f"Error calculating contribution for '{name}': {str(e)}")
                contributions[name] = 0.0
                
        # Add total utility
        contributions['total_utility'] = contributions.sum(axis=1)
        
        return contributions
    
    def analyze_component_importance(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze the relative importance of each component.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with component importance analysis
        """
        contributions = self.get_component_contributions(data, **kwargs)
        
        # Calculate absolute contributions
        abs_contributions = contributions.abs()
        total_abs_contribution = abs_contributions.sum(axis=1)
        
        # Calculate relative importance
        importance = {}
        for component_name in self.components.keys():
            if component_name in abs_contributions.columns:
                relative_importance = (abs_contributions[component_name] / total_abs_contribution).mean()
                importance[component_name] = {
                    'relative_importance': relative_importance,
                    'mean_contribution': contributions[component_name].mean(),
                    'std_contribution': contributions[component_name].std(),
                    'weight': self.component_weights[component_name]
                }
                
        return importance
    
    def get_utility_decomposition(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get detailed decomposition of utility calculation.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with utility decomposition
        """
        contributions = self.get_component_contributions(data, **kwargs)
        importance = self.analyze_component_importance(data, **kwargs)
        
        decomposition = {
            'contributions': contributions,
            'component_importance': importance,
            'aggregation_method': self.aggregation_method,
            'total_utility_stats': {
                'mean': contributions['total_utility'].mean(),
                'std': contributions['total_utility'].std(),
                'min': contributions['total_utility'].min(),
                'max': contributions['total_utility'].max(),
                'percentiles': {
                    '25th': contributions['total_utility'].quantile(0.25),
                    '50th': contributions['total_utility'].quantile(0.50),
                    '75th': contributions['total_utility'].quantile(0.75)
                }
            }
        }
        
        return decomposition
    
    def validate_components(self) -> Dict[str, bool]:
        """
        Validate all components in the aggregator.
        
        Returns:
            Dictionary with validation results for each component
        """
        validation_results = {}
        
        for name, component in self.components.items():
            try:
                # Check if component is fitted
                is_fitted = getattr(component, 'is_fitted', False)
                
                # Check if component has required methods
                has_calculate_method = hasattr(component, 'calculate_utility')
                
                validation_results[name] = {
                    'is_fitted': is_fitted,
                    'has_calculate_method': has_calculate_method,
                    'is_valid': is_fitted and has_calculate_method
                }
                
            except Exception as e:
                logger.error(f"Error validating component '{name}': {str(e)}")
                validation_results[name] = {
                    'is_fitted': False,
                    'has_calculate_method': False,
                    'is_valid': False,
                    'error': str(e)
                }
                
        return validation_results
    
    def get_aggregator_info(self) -> Dict[str, Any]:
        """
        Get information about the aggregator.
        
        Returns:
            Dictionary with aggregator information
        """
        component_info = {}
        for name, component in self.components.items():
            if hasattr(component, 'get_component_info'):
                component_info[name] = component.get_component_info()
            else:
                component_info[name] = {'name': name, 'type': type(component).__name__}
                
        return {
            'aggregation_method': self.aggregation_method,
            'n_components': len(self.components),
            'component_names': list(self.components.keys()),
            'component_weights': self.component_weights,
            'component_info': component_info,
            'aggregation_history_length': len(self.aggregation_history)
        }
