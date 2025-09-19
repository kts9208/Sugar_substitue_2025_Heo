"""
Interaction effects utility component.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging

from .base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class InteractionComponent(BaseUtilityComponent):
    """
    Utility component for interaction effects between DCE attributes.
    
    Handles various interaction terms such as:
    - Sugar × Health Label
    - Sugar × Price
    - Health Label × Price
    - Three-way interactions
    """
    
    def __init__(self, interaction_type: str, coefficient: float = 0.0):
        """
        Initialize interaction component.
        
        Args:
            interaction_type: Type of interaction ('sugar_health', 'sugar_price', 
                            'health_price', 'three_way', etc.)
            coefficient: Coefficient for the interaction term
        """
        super().__init__(name=f"interaction_{interaction_type}", coefficient=coefficient)
        self.interaction_type = interaction_type
        self.interaction_variables = self._get_interaction_variables(interaction_type)
        
    def _get_interaction_variables(self, interaction_type: str) -> List[str]:
        """
        Get the variables involved in the interaction.
        
        Args:
            interaction_type: Type of interaction
            
        Returns:
            List of variable names involved in the interaction
        """
        interaction_map = {
            'sugar_health': ['sugar_free', 'health_label'],
            'sugar_price': ['sugar_free', 'price'],
            'health_price': ['health_label', 'price'],
            'three_way': ['sugar_free', 'health_label', 'price'],
            'sugar_health_normalized': ['sugar_free', 'health_label'],
            'sugar_price_normalized': ['sugar_free', 'price_normalized'],
            'health_price_normalized': ['health_label', 'price_normalized']
        }
        
        return interaction_map.get(interaction_type, [])
    
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from interaction effects.
        
        Args:
            data: DataFrame with required interaction variables
            **kwargs: Additional parameters
            
        Returns:
            Series with utility values for each observation
        """
        if not self.validate_input_data(data, self.interaction_variables):
            raise ValueError(f"Invalid input data for {self.interaction_type} interaction")
            
        # Calculate interaction term
        interaction_term = self._calculate_interaction_term(data)
        
        # Apply coefficient
        utility = self.coefficient * interaction_term
        
        logger.debug(f"{self.interaction_type} interaction utility calculated for {len(data)} observations")
        return utility
    
    def _calculate_interaction_term(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the interaction term based on interaction type.
        
        Args:
            data: Input data
            
        Returns:
            Series with interaction term values
        """
        if self.interaction_type in ['sugar_health', 'sugar_health_normalized']:
            return data['sugar_free'] * data['health_label']
            
        elif self.interaction_type == 'sugar_price':
            return data['sugar_free'] * data['price']
            
        elif self.interaction_type == 'sugar_price_normalized':
            price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
            return data['sugar_free'] * data[price_col]
            
        elif self.interaction_type == 'health_price':
            return data['health_label'] * data['price']
            
        elif self.interaction_type == 'health_price_normalized':
            price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
            return data['health_label'] * data[price_col]
            
        elif self.interaction_type == 'three_way':
            price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
            return data['sugar_free'] * data['health_label'] * data[price_col]
            
        else:
            logger.warning(f"Unknown interaction type: {self.interaction_type}")
            return pd.Series(0, index=data.index)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'InteractionComponent':
        """
        Fit interaction component parameters to data.
        
        Args:
            data: Training data with choice outcomes
            **kwargs: Additional parameters including:
                - choice_column: Name of choice outcome column
                - method: Fitting method ('logit', 'linear', etc.)
                
        Returns:
            Self for method chaining
        """
        if not self.validate_input_data(data, self.interaction_variables):
            raise ValueError(f"Invalid input data for fitting {self.interaction_type} interaction")
            
        choice_column = kwargs.get('choice_column', 'choice_value')
        method = kwargs.get('method', 'logit')
        
        if choice_column not in data.columns:
            logger.warning(f"Choice column '{choice_column}' not found, using default coefficient")
            self.is_fitted = True
            return self
            
        # Calculate interaction term
        interaction_term = self._calculate_interaction_term(data)
        data_with_interaction = data.copy()
        data_with_interaction['interaction_term'] = interaction_term
        
        try:
            if method == 'logit':
                self._fit_logit(data_with_interaction, choice_column)
            elif method == 'linear':
                self._fit_linear(data_with_interaction, choice_column)
            else:
                logger.warning(f"Unknown fitting method '{method}', using correlation")
                self._fit_correlation(data_with_interaction, choice_column)
                
            self.is_fitted = True
            logger.info(f"{self.interaction_type} interaction fitted with coefficient: {self.coefficient}")
            
        except Exception as e:
            logger.error(f"Error fitting {self.interaction_type} interaction: {str(e)}")
            
        return self
    
    def _fit_logit(self, data: pd.DataFrame, choice_column: str):
        """
        Fit using logistic regression.
        
        Args:
            data: Training data with interaction term
            choice_column: Name of choice outcome column
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = data[['interaction_term']].values
            y = data[choice_column].values
            
            model = LogisticRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0][0]
            self.component_data['intercept'] = model.intercept_[0]
            self.component_data['model'] = model
            
        except ImportError:
            logger.warning("sklearn not available, using correlation method")
            self._fit_correlation(data, choice_column)
    
    def _fit_linear(self, data: pd.DataFrame, choice_column: str):
        """
        Fit using linear regression.
        
        Args:
            data: Training data with interaction term
            choice_column: Name of choice outcome column
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            X = data[['interaction_term']].values
            y = data[choice_column].values
            
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0]
            self.component_data['intercept'] = model.intercept_
            self.component_data['model'] = model
            
        except ImportError:
            logger.warning("sklearn not available, using correlation method")
            self._fit_correlation(data, choice_column)
    
    def _fit_correlation(self, data: pd.DataFrame, choice_column: str):
        """
        Fit using simple correlation.
        
        Args:
            data: Training data with interaction term
            choice_column: Name of choice outcome column
        """
        correlation = data['interaction_term'].corr(data[choice_column])
        self.coefficient = correlation
        self.component_data['correlation'] = correlation
    
    def get_interaction_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the interaction term.
        
        Args:
            data: Data with interaction variables
            
        Returns:
            Dictionary with interaction statistics
        """
        if not all(var in data.columns for var in self.interaction_variables):
            return {}
            
        interaction_term = self._calculate_interaction_term(data)
        
        stats = {
            'interaction_type': self.interaction_type,
            'variables': self.interaction_variables,
            'interaction_stats': {
                'mean': interaction_term.mean(),
                'std': interaction_term.std(),
                'min': interaction_term.min(),
                'max': interaction_term.max(),
                'non_zero_rate': (interaction_term != 0).mean()
            }
        }
        
        if 'choice_value' in data.columns:
            # Interaction effect on choice
            interaction_choice_corr = interaction_term.corr(data['choice_value'])
            stats['interaction_choice_correlation'] = interaction_choice_corr
            
            # Choice rates by interaction level
            if self.interaction_type in ['sugar_health', 'sugar_health_normalized']:
                # For binary × binary interaction
                stats['choice_rates_by_interaction'] = {
                    'both_present': data[(data['sugar_free'] == 1) & (data['health_label'] == 1)]['choice_value'].mean(),
                    'sugar_only': data[(data['sugar_free'] == 1) & (data['health_label'] == 0)]['choice_value'].mean(),
                    'label_only': data[(data['sugar_free'] == 0) & (data['health_label'] == 1)]['choice_value'].mean(),
                    'neither': data[(data['sugar_free'] == 0) & (data['health_label'] == 0)]['choice_value'].mean()
                }
                
        return stats
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the interaction component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'interaction_type': self.interaction_type,
            'interaction_variables': self.interaction_variables,
            'component_data': self.component_data
        })
        return info


class MultipleInteractionComponent(BaseUtilityComponent):
    """
    Component that handles multiple interaction effects simultaneously.
    """
    
    def __init__(self, interaction_types: List[str], coefficients: Optional[Dict[str, float]] = None):
        """
        Initialize multiple interaction component.
        
        Args:
            interaction_types: List of interaction types to include
            coefficients: Dictionary mapping interaction types to coefficients
        """
        super().__init__(name="multiple_interactions", coefficient=0.0)
        self.interaction_types = interaction_types
        self.interaction_components = {}
        
        # Initialize individual interaction components
        for interaction_type in interaction_types:
            coeff = coefficients.get(interaction_type, 0.0) if coefficients else 0.0
            self.interaction_components[interaction_type] = InteractionComponent(interaction_type, coeff)
    
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility from all interaction effects.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Series with total interaction utility
        """
        total_utility = pd.Series(0.0, index=data.index)
        
        for interaction_type, component in self.interaction_components.items():
            try:
                interaction_utility = component.calculate_utility(data, **kwargs)
                total_utility += interaction_utility
            except Exception as e:
                logger.warning(f"Could not calculate {interaction_type} interaction: {str(e)}")
                
        return total_utility
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'MultipleInteractionComponent':
        """
        Fit all interaction components.
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        for interaction_type, component in self.interaction_components.items():
            try:
                component.fit(data, **kwargs)
            except Exception as e:
                logger.warning(f"Could not fit {interaction_type} interaction: {str(e)}")
                
        self.is_fitted = True
        return self
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about all interaction components.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info['interaction_components'] = {
            interaction_type: component.get_component_info()
            for interaction_type, component in self.interaction_components.items()
        }
        return info
