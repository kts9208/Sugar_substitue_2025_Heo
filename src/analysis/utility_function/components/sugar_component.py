"""
Sugar presence/absence utility component.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class SugarComponent(BaseUtilityComponent):
    """
    Utility component for sugar presence/absence in DCE choices.
    
    Handles the utility contribution from choosing sugar-free vs regular sugar options.
    """
    
    def __init__(self, coefficient: float = 0.0):
        """
        Initialize sugar component.
        
        Args:
            coefficient: Base coefficient for sugar-free preference
        """
        super().__init__(name="sugar_component", coefficient=coefficient)
        self.sugar_free_preference = coefficient
        
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from sugar presence/absence.
        
        Args:
            data: DataFrame with 'sugar_free' column (1=sugar-free, 0=regular sugar)
            **kwargs: Additional parameters
            
        Returns:
            Series with utility values for each observation
        """
        required_columns = ['sugar_free']
        if not self.validate_input_data(data, required_columns):
            raise ValueError("Invalid input data for sugar component")
            
        # Basic utility: coefficient * sugar_free_indicator
        utility = self.coefficient * data['sugar_free']
        
        # Add any additional effects if specified
        if 'health_concern_level' in data.columns and 'health_concern_level' in kwargs:
            # Interaction with health concern
            health_interaction = kwargs.get('health_interaction_coeff', 0.0)
            utility += health_interaction * data['sugar_free'] * data['health_concern_level']
            
        logger.debug(f"Sugar component utility calculated for {len(data)} observations")
        return utility
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'SugarComponent':
        """
        Fit sugar component parameters to data.
        
        Args:
            data: Training data with choice outcomes
            **kwargs: Additional parameters including:
                - choice_column: Name of choice outcome column
                - method: Fitting method ('logit', 'linear', etc.)
                
        Returns:
            Self for method chaining
        """
        required_columns = ['sugar_free']
        if not self.validate_input_data(data, required_columns):
            raise ValueError("Invalid input data for fitting sugar component")
            
        choice_column = kwargs.get('choice_column', 'choice_value')
        method = kwargs.get('method', 'logit')
        
        if choice_column not in data.columns:
            logger.warning(f"Choice column '{choice_column}' not found, using default coefficient")
            self.is_fitted = True
            return self
            
        try:
            if method == 'logit':
                self._fit_logit(data, choice_column)
            elif method == 'linear':
                self._fit_linear(data, choice_column)
            else:
                logger.warning(f"Unknown fitting method '{method}', using correlation")
                self._fit_correlation(data, choice_column)
                
            self.is_fitted = True
            logger.info(f"Sugar component fitted with coefficient: {self.coefficient}")
            
        except Exception as e:
            logger.error(f"Error fitting sugar component: {str(e)}")
            
        return self
    
    def _fit_logit(self, data: pd.DataFrame, choice_column: str):
        """
        Fit using logistic regression.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = data[['sugar_free']].values
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
            data: Training data
            choice_column: Name of choice outcome column
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            X = data[['sugar_free']].values
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
            data: Training data
            choice_column: Name of choice outcome column
        """
        correlation = data['sugar_free'].corr(data[choice_column])
        self.coefficient = correlation
        self.component_data['correlation'] = correlation
    
    def get_sugar_preference_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about sugar preferences in the data.
        
        Args:
            data: Data with sugar choices
            
        Returns:
            Dictionary with preference statistics
        """
        if 'sugar_free' not in data.columns:
            return {}
            
        stats = {
            'sugar_free_rate': data['sugar_free'].mean(),
            'n_sugar_free_choices': data['sugar_free'].sum(),
            'n_regular_sugar_choices': (1 - data['sugar_free']).sum(),
            'total_choices': len(data)
        }
        
        if 'choice_value' in data.columns:
            # Choice rates by sugar type
            sugar_free_data = data[data['sugar_free'] == 1]
            regular_sugar_data = data[data['sugar_free'] == 0]
            
            if len(sugar_free_data) > 0:
                stats['sugar_free_choice_rate'] = sugar_free_data['choice_value'].mean()
            if len(regular_sugar_data) > 0:
                stats['regular_sugar_choice_rate'] = regular_sugar_data['choice_value'].mean()
                
        return stats
    
    def set_health_interaction(self, interaction_coeff: float):
        """
        Set interaction coefficient with health concern.
        
        Args:
            interaction_coeff: Interaction coefficient
        """
        self.component_data['health_interaction_coeff'] = interaction_coeff
        
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the sugar component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'sugar_free_preference': self.sugar_free_preference,
            'component_data': self.component_data
        })
        return info
