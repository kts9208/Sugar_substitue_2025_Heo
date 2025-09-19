"""
Health label presence/absence utility component.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class HealthLabelComponent(BaseUtilityComponent):
    """
    Utility component for health label presence/absence in DCE choices.
    
    Handles the utility contribution from choosing options with or without health labels.
    """
    
    def __init__(self, coefficient: float = 0.0):
        """
        Initialize health label component.
        
        Args:
            coefficient: Base coefficient for health label preference
        """
        super().__init__(name="health_label_component", coefficient=coefficient)
        self.label_preference = coefficient
        
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from health label presence/absence.
        
        Args:
            data: DataFrame with 'health_label' column (1=has label, 0=no label)
            **kwargs: Additional parameters including:
                - health_concern_level: Individual health concern levels
                - nutrition_knowledge_level: Individual nutrition knowledge levels
            
        Returns:
            Series with utility values for each observation
        """
        required_columns = ['health_label']
        if not self.validate_input_data(data, required_columns):
            raise ValueError("Invalid input data for health label component")
            
        # Basic utility: coefficient * health_label_indicator
        utility = self.coefficient * data['health_label']
        
        # Add interaction with health concern if available
        if 'health_concern_level' in data.columns:
            health_interaction_coeff = kwargs.get('health_interaction_coeff', 0.0)
            utility += health_interaction_coeff * data['health_label'] * data['health_concern_level']
            
        # Add interaction with nutrition knowledge if available
        if 'nutrition_knowledge_level' in data.columns:
            nutrition_interaction_coeff = kwargs.get('nutrition_interaction_coeff', 0.0)
            utility += nutrition_interaction_coeff * data['health_label'] * data['nutrition_knowledge_level']
            
        logger.debug(f"Health label component utility calculated for {len(data)} observations")
        return utility
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'HealthLabelComponent':
        """
        Fit health label component parameters to data.
        
        Args:
            data: Training data with choice outcomes
            **kwargs: Additional parameters including:
                - choice_column: Name of choice outcome column
                - method: Fitting method ('logit', 'linear', etc.)
                - include_interactions: Whether to include interaction terms
                
        Returns:
            Self for method chaining
        """
        required_columns = ['health_label']
        if not self.validate_input_data(data, required_columns):
            raise ValueError("Invalid input data for fitting health label component")
            
        choice_column = kwargs.get('choice_column', 'choice_value')
        method = kwargs.get('method', 'logit')
        include_interactions = kwargs.get('include_interactions', True)
        
        if choice_column not in data.columns:
            logger.warning(f"Choice column '{choice_column}' not found, using default coefficient")
            self.is_fitted = True
            return self
            
        try:
            if method == 'logit':
                self._fit_logit(data, choice_column, include_interactions)
            elif method == 'linear':
                self._fit_linear(data, choice_column, include_interactions)
            else:
                logger.warning(f"Unknown fitting method '{method}', using correlation")
                self._fit_correlation(data, choice_column)
                
            self.is_fitted = True
            logger.info(f"Health label component fitted with coefficient: {self.coefficient}")
            
        except Exception as e:
            logger.error(f"Error fitting health label component: {str(e)}")
            
        return self
    
    def _fit_logit(self, data: pd.DataFrame, choice_column: str, include_interactions: bool):
        """
        Fit using logistic regression.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
            include_interactions: Whether to include interaction terms
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            # Prepare features
            features = ['health_label']
            if include_interactions:
                if 'health_concern_level' in data.columns:
                    data['health_label_x_concern'] = data['health_label'] * data['health_concern_level']
                    features.append('health_label_x_concern')
                if 'nutrition_knowledge_level' in data.columns:
                    data['health_label_x_nutrition'] = data['health_label'] * data['nutrition_knowledge_level']
                    features.append('health_label_x_nutrition')
            
            X = data[features].values
            y = data[choice_column].values
            
            model = LogisticRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0][0]  # Main effect
            self.component_data['intercept'] = model.intercept_[0]
            self.component_data['model'] = model
            self.component_data['features'] = features
            
            # Store interaction coefficients
            if len(model.coef_[0]) > 1:
                self.component_data['interaction_coeffs'] = {
                    features[i]: model.coef_[0][i] for i in range(1, len(features))
                }
                
        except ImportError:
            logger.warning("sklearn not available, using correlation method")
            self._fit_correlation(data, choice_column)
    
    def _fit_linear(self, data: pd.DataFrame, choice_column: str, include_interactions: bool):
        """
        Fit using linear regression.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
            include_interactions: Whether to include interaction terms
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare features
            features = ['health_label']
            if include_interactions:
                if 'health_concern_level' in data.columns:
                    data['health_label_x_concern'] = data['health_label'] * data['health_concern_level']
                    features.append('health_label_x_concern')
                if 'nutrition_knowledge_level' in data.columns:
                    data['health_label_x_nutrition'] = data['health_label'] * data['nutrition_knowledge_level']
                    features.append('health_label_x_nutrition')
            
            X = data[features].values
            y = data[choice_column].values
            
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0]  # Main effect
            self.component_data['intercept'] = model.intercept_
            self.component_data['model'] = model
            self.component_data['features'] = features
            
            # Store interaction coefficients
            if len(model.coef_) > 1:
                self.component_data['interaction_coeffs'] = {
                    features[i]: model.coef_[i] for i in range(1, len(features))
                }
                
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
        correlation = data['health_label'].corr(data[choice_column])
        self.coefficient = correlation
        self.component_data['correlation'] = correlation
    
    def get_label_preference_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about health label preferences in the data.
        
        Args:
            data: Data with health label choices
            
        Returns:
            Dictionary with preference statistics
        """
        if 'health_label' not in data.columns:
            return {}
            
        stats = {
            'health_label_rate': data['health_label'].mean(),
            'n_with_label_choices': data['health_label'].sum(),
            'n_without_label_choices': (1 - data['health_label']).sum(),
            'total_choices': len(data)
        }
        
        if 'choice_value' in data.columns:
            # Choice rates by label presence
            with_label_data = data[data['health_label'] == 1]
            without_label_data = data[data['health_label'] == 0]
            
            if len(with_label_data) > 0:
                stats['with_label_choice_rate'] = with_label_data['choice_value'].mean()
            if len(without_label_data) > 0:
                stats['without_label_choice_rate'] = without_label_data['choice_value'].mean()
                
        return stats
    
    def set_interaction_coefficients(self, health_interaction: float = 0.0, 
                                   nutrition_interaction: float = 0.0):
        """
        Set interaction coefficients.
        
        Args:
            health_interaction: Interaction coefficient with health concern
            nutrition_interaction: Interaction coefficient with nutrition knowledge
        """
        self.component_data['health_interaction_coeff'] = health_interaction
        self.component_data['nutrition_interaction_coeff'] = nutrition_interaction
        
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the health label component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'label_preference': self.label_preference,
            'component_data': self.component_data
        })
        return info
