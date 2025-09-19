"""
Price utility component.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class PriceComponent(BaseUtilityComponent):
    """
    Utility component for price effects in DCE choices.
    
    Handles the utility contribution from price variations, including:
    - Linear price effects
    - Price sensitivity interactions
    - Perceived price effects from SEM results
    """
    
    def __init__(self, coefficient: float = 0.0):
        """
        Initialize price component.
        
        Args:
            coefficient: Base coefficient for price sensitivity (typically negative)
        """
        super().__init__(name="price_component", coefficient=coefficient)
        self.price_sensitivity = coefficient
        self.price_normalization_params = {}
        
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from price.
        
        Args:
            data: DataFrame with price information
            **kwargs: Additional parameters including:
                - use_normalized_price: Whether to use normalized price
                - perceived_price_level: Individual perceived price levels
                - price_interaction_coeff: Coefficient for price interactions
            
        Returns:
            Series with utility values for each observation
        """
        # Determine which price column to use
        price_column = self._get_price_column(data, kwargs)
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
            
        # Basic price utility (typically negative coefficient)
        utility = self.coefficient * data[price_column]
        
        # Add perceived price effects if available
        if 'perceived_price_level' in data.columns:
            perceived_price_coeff = kwargs.get('perceived_price_coeff', 0.0)
            utility += perceived_price_coeff * data['perceived_price_level']
            
        # Add price-quality interaction if available
        if 'health_label' in data.columns:
            price_quality_coeff = kwargs.get('price_quality_coeff', 0.0)
            utility += price_quality_coeff * data[price_column] * data['health_label']
            
        logger.debug(f"Price component utility calculated for {len(data)} observations")
        return utility
    
    def _get_price_column(self, data: pd.DataFrame, kwargs: Dict[str, Any]) -> str:
        """
        Determine which price column to use.
        
        Args:
            data: Input data
            kwargs: Additional parameters
            
        Returns:
            Name of price column to use
        """
        use_normalized = kwargs.get('use_normalized_price', True)
        
        if use_normalized and 'price_normalized' in data.columns:
            return 'price_normalized'
        elif 'price' in data.columns:
            return 'price'
        elif 'chosen_price' in data.columns:
            return 'chosen_price'
        else:
            # Find any column with 'price' in the name
            price_columns = [col for col in data.columns if 'price' in col.lower()]
            if price_columns:
                return price_columns[0]
            else:
                raise ValueError("No price column found in data")
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'PriceComponent':
        """
        Fit price component parameters to data.
        
        Args:
            data: Training data with choice outcomes
            **kwargs: Additional parameters including:
                - choice_column: Name of choice outcome column
                - method: Fitting method ('logit', 'linear', etc.)
                - normalize_price: Whether to normalize price values
                
        Returns:
            Self for method chaining
        """
        price_column = self._get_price_column(data, kwargs)
        
        if not self.validate_input_data(data, [price_column]):
            raise ValueError("Invalid input data for fitting price component")
            
        choice_column = kwargs.get('choice_column', 'choice_value')
        method = kwargs.get('method', 'logit')
        normalize_price = kwargs.get('normalize_price', True)
        
        if choice_column not in data.columns:
            logger.warning(f"Choice column '{choice_column}' not found, using default coefficient")
            self.is_fitted = True
            return self
            
        # Normalize price if requested
        if normalize_price:
            data = self._normalize_price(data, price_column)
            price_column = 'price_normalized'
            
        try:
            if method == 'logit':
                self._fit_logit(data, choice_column, price_column)
            elif method == 'linear':
                self._fit_linear(data, choice_column, price_column)
            else:
                logger.warning(f"Unknown fitting method '{method}', using correlation")
                self._fit_correlation(data, choice_column, price_column)
                
            self.is_fitted = True
            logger.info(f"Price component fitted with coefficient: {self.coefficient}")
            
        except Exception as e:
            logger.error(f"Error fitting price component: {str(e)}")
            
        return self
    
    def _normalize_price(self, data: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """
        Normalize price values.
        
        Args:
            data: Data with price column
            price_column: Name of price column
            
        Returns:
            Data with normalized price column added
        """
        data = data.copy()
        
        price_mean = data[price_column].mean()
        price_std = data[price_column].std()
        
        data['price_normalized'] = (data[price_column] - price_mean) / price_std
        
        # Store normalization parameters
        self.price_normalization_params = {
            'mean': price_mean,
            'std': price_std,
            'original_column': price_column
        }
        
        logger.info(f"Price normalized: mean={price_mean:.2f}, std={price_std:.2f}")
        return data
    
    def _fit_logit(self, data: pd.DataFrame, choice_column: str, price_column: str):
        """
        Fit using logistic regression.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
            price_column: Name of price column
        """
        try:
            from sklearn.linear_model import LogisticRegression
            
            X = data[[price_column]].values
            y = data[choice_column].values
            
            model = LogisticRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0][0]
            self.component_data['intercept'] = model.intercept_[0]
            self.component_data['model'] = model
            
        except ImportError:
            logger.warning("sklearn not available, using correlation method")
            self._fit_correlation(data, choice_column, price_column)
    
    def _fit_linear(self, data: pd.DataFrame, choice_column: str, price_column: str):
        """
        Fit using linear regression.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
            price_column: Name of price column
        """
        try:
            from sklearn.linear_model import LinearRegression
            
            X = data[[price_column]].values
            y = data[choice_column].values
            
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            
            self.coefficient = model.coef_[0]
            self.component_data['intercept'] = model.intercept_
            self.component_data['model'] = model
            
        except ImportError:
            logger.warning("sklearn not available, using correlation method")
            self._fit_correlation(data, choice_column, price_column)
    
    def _fit_correlation(self, data: pd.DataFrame, choice_column: str, price_column: str):
        """
        Fit using simple correlation.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
            price_column: Name of price column
        """
        correlation = data[price_column].corr(data[choice_column])
        self.coefficient = correlation
        self.component_data['correlation'] = correlation
    
    def get_price_sensitivity_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about price sensitivity in the data.
        
        Args:
            data: Data with price and choice information
            
        Returns:
            Dictionary with price sensitivity statistics
        """
        price_column = self._get_price_column(data, {})
        
        if price_column not in data.columns:
            return {}
            
        stats = {
            'price_stats': {
                'mean': data[price_column].mean(),
                'std': data[price_column].std(),
                'min': data[price_column].min(),
                'max': data[price_column].max()
            }
        }
        
        if 'choice_value' in data.columns:
            # Price sensitivity by choice
            chosen_data = data[data['choice_value'] == 1]
            not_chosen_data = data[data['choice_value'] == 0]
            
            if len(chosen_data) > 0:
                stats['chosen_price_mean'] = chosen_data[price_column].mean()
            if len(not_chosen_data) > 0:
                stats['not_chosen_price_mean'] = not_chosen_data[price_column].mean()
                
            # Price elasticity approximation
            price_choice_corr = data[price_column].corr(data['choice_value'])
            stats['price_choice_correlation'] = price_choice_corr
            
        return stats
    
    def normalize_new_price(self, price_values: pd.Series) -> pd.Series:
        """
        Normalize new price values using stored parameters.
        
        Args:
            price_values: New price values to normalize
            
        Returns:
            Normalized price values
        """
        if not self.price_normalization_params:
            logger.warning("No normalization parameters available")
            return price_values
            
        mean = self.price_normalization_params['mean']
        std = self.price_normalization_params['std']
        
        return (price_values - mean) / std
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the price component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'price_sensitivity': self.price_sensitivity,
            'normalization_params': self.price_normalization_params,
            'component_data': self.component_data
        })
        return info
