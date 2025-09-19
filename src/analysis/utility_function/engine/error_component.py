"""
Error term component for utility function.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from ..components.base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class ErrorComponent(BaseUtilityComponent):
    """
    Error term component for utility function.
    
    Handles the random utility component that captures unobserved factors
    affecting choice behavior.
    """
    
    def __init__(self, error_type: str = 'gumbel', scale: float = 1.0, random_seed: Optional[int] = None):
        """
        Initialize error component.
        
        Args:
            error_type: Type of error distribution ('gumbel', 'normal', 'logistic')
            scale: Scale parameter for the error distribution
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="error_component", coefficient=1.0)
        self.error_type = error_type
        self.scale = scale
        self.random_seed = random_seed
        self.error_values = None
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate error term contribution to utility.
        
        Args:
            data: Input data
            **kwargs: Additional parameters including:
                - regenerate_errors: Whether to generate new error terms
                - use_cached_errors: Whether to use previously generated errors
            
        Returns:
            Series with error term values
        """
        regenerate = kwargs.get('regenerate_errors', False)
        use_cached = kwargs.get('use_cached_errors', True)
        
        # Use cached errors if available and requested
        if use_cached and self.error_values is not None and len(self.error_values) == len(data):
            logger.debug("Using cached error values")
            return self.error_values
            
        # Generate new error terms
        if regenerate or self.error_values is None:
            self.error_values = self._generate_error_terms(len(data))
            
        logger.debug(f"Error component calculated for {len(data)} observations")
        return self.error_values
    
    def _generate_error_terms(self, n_observations: int) -> pd.Series:
        """
        Generate error terms based on specified distribution.
        
        Args:
            n_observations: Number of observations
            
        Returns:
            Series with error terms
        """
        if self.error_type == 'gumbel':
            # Gumbel distribution (Type I extreme value)
            # Standard Gumbel has location=0, scale=1
            errors = np.random.gumbel(loc=0, scale=self.scale, size=n_observations)
            
        elif self.error_type == 'normal':
            # Normal distribution
            errors = np.random.normal(loc=0, scale=self.scale, size=n_observations)
            
        elif self.error_type == 'logistic':
            # Logistic distribution
            errors = np.random.logistic(loc=0, scale=self.scale, size=n_observations)
            
        else:
            logger.warning(f"Unknown error type '{self.error_type}', using normal distribution")
            errors = np.random.normal(loc=0, scale=self.scale, size=n_observations)
            
        return pd.Series(errors)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'ErrorComponent':
        """
        Fit error component parameters.
        
        Args:
            data: Training data
            **kwargs: Additional parameters including:
                - choice_column: Name of choice outcome column
                - estimate_scale: Whether to estimate scale parameter
                
        Returns:
            Self for method chaining
        """
        choice_column = kwargs.get('choice_column', 'choice_value')
        estimate_scale = kwargs.get('estimate_scale', False)
        
        if estimate_scale and choice_column in data.columns:
            # Estimate scale parameter from choice data
            self._estimate_scale_parameter(data, choice_column)
            
        self.is_fitted = True
        return self
    
    def _estimate_scale_parameter(self, data: pd.DataFrame, choice_column: str):
        """
        Estimate scale parameter from choice data.
        
        Args:
            data: Training data
            choice_column: Name of choice outcome column
        """
        # Simple estimation based on choice variance
        # In practice, this would be done through maximum likelihood estimation
        choice_variance = data[choice_column].var()
        
        if self.error_type == 'gumbel':
            # For Gumbel distribution, variance = (π²/6) * scale²
            estimated_scale = np.sqrt(choice_variance * 6 / (np.pi**2))
        elif self.error_type == 'normal':
            # For normal distribution, variance = scale²
            estimated_scale = np.sqrt(choice_variance)
        elif self.error_type == 'logistic':
            # For logistic distribution, variance = (π²/3) * scale²
            estimated_scale = np.sqrt(choice_variance * 3 / (np.pi**2))
        else:
            estimated_scale = np.sqrt(choice_variance)
            
        self.scale = estimated_scale
        logger.info(f"Estimated scale parameter: {self.scale:.4f}")
    
    def set_random_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
        np.random.seed(seed)
        # Clear cached errors to force regeneration with new seed
        self.error_values = None
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated error terms.
        
        Returns:
            Dictionary with error statistics
        """
        if self.error_values is None:
            return {}
            
        stats = {
            'error_type': self.error_type,
            'scale': self.scale,
            'n_observations': len(self.error_values),
            'mean': self.error_values.mean(),
            'std': self.error_values.std(),
            'min': self.error_values.min(),
            'max': self.error_values.max(),
            'percentiles': {
                '5th': self.error_values.quantile(0.05),
                '25th': self.error_values.quantile(0.25),
                '50th': self.error_values.quantile(0.50),
                '75th': self.error_values.quantile(0.75),
                '95th': self.error_values.quantile(0.95)
            }
        }
        
        return stats
    
    def reset_errors(self):
        """Reset cached error values."""
        self.error_values = None
        
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the error component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'error_type': self.error_type,
            'scale': self.scale,
            'random_seed': self.random_seed,
            'has_cached_errors': self.error_values is not None,
            'n_cached_errors': len(self.error_values) if self.error_values is not None else 0
        })
        return info


class MultipleErrorComponent(BaseUtilityComponent):
    """
    Component for handling multiple error terms (e.g., for different choice alternatives).
    """
    
    def __init__(self, n_alternatives: int, error_type: str = 'gumbel', 
                 scale: float = 1.0, correlation: float = 0.0, random_seed: Optional[int] = None):
        """
        Initialize multiple error component.
        
        Args:
            n_alternatives: Number of choice alternatives
            error_type: Type of error distribution
            scale: Scale parameter for error distributions
            correlation: Correlation between error terms
            random_seed: Random seed for reproducibility
        """
        super().__init__(name="multiple_error_component", coefficient=1.0)
        self.n_alternatives = n_alternatives
        self.error_type = error_type
        self.scale = scale
        self.correlation = correlation
        self.random_seed = random_seed
        self.error_matrix = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate error terms for multiple alternatives.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with error terms for each alternative
        """
        alternative_id = kwargs.get('alternative_id', 0)
        
        if self.error_matrix is None or len(self.error_matrix) != len(data):
            self.error_matrix = self._generate_error_matrix(len(data))
            
        # Return error terms for specified alternative
        return self.error_matrix.iloc[:, alternative_id]
    
    def _generate_error_matrix(self, n_observations: int) -> pd.DataFrame:
        """
        Generate correlated error terms for multiple alternatives.
        
        Args:
            n_observations: Number of observations
            
        Returns:
            DataFrame with error terms for each alternative
        """
        if self.correlation == 0.0:
            # Independent errors
            error_matrix = np.zeros((n_observations, self.n_alternatives))
            for alt in range(self.n_alternatives):
                if self.error_type == 'gumbel':
                    error_matrix[:, alt] = np.random.gumbel(0, self.scale, n_observations)
                elif self.error_type == 'normal':
                    error_matrix[:, alt] = np.random.normal(0, self.scale, n_observations)
                else:
                    error_matrix[:, alt] = np.random.normal(0, self.scale, n_observations)
        else:
            # Correlated errors using multivariate normal
            # Create correlation matrix
            corr_matrix = np.full((self.n_alternatives, self.n_alternatives), self.correlation)
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Generate correlated normal errors
            error_matrix = np.random.multivariate_normal(
                mean=np.zeros(self.n_alternatives),
                cov=corr_matrix * (self.scale**2),
                size=n_observations
            )
            
        # Create DataFrame
        columns = [f'error_alt_{i}' for i in range(self.n_alternatives)]
        return pd.DataFrame(error_matrix, columns=columns)
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about the multiple error component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'n_alternatives': self.n_alternatives,
            'error_type': self.error_type,
            'scale': self.scale,
            'correlation': self.correlation,
            'random_seed': self.random_seed,
            'has_error_matrix': self.error_matrix is not None
        })
        return info
