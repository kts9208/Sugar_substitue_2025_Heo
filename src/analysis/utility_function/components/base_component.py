"""
Base utility function component class.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class BaseUtilityComponent(ABC):
    """
    Abstract base class for utility function components.
    
    Each component represents a specific part of the utility function
    (e.g., sugar presence, health label, price effects).
    """
    
    def __init__(self, name: str, coefficient: float = 0.0):
        """
        Initialize the utility component.
        
        Args:
            name: Name of the component
            coefficient: Base coefficient for the component
        """
        self.name = name
        self.coefficient = coefficient
        self.is_fitted = False
        self.component_data = {}
        
    @abstractmethod
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution for this component.
        
        Args:
            data: Input data for utility calculation
            **kwargs: Additional parameters
            
        Returns:
            Series with utility values for each observation
        """
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseUtilityComponent':
        """
        Fit the component parameters to data.
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    def validate_input_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate input data for the component.
        
        Args:
            data: Input data to validate
            required_columns: List of required column names
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            logger.error(f"{self.name}: Input data is empty")
            return False
            
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            logger.error(f"{self.name}: Missing required columns: {missing_columns}")
            return False
            
        return True
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about the component.
        
        Returns:
            Dictionary with component information
        """
        return {
            'name': self.name,
            'coefficient': self.coefficient,
            'is_fitted': self.is_fitted,
            'component_type': self.__class__.__name__
        }
    
    def set_coefficient(self, coefficient: float) -> 'BaseUtilityComponent':
        """
        Set the component coefficient.
        
        Args:
            coefficient: New coefficient value
            
        Returns:
            Self for method chaining
        """
        self.coefficient = coefficient
        return self
    
    def get_coefficient(self) -> float:
        """
        Get the component coefficient.
        
        Returns:
            Current coefficient value
        """
        return self.coefficient
    
    def reset(self) -> 'BaseUtilityComponent':
        """
        Reset the component to initial state.
        
        Returns:
            Self for method chaining
        """
        self.coefficient = 0.0
        self.is_fitted = False
        self.component_data = {}
        return self
    
    def __str__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}(name='{self.name}', coefficient={self.coefficient})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the component."""
        return self.__str__()
