"""
Base factor integrator for SEM results integration.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class FactorIntegrator(ABC):
    """
    Abstract base class for integrating SEM factor results into utility functions.
    
    Each integrator handles a specific latent factor from SEM analysis
    and its effects on utility.
    """
    
    def __init__(self, factor_name: str):
        """
        Initialize factor integrator.
        
        Args:
            factor_name: Name of the latent factor (e.g., 'health_concern', 'perceived_benefit')
        """
        self.factor_name = factor_name
        self.factor_scores = None
        self.path_coefficients = {}
        self.is_fitted = False
        
    @abstractmethod
    def integrate_factor_effects(self, dce_data: pd.DataFrame, 
                                sem_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate SEM factor effects into DCE data for utility calculation.
        
        Args:
            dce_data: DCE choice data
            sem_results: SEM analysis results
            
        Returns:
            Enhanced DCE data with factor effects
        """
        pass
    
    @abstractmethod
    def calculate_factor_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from this factor.
        
        Args:
            data: Data with factor information
            **kwargs: Additional parameters
            
        Returns:
            Series with factor utility values
        """
        pass
    
    def load_factor_scores(self, survey_data: pd.DataFrame, 
                          factor_file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load factor scores for individuals.
        
        Args:
            survey_data: Survey data with factor items
            factor_file_path: Optional path to pre-calculated factor scores
            
        Returns:
            DataFrame with factor scores
        """
        if factor_file_path and pd.io.common.file_exists(factor_file_path):
            logger.info(f"Loading pre-calculated factor scores from {factor_file_path}")
            factor_scores = pd.read_csv(factor_file_path)
        else:
            logger.info(f"Calculating factor scores for {self.factor_name}")
            factor_scores = self._calculate_factor_scores(survey_data)
            
        self.factor_scores = factor_scores
        return factor_scores
    
    def _calculate_factor_scores(self, survey_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor scores from survey items.
        
        Args:
            survey_data: Survey data with factor items
            
        Returns:
            DataFrame with calculated factor scores
        """
        # Simple mean-based factor score calculation
        # In practice, this could use factor loadings from SEM results
        
        factor_columns = self._get_factor_columns(survey_data)
        if not factor_columns:
            logger.warning(f"No columns found for factor {self.factor_name}")
            return pd.DataFrame()
            
        # Calculate mean score
        factor_scores = survey_data[factor_columns].mean(axis=1)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'respondent_id': survey_data.get('respondent_id', survey_data.index),
            f'{self.factor_name}_score': factor_scores
        })
        
        return result_df
    
    def _get_factor_columns(self, survey_data: pd.DataFrame) -> List[str]:
        """
        Get column names for this factor.
        
        Args:
            survey_data: Survey data
            
        Returns:
            List of column names for this factor
        """
        # This should be overridden by specific factor integrators
        # or use a mapping from configuration
        return []
    
    def extract_path_coefficients(self, sem_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract relevant path coefficients from SEM results.
        
        Args:
            sem_results: SEM analysis results
            
        Returns:
            Dictionary with path coefficients
        """
        if 'structural_paths' not in sem_results:
            logger.warning("No structural paths found in SEM results")
            return {}
            
        structural_paths = sem_results['structural_paths']
        
        # Extract paths involving this factor
        factor_paths = structural_paths[
            (structural_paths['From_Variable'] == self.factor_name) |
            (structural_paths['To_Variable'] == self.factor_name)
        ]
        
        coefficients = {}
        for _, row in factor_paths.iterrows():
            path_name = f"{row['From_Variable']}_to_{row['To_Variable']}"
            coefficients[path_name] = row['Coefficient']
            
        self.path_coefficients = coefficients
        logger.info(f"Extracted {len(coefficients)} path coefficients for {self.factor_name}")
        
        return coefficients
    
    def validate_factor_data(self, data: pd.DataFrame) -> bool:
        """
        Validate factor data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_column = f'{self.factor_name}_score'
        
        if required_column not in data.columns:
            logger.error(f"Factor score column '{required_column}' not found")
            return False
            
        if data[required_column].isnull().all():
            logger.error(f"All factor scores are null for {self.factor_name}")
            return False
            
        return True
    
    def get_factor_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the factor.
        
        Args:
            data: Data with factor scores
            
        Returns:
            Dictionary with summary statistics
        """
        score_column = f'{self.factor_name}_score'
        
        if score_column not in data.columns:
            return {}
            
        scores = data[score_column].dropna()
        
        summary = {
            'factor_name': self.factor_name,
            'n_observations': len(scores),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'percentiles': {
                '25th': scores.quantile(0.25),
                '50th': scores.quantile(0.50),
                '75th': scores.quantile(0.75)
            }
        }
        
        return summary
    
    def standardize_factor_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize factor scores (mean=0, std=1).
        
        Args:
            data: Data with factor scores
            
        Returns:
            Data with standardized factor scores
        """
        data = data.copy()
        score_column = f'{self.factor_name}_score'
        
        if score_column in data.columns:
            scores = data[score_column]
            standardized_scores = (scores - scores.mean()) / scores.std()
            data[f'{self.factor_name}_score_std'] = standardized_scores
            
        return data
    
    def get_integrator_info(self) -> Dict[str, Any]:
        """
        Get information about the integrator.
        
        Returns:
            Dictionary with integrator information
        """
        return {
            'factor_name': self.factor_name,
            'is_fitted': self.is_fitted,
            'n_path_coefficients': len(self.path_coefficients),
            'path_coefficients': self.path_coefficients,
            'has_factor_scores': self.factor_scores is not None
        }
