"""
SEM (Structural Equation Modeling) results loader for utility function module.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

from .base_loader import BaseDataLoader
from ..config.settings import (
    SEM_STRUCTURAL_PATHS_FILE, SEM_PATH_ANALYSIS_FILE, SEM_FIT_INDICES_FILE,
    PATH_ANALYSIS_DIR
)

logger = logging.getLogger(__name__)


class SEMResultsLoader(BaseDataLoader):
    """
    Loader for SEM (Structural Equation Modeling) results.
    
    Handles loading and processing of SEM analysis results including:
    - Structural path coefficients
    - Path analysis results
    - Model fit indices
    - Factor scores and relationships
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize SEM results loader.
        
        Args:
            data_dir: Directory containing SEM results files (defaults to config setting)
        """
        super().__init__(data_dir or PATH_ANALYSIS_DIR)
        self.structural_paths_file = SEM_STRUCTURAL_PATHS_FILE
        self.path_analysis_file = SEM_PATH_ANALYSIS_FILE
        self.fit_indices_file = SEM_FIT_INDICES_FILE
        
    def load_data(self) -> Dict[str, Any]:
        """
        Load all SEM results data.
        
        Returns:
            Dictionary containing:
            - 'structural_paths': Structural path coefficients
            - 'path_analysis': Complete path analysis results
            - 'fit_indices': Model fit indices
            - 'factor_effects': Processed factor effects for utility function
        """
        logger.info("Loading SEM results...")
        
        data = {}
        
        # Load structural paths
        data['structural_paths'] = self._load_structural_paths()
        
        # Load path analysis results
        data['path_analysis'] = self._load_path_analysis()
        
        # Load fit indices
        data['fit_indices'] = self._load_fit_indices()
        
        # Process factor effects for utility function
        data['factor_effects'] = self._process_factor_effects(data)
        
        logger.info("SEM results loading completed")
        return data
    
    def _load_structural_paths(self) -> pd.DataFrame:
        """
        Load structural path coefficients.
        
        Returns:
            DataFrame with structural path coefficients
        """
        logger.info("Loading structural path coefficients...")
        
        df = self._load_csv_file(self.structural_paths_file)
        
        # Validate required columns
        expected_columns = [
            'From_Variable', 'To_Variable', 'Path', 'Coefficient', 
            'Standard_Error', 'Z_Value', 'P_Value', 'Significance'
        ]
        if not self._validate_required_columns(df, expected_columns):
            raise ValueError("Structural paths data missing required columns")
            
        # Clean data
        df = self._clean_data(df)
        
        # Filter for significant paths only
        significant_paths = df[df['P_Value'] < 0.05].copy()
        logger.info(f"Found {len(significant_paths)} significant structural paths")
        
        return df
    
    def _load_path_analysis(self) -> pd.DataFrame:
        """
        Load complete path analysis results.
        
        Returns:
            DataFrame with path analysis results
        """
        logger.info("Loading path analysis results...")
        
        df = self._load_csv_file(self.path_analysis_file)
        df = self._clean_data(df)
        
        return df
    
    def _load_fit_indices(self) -> pd.DataFrame:
        """
        Load model fit indices.
        
        Returns:
            DataFrame with model fit indices
        """
        logger.info("Loading model fit indices...")
        
        df = self._load_csv_file(self.fit_indices_file)
        df = self._clean_data(df)
        
        return df
    
    def _process_factor_effects(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process SEM results to extract factor effects for utility function.
        
        Args:
            data: Dictionary containing loaded SEM data
            
        Returns:
            Dictionary with processed factor effects
        """
        logger.info("Processing factor effects for utility function...")
        
        structural_paths = data['structural_paths']
        
        # Extract key factor relationships for utility function
        factor_effects = {
            'health_benefits': self._extract_health_benefit_effects(structural_paths),
            'nutrition_knowledge': self._extract_nutrition_knowledge_effects(structural_paths),
            'perceived_price': self._extract_perceived_price_effects(structural_paths),
            'direct_effects': self._extract_direct_effects(structural_paths),
            'interaction_effects': self._extract_interaction_effects(structural_paths)
        }
        
        return factor_effects
    
    def _extract_health_benefit_effects(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract health benefit related effects.
        
        Args:
            df: Structural paths DataFrame
            
        Returns:
            Dictionary with health benefit effects
        """
        effects = {}
        
        # Health concern -> Perceived benefit
        health_benefit_path = df[
            (df['From_Variable'] == 'health_concern') & 
            (df['To_Variable'] == 'perceived_benefit')
        ]
        if not health_benefit_path.empty:
            effects['health_to_benefit'] = health_benefit_path.iloc[0]['Coefficient']
            
        # Perceived benefit -> Purchase intention
        benefit_purchase_path = df[
            (df['From_Variable'] == 'perceived_benefit') & 
            (df['To_Variable'] == 'purchase_intention')
        ]
        if not benefit_purchase_path.empty:
            effects['benefit_to_purchase'] = benefit_purchase_path.iloc[0]['Coefficient']
            
        return effects
    
    def _extract_nutrition_knowledge_effects(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract nutrition knowledge related effects.
        
        Args:
            df: Structural paths DataFrame
            
        Returns:
            Dictionary with nutrition knowledge effects
        """
        effects = {}
        
        # Nutrition knowledge -> Perceived benefit
        nutrition_benefit_path = df[
            (df['From_Variable'] == 'nutrition_knowledge') & 
            (df['To_Variable'] == 'perceived_benefit')
        ]
        if not nutrition_benefit_path.empty:
            effects['nutrition_to_benefit'] = nutrition_benefit_path.iloc[0]['Coefficient']
            
        # Nutrition knowledge -> Purchase intention
        nutrition_purchase_path = df[
            (df['From_Variable'] == 'nutrition_knowledge') & 
            (df['To_Variable'] == 'purchase_intention')
        ]
        if not nutrition_purchase_path.empty:
            effects['nutrition_to_purchase'] = nutrition_purchase_path.iloc[0]['Coefficient']
            
        return effects
    
    def _extract_perceived_price_effects(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract perceived price related effects.
        
        Args:
            df: Structural paths DataFrame
            
        Returns:
            Dictionary with perceived price effects
        """
        effects = {}
        
        # Perceived price -> Perceived benefit
        price_benefit_path = df[
            (df['From_Variable'] == 'perceived_price') & 
            (df['To_Variable'] == 'perceived_benefit')
        ]
        if not price_benefit_path.empty:
            effects['price_to_benefit'] = price_benefit_path.iloc[0]['Coefficient']
            
        # Perceived price -> Purchase intention
        price_purchase_path = df[
            (df['From_Variable'] == 'perceived_price') & 
            (df['To_Variable'] == 'purchase_intention')
        ]
        if not price_purchase_path.empty:
            effects['price_to_purchase'] = price_purchase_path.iloc[0]['Coefficient']
            
        return effects
    
    def _extract_direct_effects(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract direct effects between factors.
        
        Args:
            df: Structural paths DataFrame
            
        Returns:
            Dictionary with direct effects
        """
        effects = {}
        
        # Health concern -> Purchase intention
        health_purchase_path = df[
            (df['From_Variable'] == 'health_concern') & 
            (df['To_Variable'] == 'purchase_intention')
        ]
        if not health_purchase_path.empty:
            effects['health_to_purchase'] = health_purchase_path.iloc[0]['Coefficient']
            
        return effects
    
    def _extract_interaction_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract potential interaction effects.
        
        Args:
            df: Structural paths DataFrame
            
        Returns:
            Dictionary with interaction effects
        """
        # For now, return empty dict - can be extended for interaction terms
        return {}
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate loaded SEM results data.
        
        Args:
            data: Loaded SEM data
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating SEM results data...")
        
        try:
            # Check if required data exists
            required_keys = ['structural_paths', 'factor_effects']
            for key in required_keys:
                if key not in data or data[key] is None:
                    logger.error(f"Required SEM data missing: {key}")
                    return False
            
            # Validate structural paths
            structural_paths = data['structural_paths']
            if structural_paths.empty:
                logger.error("Structural paths data is empty")
                return False
                
            # Check for key factor effects
            factor_effects = data['factor_effects']
            required_factors = ['health_benefits', 'nutrition_knowledge', 'perceived_price']
            for factor in required_factors:
                if factor not in factor_effects:
                    logger.warning(f"Factor effects missing for: {factor}")
                    
            logger.info("SEM results validation passed")
            return True
            
        except Exception as e:
            logger.error(f"SEM results validation failed: {str(e)}")
            return False
    
    def get_summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics for SEM results.
        
        Args:
            data: Loaded SEM data
            
        Returns:
            Dictionary with summary statistics
        """
        if 'structural_paths' not in data:
            return {}
            
        df = data['structural_paths']
        
        summary = {
            'n_paths': len(df),
            'n_significant_paths': len(df[df['P_Value'] < 0.05]),
            'coefficient_stats': {
                'mean': df['Coefficient'].mean(),
                'std': df['Coefficient'].std(),
                'min': df['Coefficient'].min(),
                'max': df['Coefficient'].max()
            },
            'factors_involved': {
                'from_variables': df['From_Variable'].unique().tolist(),
                'to_variables': df['To_Variable'].unique().tolist()
            }
        }
        
        return summary
