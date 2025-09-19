"""
DCE (Discrete Choice Experiment) data loader for utility function module.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List

from .base_loader import BaseDataLoader
from ..config.settings import (
    DCE_RAW_DATA_FILE, DCE_ATTRIBUTE_DATA_FILE, DCE_CHOICE_MATRIX_FILE,
    DCE_DATA_DIR
)

logger = logging.getLogger(__name__)


class DCEDataLoader(BaseDataLoader):
    """
    Loader for DCE (Discrete Choice Experiment) data.
    
    Handles loading and preprocessing of DCE choice data including:
    - Raw choice responses
    - Attribute data (sugar type, health label, price)
    - Choice matrix data
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DCE data loader.
        
        Args:
            data_dir: Directory containing DCE data files (defaults to config setting)
        """
        super().__init__(data_dir or DCE_DATA_DIR)
        self.raw_data_file = DCE_RAW_DATA_FILE
        self.attribute_data_file = DCE_ATTRIBUTE_DATA_FILE
        self.choice_matrix_file = DCE_CHOICE_MATRIX_FILE
        
    def load_data(self) -> Dict[str, Any]:
        """
        Load all DCE data files.
        
        Returns:
            Dictionary containing:
            - 'raw_data': Raw DCE responses
            - 'attribute_data': Processed attribute data
            - 'choice_matrix': Choice matrix data
            - 'processed_data': Processed and cleaned DCE data
        """
        logger.info("Loading DCE data...")
        
        data = {}
        
        # Load raw DCE data
        data['raw_data'] = self._load_raw_dce_data()
        
        # Load attribute data
        data['attribute_data'] = self._load_attribute_data()
        
        # Load choice matrix if available
        if self.choice_matrix_file.exists():
            data['choice_matrix'] = self._load_choice_matrix()
        else:
            logger.warning("Choice matrix file not found, skipping...")
            data['choice_matrix'] = None
            
        # Process and combine data
        data['processed_data'] = self._process_dce_data(data)
        
        logger.info("DCE data loading completed")
        return data
    
    def _load_raw_dce_data(self) -> pd.DataFrame:
        """
        Load raw DCE response data.
        
        Returns:
            DataFrame with raw DCE responses
        """
        logger.info("Loading raw DCE data...")
        
        df = self._load_csv_file(self.raw_data_file)
        
        # Validate required columns
        expected_columns = ['no', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26']
        if not self._validate_required_columns(df, expected_columns):
            raise ValueError("Raw DCE data missing required columns")
            
        # Clean data
        df = self._clean_data(df)
        
        # Rename columns for clarity
        df = df.rename(columns={'no': 'respondent_id'})
        
        return df
    
    def _load_attribute_data(self) -> pd.DataFrame:
        """
        Load DCE attribute data.
        
        Returns:
            DataFrame with DCE attribute information
        """
        logger.info("Loading DCE attribute data...")
        
        df = self._load_csv_file(self.attribute_data_file)
        
        # Validate required columns
        expected_columns = [
            'respondent_id', 'question_id', 'chosen_sugar_type', 
            'chosen_health_label', 'chosen_price', 'chose_sugar_free',
            'chose_health_label', 'choice_value'
        ]
        if not self._validate_required_columns(df, expected_columns):
            raise ValueError("Attribute data missing required columns")
            
        # Clean data
        df = self._clean_data(df)
        
        return df
    
    def _load_choice_matrix(self) -> Optional[pd.DataFrame]:
        """
        Load DCE choice matrix data.
        
        Returns:
            DataFrame with choice matrix or None if not available
        """
        logger.info("Loading DCE choice matrix...")
        
        try:
            df = self._load_csv_file(self.choice_matrix_file)
            df = self._clean_data(df)
            return df
        except Exception as e:
            logger.warning(f"Could not load choice matrix: {str(e)}")
            return None
    
    def _process_dce_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process and combine DCE data for utility function calculation.
        
        Args:
            data: Dictionary containing loaded DCE data
            
        Returns:
            Processed DataFrame ready for utility calculation
        """
        logger.info("Processing DCE data...")
        
        attribute_data = data['attribute_data']
        
        # Create binary variables for utility function
        processed_df = attribute_data.copy()
        
        # Sugar presence (1 = sugar-free, 0 = regular sugar)
        processed_df['sugar_free'] = processed_df['chose_sugar_free']
        
        # Health label presence (1 = has label, 0 = no label)
        processed_df['health_label'] = processed_df['chose_health_label']
        
        # Price as continuous variable
        processed_df['price'] = processed_df['chosen_price']
        
        # Normalize price for utility calculation
        processed_df['price_normalized'] = (
            processed_df['price'] - processed_df['price'].mean()
        ) / processed_df['price'].std()
        
        # Create interaction terms
        processed_df['sugar_health_interaction'] = (
            processed_df['sugar_free'] * processed_df['health_label']
        )
        
        processed_df['sugar_price_interaction'] = (
            processed_df['sugar_free'] * processed_df['price_normalized']
        )
        
        processed_df['health_price_interaction'] = (
            processed_df['health_label'] * processed_df['price_normalized']
        )
        
        logger.info(f"Processed DCE data shape: {processed_df.shape}")
        return processed_df
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate loaded DCE data.
        
        Args:
            data: Loaded DCE data
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.info("Validating DCE data...")
        
        try:
            # Check if required data exists
            if 'processed_data' not in data or data['processed_data'] is None:
                logger.error("Processed DCE data is missing")
                return False
                
            df = data['processed_data']
            
            # Check for required columns
            required_columns = [
                'respondent_id', 'question_id', 'sugar_free', 
                'health_label', 'price', 'choice_value'
            ]
            if not self._validate_required_columns(df, required_columns):
                return False
                
            # Check data ranges
            if not df['sugar_free'].isin([0, 1]).all():
                logger.error("Invalid values in sugar_free column")
                return False
                
            if not df['health_label'].isin([0, 1]).all():
                logger.error("Invalid values in health_label column")
                return False
                
            if df['price'].min() <= 0:
                logger.error("Invalid price values (must be positive)")
                return False
                
            # Check for sufficient data
            if len(df) < 100:  # Minimum threshold
                logger.warning("DCE data may be insufficient for analysis")
                
            logger.info("DCE data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"DCE data validation failed: {str(e)}")
            return False
    
    def get_summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary statistics for DCE data.
        
        Args:
            data: Loaded DCE data
            
        Returns:
            Dictionary with summary statistics
        """
        if 'processed_data' not in data:
            return {}
            
        df = data['processed_data']
        
        summary = {
            'n_respondents': df['respondent_id'].nunique(),
            'n_choices': len(df),
            'sugar_free_rate': df['sugar_free'].mean(),
            'health_label_rate': df['health_label'].mean(),
            'price_stats': {
                'mean': df['price'].mean(),
                'std': df['price'].std(),
                'min': df['price'].min(),
                'max': df['price'].max()
            },
            'choice_distribution': df['choice_value'].value_counts().to_dict()
        }
        
        return summary
