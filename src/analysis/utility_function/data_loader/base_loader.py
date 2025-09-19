"""
Base data loader class for utility function module.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union

from ..config.settings import LOG_LEVEL, LOG_FORMAT

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders in the utility function module.
    
    Provides common functionality for loading and validating data files.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the base data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.data_cache = {}
        
    def _validate_file_exists(self, file_path: Path) -> bool:
        """
        Validate that a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        return True
    
    def _load_csv_file(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        if not self._validate_file_exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        try:
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path.name}")
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"Empty data file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def _validate_required_columns(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that DataFrame contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if all required columns exist, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning operations.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Log data quality info
        logger.info(f"Data shape after cleaning: {df.shape}")
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            
        return df
    
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """
        Load and return data.
        
        Returns:
            Dictionary containing loaded data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
