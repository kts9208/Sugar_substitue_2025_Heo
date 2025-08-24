"""
Survey Data Preprocessor Module

This module provides functionality to load and preprocess survey data from Excel files,
grouping questions by factors for statistical analysis.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorConfig:
    """Configuration class for factor definitions and question groupings."""
    
    # Factor definitions with question ranges
    FACTOR_DEFINITIONS = {
        'demographics_1': {'questions': ['q1', 'q2_1', 'q3', 'q4', 'q5'], 'description': '인구통계학적 변수 (1차)'},
        'health_concern': {'questions': ['q6', 'q7', 'q8', 'q9', 'q10', 'q11'], 'description': '소비자의 건강관심도'},
        'perceived_benefit': {'questions': ['q12', 'q13', 'q14', 'q15', 'q16', 'q17'], 'description': 'substitute의 지각된 유익성'},
        'purchase_intention': {'questions': ['q18', 'q19', 'q20'], 'description': 'substitute의 구매의도'},
        'dce_variables': {'questions': ['q21', 'q22', 'q23', 'q24', 'q25', 'q26'], 'description': 'DCE 변수 (추후 별도 처리예정)'},
        'perceived_price': {'questions': ['q27', 'q28', 'q29'], 'description': '인지된 가격수준'},
        'nutrition_knowledge': {'questions': [f'q{i}' for i in range(30, 50)], 'description': '소비자의 영양지식 수준'},
        'demographics_2': {'questions': ['q50', 'q51', 'q51_14', 'q52', 'q53', 'q54', 'q55', 'q56'], 'description': '인구통계학적 변수 (2차)'}
    }
    
    @classmethod
    def get_all_factors(cls) -> List[str]:
        """Get list of all factor names."""
        return list(cls.FACTOR_DEFINITIONS.keys())
    
    @classmethod
    def get_factor_questions(cls, factor_name: str) -> List[str]:
        """Get questions for a specific factor."""
        if factor_name not in cls.FACTOR_DEFINITIONS:
            raise ValueError(f"Unknown factor: {factor_name}")
        return cls.FACTOR_DEFINITIONS[factor_name]['questions']
    
    @classmethod
    def get_factor_description(cls, factor_name: str) -> str:
        """Get description for a specific factor."""
        if factor_name not in cls.FACTOR_DEFINITIONS:
            raise ValueError(f"Unknown factor: {factor_name}")
        return cls.FACTOR_DEFINITIONS[factor_name]['description']


class DataLoader:
    """Handles loading and basic preprocessing of survey data from Excel files."""
    
    def __init__(self, file_path: str, sheet_name: str = 'DATA'):
        """
        Initialize DataLoader.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet containing data (default: 'DATA')
        """
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that the file exists and is accessible."""
        if not self.file_path.suffix.lower() in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Returns:
            DataFrame containing the survey data
        """
        try:
            logger.info(f"Loading data from {self.file_path}, sheet: {self.sheet_name}")
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_available_sheets(self) -> List[str]:
        """Get list of available sheet names in the Excel file."""
        try:
            xl_file = pd.ExcelFile(self.file_path)
            return xl_file.sheet_names
        except Exception as e:
            logger.error(f"Error reading sheet names: {e}")
            raise


class FactorGrouper:
    """Handles grouping of survey questions by factors for statistical analysis."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize FactorGrouper.
        
        Args:
            data: DataFrame containing survey data
        """
        self.data = data.copy()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate that the data contains expected columns."""
        if self.data.empty:
            raise ValueError("Data is empty")
        
        # Check if essential columns exist
        missing_questions = []
        for factor in FactorConfig.get_all_factors():
            questions = FactorConfig.get_factor_questions(factor)
            for question in questions:
                if question not in self.data.columns:
                    missing_questions.append(question)
        
        if missing_questions:
            logger.warning(f"Missing questions in data: {missing_questions}")
    
    def group_by_factor(self, factor_name: str) -> pd.DataFrame:
        """
        Group questions by a specific factor.
        
        Args:
            factor_name: Name of the factor to group
            
        Returns:
            DataFrame containing only the questions for the specified factor
        """
        if factor_name not in FactorConfig.get_all_factors():
            raise ValueError(f"Unknown factor: {factor_name}")
        
        questions = FactorConfig.get_factor_questions(factor_name)
        available_questions = [q for q in questions if q in self.data.columns]
        
        if not available_questions:
            raise ValueError(f"No questions found for factor: {factor_name}")
        
        # Include 'no' column if it exists for identification
        columns_to_include = ['no'] if 'no' in self.data.columns else []
        columns_to_include.extend(available_questions)
        
        return self.data[columns_to_include].copy()
    
    def get_all_factors_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data grouped by all factors.
        
        Returns:
            Dictionary with factor names as keys and DataFrames as values
        """
        factors_data = {}
        for factor in FactorConfig.get_all_factors():
            try:
                factors_data[factor] = self.group_by_factor(factor)
                logger.info(f"Factor '{factor}' grouped successfully. Shape: {factors_data[factor].shape}")
            except ValueError as e:
                logger.warning(f"Skipping factor '{factor}': {e}")
        
        return factors_data
    
    def get_factor_summary(self) -> pd.DataFrame:
        """
        Get summary information about all factors.
        
        Returns:
            DataFrame containing factor information
        """
        summary_data = []
        for factor in FactorConfig.get_all_factors():
            questions = FactorConfig.get_factor_questions(factor)
            available_questions = [q for q in questions if q in self.data.columns]
            
            summary_data.append({
                'factor_name': factor,
                'description': FactorConfig.get_factor_description(factor),
                'total_questions': len(questions),
                'available_questions': len(available_questions),
                'questions': ', '.join(available_questions)
            })
        
        return pd.DataFrame(summary_data)


class SurveyDataPreprocessor:
    """Main class for survey data preprocessing and factor analysis preparation."""
    
    def __init__(self, file_path: str, sheet_name: str = 'DATA'):
        """
        Initialize SurveyDataPreprocessor.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet containing data
        """
        self.loader = DataLoader(file_path, sheet_name)
        self.data = None
        self.grouper = None
    
    def load_and_prepare_data(self) -> None:
        """Load data and prepare for factor analysis."""
        self.data = self.loader.load_data()
        self.grouper = FactorGrouper(self.data)
        logger.info("Data loaded and prepared for factor analysis")
    
    def get_factor_data(self, factor_name: str) -> pd.DataFrame:
        """
        Get data for a specific factor.
        
        Args:
            factor_name: Name of the factor
            
        Returns:
            DataFrame containing factor data
        """
        if self.grouper is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.grouper.group_by_factor(factor_name)
    
    def get_all_factors_data(self) -> Dict[str, pd.DataFrame]:
        """Get data for all factors."""
        if self.grouper is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.grouper.get_all_factors_data()
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all factors."""
        if self.grouper is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.grouper.get_factor_summary()
    
    def export_factor_data(self, output_dir: str = 'processed_data') -> None:
        """
        Export factor data to separate CSV files.
        
        Args:
            output_dir: Directory to save the processed files
        """
        if self.grouper is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        factors_data = self.get_all_factors_data()
        
        for factor_name, factor_data in factors_data.items():
            file_path = output_path / f"{factor_name}.csv"
            factor_data.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"Factor '{factor_name}' exported to {file_path}")
        
        # Export summary
        summary = self.get_summary()
        summary_path = output_path / "factors_summary.csv"
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"Summary exported to {summary_path}")


# Example usage
if __name__ == "__main__":
    # Example usage of the preprocessor
    file_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
    
    try:
        # Initialize preprocessor
        preprocessor = SurveyDataPreprocessor(file_path)
        
        # Load and prepare data
        preprocessor.load_and_prepare_data()
        
        # Get summary
        summary = preprocessor.get_summary()
        print("Factor Summary:")
        print(summary)
        
        # Get specific factor data
        health_concern_data = preprocessor.get_factor_data('health_concern')
        print(f"\nHealth Concern Factor Data Shape: {health_concern_data.shape}")
        print(health_concern_data.head())
        
        # Export all factor data
        preprocessor.export_factor_data()
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
