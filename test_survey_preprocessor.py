"""
Unit tests for Survey Data Preprocessor Module

Tests are designed to work with the actual raw data file without creating separate test data.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import unittest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from survey_data_preprocessor import (
    FactorConfig, 
    DataLoader, 
    FactorGrouper, 
    SurveyDataPreprocessor
)


class TestFactorConfig(unittest.TestCase):
    """Test cases for FactorConfig class."""
    
    def test_get_all_factors(self):
        """Test getting all factor names."""
        factors = FactorConfig.get_all_factors()
        expected_factors = [
            'demographics_1', 'health_concern', 'perceived_benefit', 
            'purchase_intention', 'dce_variables', 'perceived_price', 
            'nutrition_knowledge', 'demographics_2'
        ]
        self.assertEqual(set(factors), set(expected_factors))
    
    def test_get_factor_questions(self):
        """Test getting questions for specific factors."""
        # Test health_concern factor
        health_questions = FactorConfig.get_factor_questions('health_concern')
        expected_health = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
        self.assertEqual(health_questions, expected_health)
        
        # Test demographics_1 factor
        demo1_questions = FactorConfig.get_factor_questions('demographics_1')
        expected_demo1 = ['q1', 'q2_1', 'q3', 'q4', 'q5']
        self.assertEqual(demo1_questions, expected_demo1)
        
        # Test nutrition_knowledge factor (range-based)
        nutrition_questions = FactorConfig.get_factor_questions('nutrition_knowledge')
        expected_nutrition = [f'q{i}' for i in range(30, 50)]
        self.assertEqual(nutrition_questions, expected_nutrition)
    
    def test_get_factor_questions_invalid(self):
        """Test getting questions for invalid factor."""
        with self.assertRaises(ValueError):
            FactorConfig.get_factor_questions('invalid_factor')
    
    def test_get_factor_description(self):
        """Test getting factor descriptions."""
        desc = FactorConfig.get_factor_description('health_concern')
        self.assertEqual(desc, '소비자의 건강관심도')
        
        with self.assertRaises(ValueError):
            FactorConfig.get_factor_description('invalid_factor')


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
    
    def test_init_valid_file(self):
        """Test DataLoader initialization with valid file."""
        loader = DataLoader(self.raw_data_path)
        self.assertEqual(loader.file_path, Path(self.raw_data_path))
        self.assertEqual(loader.sheet_name, 'DATA')
    
    def test_init_invalid_file(self):
        """Test DataLoader initialization with invalid file."""
        with self.assertRaises(FileNotFoundError):
            DataLoader("nonexistent_file.xlsx")
    
    def test_init_invalid_format(self):
        """Test DataLoader initialization with invalid file format."""
        with self.assertRaises(ValueError):
            DataLoader("test_file.txt")
    
    def test_load_data(self):
        """Test loading data from Excel file."""
        loader = DataLoader(self.raw_data_path)
        df = loader.load_data()
        
        # Check data shape (should be 300 rows, 58 columns)
        self.assertEqual(df.shape[0], 300)  # 300 respondents
        self.assertEqual(df.shape[1], 58)   # 58 columns including 'no'
        
        # Check essential columns exist
        self.assertIn('no', df.columns)
        self.assertIn('q1', df.columns)
        self.assertIn('q56', df.columns)
    
    def test_get_available_sheets(self):
        """Test getting available sheet names."""
        loader = DataLoader(self.raw_data_path)
        sheets = loader.get_available_sheets()
        
        expected_sheets = ['DATA', 'LABEL', 'CODE']
        self.assertEqual(set(sheets), set(expected_sheets))


class TestFactorGrouper(unittest.TestCase):
    """Test cases for FactorGrouper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
        
        # Load real data for testing
        loader = DataLoader(self.raw_data_path)
        self.data = loader.load_data()
        self.grouper = FactorGrouper(self.data)
    
    def test_init_with_data(self):
        """Test FactorGrouper initialization with data."""
        self.assertIsNotNone(self.grouper.data)
        self.assertEqual(self.grouper.data.shape, self.data.shape)
    
    def test_init_with_empty_data(self):
        """Test FactorGrouper initialization with empty data."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            FactorGrouper(empty_df)
    
    def test_group_by_factor_health_concern(self):
        """Test grouping by health_concern factor."""
        health_data = self.grouper.group_by_factor('health_concern')
        
        # Check columns
        expected_cols = ['no', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']
        self.assertEqual(set(health_data.columns), set(expected_cols))
        
        # Check data shape
        self.assertEqual(health_data.shape[0], 300)  # Same number of rows
        self.assertEqual(health_data.shape[1], 7)    # 6 questions + no column
    
    def test_group_by_factor_demographics_1(self):
        """Test grouping by demographics_1 factor."""
        demo_data = self.grouper.group_by_factor('demographics_1')
        
        # Check columns
        expected_cols = ['no', 'q1', 'q2_1', 'q3', 'q4', 'q5']
        self.assertEqual(set(demo_data.columns), set(expected_cols))
        
        # Check data shape
        self.assertEqual(demo_data.shape[0], 300)
        self.assertEqual(demo_data.shape[1], 6)
    
    def test_group_by_factor_invalid(self):
        """Test grouping by invalid factor."""
        with self.assertRaises(ValueError):
            self.grouper.group_by_factor('invalid_factor')
    
    def test_get_all_factors_data(self):
        """Test getting data for all factors."""
        all_factors_data = self.grouper.get_all_factors_data()
        
        # Check that we get data for all factors
        expected_factors = FactorConfig.get_all_factors()
        self.assertEqual(set(all_factors_data.keys()), set(expected_factors))
        
        # Check each factor data
        for factor_name, factor_data in all_factors_data.items():
            self.assertIsInstance(factor_data, pd.DataFrame)
            self.assertEqual(factor_data.shape[0], 300)  # Same number of rows
            self.assertIn('no', factor_data.columns)     # Should include ID column
    
    def test_get_factor_summary(self):
        """Test getting factor summary."""
        summary = self.grouper.get_factor_summary()
        
        # Check summary structure
        expected_columns = ['factor_name', 'description', 'total_questions', 'available_questions', 'questions']
        self.assertEqual(set(summary.columns), set(expected_columns))
        
        # Check number of factors
        self.assertEqual(len(summary), len(FactorConfig.get_all_factors()))
        
        # Check specific factor
        health_row = summary[summary['factor_name'] == 'health_concern'].iloc[0]
        self.assertEqual(health_row['total_questions'], 6)
        self.assertEqual(health_row['available_questions'], 6)


class TestSurveyDataPreprocessor(unittest.TestCase):
    """Test cases for SurveyDataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
        
        self.preprocessor = SurveyDataPreprocessor(self.raw_data_path)
    
    def test_init(self):
        """Test SurveyDataPreprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.loader)
        self.assertIsNone(self.preprocessor.data)
        self.assertIsNone(self.preprocessor.grouper)
    
    def test_load_and_prepare_data(self):
        """Test loading and preparing data."""
        self.preprocessor.load_and_prepare_data()
        
        self.assertIsNotNone(self.preprocessor.data)
        self.assertIsNotNone(self.preprocessor.grouper)
        self.assertEqual(self.preprocessor.data.shape[0], 300)
    
    def test_get_factor_data_before_loading(self):
        """Test getting factor data before loading."""
        with self.assertRaises(RuntimeError):
            self.preprocessor.get_factor_data('health_concern')
    
    def test_get_factor_data_after_loading(self):
        """Test getting factor data after loading."""
        self.preprocessor.load_and_prepare_data()
        
        health_data = self.preprocessor.get_factor_data('health_concern')
        self.assertEqual(health_data.shape[0], 300)
        self.assertIn('q6', health_data.columns)
    
    def test_get_all_factors_data(self):
        """Test getting all factors data."""
        self.preprocessor.load_and_prepare_data()
        
        all_data = self.preprocessor.get_all_factors_data()
        self.assertEqual(len(all_data), len(FactorConfig.get_all_factors()))
    
    def test_get_summary(self):
        """Test getting summary."""
        self.preprocessor.load_and_prepare_data()
        
        summary = self.preprocessor.get_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), len(FactorConfig.get_all_factors()))
    
    def test_export_factor_data(self):
        """Test exporting factor data."""
        self.preprocessor.load_and_prepare_data()
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            self.preprocessor.export_factor_data(temp_dir)
            
            # Check that files were created
            temp_path = Path(temp_dir)
            csv_files = list(temp_path.glob("*.csv"))
            
            # Should have one file per factor plus summary
            expected_files = len(FactorConfig.get_all_factors()) + 1  # +1 for summary
            self.assertEqual(len(csv_files), expected_files)
            
            # Check specific files exist
            self.assertTrue((temp_path / "health_concern.csv").exists())
            self.assertTrue((temp_path / "factors_summary.csv").exists())


class TestIntegration(unittest.TestCase):
    """Integration tests using real data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
    
    def test_full_preprocessing_workflow(self):
        """Test the complete preprocessing workflow."""
        # Initialize preprocessor
        preprocessor = SurveyDataPreprocessor(self.raw_data_path)
        
        # Load and prepare data
        preprocessor.load_and_prepare_data()
        
        # Get summary
        summary = preprocessor.get_summary()
        self.assertGreater(len(summary), 0)
        
        # Get specific factor data
        health_data = preprocessor.get_factor_data('health_concern')
        self.assertEqual(health_data.shape[0], 300)
        
        # Get all factors data
        all_data = preprocessor.get_all_factors_data()
        self.assertEqual(len(all_data), 8)  # 8 factors
        
        # Test data integrity
        for factor_name, factor_data in all_data.items():
            self.assertEqual(factor_data.shape[0], 300)
            self.assertIn('no', factor_data.columns)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
