"""
Unit tests for DCE (Discrete Choice Experiment) Preprocessor Module

Tests are designed to work with the actual raw data file without creating separate test data.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from dce_preprocessor import (
    SugarType, HealthLabel, Choice, ProductAttributes, ChoiceSet,
    DCEConfig, DCEDataLoader, DCEPreprocessor, DCEAnalyzer, DCEProcessor
)


class TestEnumsAndDataClasses(unittest.TestCase):
    """Test cases for enums and data classes."""
    
    def test_sugar_type_enum(self):
        """Test SugarType enum."""
        self.assertEqual(SugarType.REGULAR.value, "일반당")
        self.assertEqual(SugarType.SUGAR_FREE.value, "무설탕")
    
    def test_health_label_enum(self):
        """Test HealthLabel enum."""
        self.assertEqual(HealthLabel.WITH_LABEL.value, "건강라벨 있음")
        self.assertEqual(HealthLabel.WITHOUT_LABEL.value, "건강라벨 없음")
    
    def test_choice_enum(self):
        """Test Choice enum."""
        self.assertEqual(Choice.PRODUCT_A.value, 1)
        self.assertEqual(Choice.PRODUCT_B.value, 2)
        self.assertEqual(Choice.NEITHER.value, 3)
    
    def test_product_attributes(self):
        """Test ProductAttributes dataclass."""
        product = ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITH_LABEL, 2500)
        self.assertEqual(product.sugar_type, SugarType.SUGAR_FREE)
        self.assertEqual(product.health_label, HealthLabel.WITH_LABEL)
        self.assertEqual(product.price, 2500)
        
        expected_str = "무설탕, 건강라벨 있음, 2500원"
        self.assertEqual(str(product), expected_str)
    
    def test_choice_set(self):
        """Test ChoiceSet dataclass."""
        product_a = ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITH_LABEL, 2500)
        product_b = ProductAttributes(SugarType.REGULAR, HealthLabel.WITHOUT_LABEL, 2000)
        choice_set = ChoiceSet("q21", product_a, product_b)
        
        self.assertEqual(choice_set.question_id, "q21")
        self.assertEqual(choice_set.product_a, product_a)
        self.assertEqual(choice_set.product_b, product_b)


class TestDCEConfig(unittest.TestCase):
    """Test cases for DCEConfig class."""
    
    def test_get_all_questions(self):
        """Test getting all DCE question IDs."""
        questions = DCEConfig.get_all_questions()
        expected_questions = ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']
        self.assertEqual(set(questions), set(expected_questions))
    
    def test_get_choice_set(self):
        """Test getting specific choice sets."""
        # Test q21
        q21_set = DCEConfig.get_choice_set('q21')
        self.assertEqual(q21_set.question_id, 'q21')
        self.assertEqual(q21_set.product_a.sugar_type, SugarType.SUGAR_FREE)
        self.assertEqual(q21_set.product_a.health_label, HealthLabel.WITH_LABEL)
        self.assertEqual(q21_set.product_a.price, 2500)
        self.assertEqual(q21_set.product_b.sugar_type, SugarType.REGULAR)
        self.assertEqual(q21_set.product_b.health_label, HealthLabel.WITHOUT_LABEL)
        self.assertEqual(q21_set.product_b.price, 2000)
        
        # Test q22
        q22_set = DCEConfig.get_choice_set('q22')
        self.assertEqual(q22_set.product_a.sugar_type, SugarType.REGULAR)
        self.assertEqual(q22_set.product_b.sugar_type, SugarType.SUGAR_FREE)
    
    def test_get_choice_set_invalid(self):
        """Test getting invalid choice set."""
        with self.assertRaises(ValueError):
            DCEConfig.get_choice_set('invalid_question')
    
    def test_get_choice_sets_summary(self):
        """Test getting choice sets summary."""
        summary = DCEConfig.get_choice_sets_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 6)  # 6 questions
        
        expected_columns = [
            'question_id', 'product_a_sugar', 'product_a_label', 'product_a_price',
            'product_b_sugar', 'product_b_label', 'product_b_price'
        ]
        self.assertEqual(set(summary.columns), set(expected_columns))


class TestDCEDataLoader(unittest.TestCase):
    """Test cases for DCEDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "../../Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
    
    def test_init_valid_file(self):
        """Test DCEDataLoader initialization with valid file."""
        loader = DCEDataLoader(self.raw_data_path)
        self.assertEqual(loader.file_path, Path(self.raw_data_path))
        self.assertEqual(loader.sheet_name, 'DATA')
        self.assertEqual(len(loader.dce_questions), 6)
    
    def test_init_invalid_file(self):
        """Test DCEDataLoader initialization with invalid file."""
        with self.assertRaises(FileNotFoundError):
            DCEDataLoader("nonexistent_file.xlsx")
    
    def test_load_dce_data(self):
        """Test loading DCE data."""
        loader = DCEDataLoader(self.raw_data_path)
        dce_data = loader.load_dce_data()
        
        # Check data structure
        self.assertEqual(dce_data.shape[0], 300)  # 300 respondents (numbering corrected)
        self.assertEqual(dce_data.shape[1], 7)    # 6 DCE questions + no column
        
        # Check columns
        expected_columns = ['no', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26']
        self.assertEqual(set(dce_data.columns), set(expected_columns))
        
        # Check data types and values
        for question in ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']:
            unique_values = set(dce_data[question].dropna().unique())
            self.assertTrue(unique_values.issubset({1, 2, 3}))
    
    def test_validate_dce_responses(self):
        """Test DCE response validation."""
        loader = DCEDataLoader(self.raw_data_path)
        dce_data = loader.load_dce_data()
        validation_results = loader.validate_dce_responses(dce_data)
        
        # Check validation structure
        self.assertIn('total_responses', validation_results)
        self.assertIn('valid_responses', validation_results)
        self.assertIn('question_stats', validation_results)
        
        # Check that we have stats for all questions
        self.assertEqual(len(validation_results['question_stats']), 6)
        
        # Check individual question stats
        for question in ['q21', 'q22', 'q23', 'q24', 'q25', 'q26']:
            self.assertIn(question, validation_results['question_stats'])
            q_stats = validation_results['question_stats'][question]
            self.assertIn('valid_count', q_stats)
            self.assertIn('choice_distribution', q_stats)


class TestDCEPreprocessor(unittest.TestCase):
    """Test cases for DCEPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
        
        # Load real DCE data for testing
        loader = DCEDataLoader(self.raw_data_path)
        self.dce_data = loader.load_dce_data()
        self.preprocessor = DCEPreprocessor(self.dce_data)
    
    def test_init_with_data(self):
        """Test DCEPreprocessor initialization with data."""
        self.assertIsNotNone(self.preprocessor.dce_data)
        self.assertEqual(self.preprocessor.dce_data.shape, self.dce_data.shape)
        self.assertEqual(len(self.preprocessor.dce_questions), 6)
    
    def test_init_with_empty_data(self):
        """Test DCEPreprocessor initialization with empty data."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            DCEPreprocessor(empty_df)
    
    def test_create_choice_matrix(self):
        """Test creating choice matrix."""
        choice_matrix = self.preprocessor.create_choice_matrix()
        
        # Check structure
        expected_rows = 300 * 6 * 3  # 300 respondents * 6 questions * 3 alternatives
        self.assertEqual(len(choice_matrix), expected_rows)
        
        # Check columns
        expected_columns = [
            'respondent_id', 'question_id', 'alternative', 'chosen',
            'sugar_type', 'health_label', 'price', 'sugar_free', 'has_health_label'
        ]
        self.assertEqual(set(choice_matrix.columns), set(expected_columns))
        
        # Check that each respondent has exactly one choice per question
        for respondent_id in choice_matrix['respondent_id'].unique():
            for question_id in choice_matrix['question_id'].unique():
                respondent_question_data = choice_matrix[
                    (choice_matrix['respondent_id'] == respondent_id) & 
                    (choice_matrix['question_id'] == question_id)
                ]
                chosen_count = respondent_question_data['chosen'].sum()
                self.assertEqual(chosen_count, 1)  # Exactly one choice per question
    
    def test_create_attribute_effects_data(self):
        """Test creating attribute effects data."""
        attribute_data = self.preprocessor.create_attribute_effects_data()
        
        # Check structure
        self.assertGreater(len(attribute_data), 0)
        
        # Check columns
        expected_columns = [
            'respondent_id', 'question_id', 'chosen_sugar_type', 'chosen_health_label',
            'chosen_price', 'chose_sugar_free', 'chose_health_label', 'choice_value'
        ]
        self.assertEqual(set(attribute_data.columns), set(expected_columns))
        
        # Check that choice_value is only 1 or 2 (excluding "neither")
        unique_choices = set(attribute_data['choice_value'].unique())
        self.assertTrue(unique_choices.issubset({1, 2}))
    
    def test_get_choice_summary(self):
        """Test getting choice summary."""
        summary = self.preprocessor.get_choice_summary()
        
        # Check structure
        self.assertEqual(len(summary), 6)  # 6 questions
        
        # Check columns
        expected_columns = [
            'question_id', 'total_responses', 'product_a_choices', 'product_b_choices',
            'neither_choices', 'product_a_share', 'product_b_share', 'neither_share',
            'product_a_description', 'product_b_description'
        ]
        self.assertEqual(set(summary.columns), set(expected_columns))
        
        # Check that shares sum to approximately 100% for each question
        for _, row in summary.iterrows():
            total_share = row['product_a_share'] + row['product_b_share'] + row['neither_share']
            self.assertAlmostEqual(total_share, 100.0, places=1)


class TestDCEAnalyzer(unittest.TestCase):
    """Test cases for DCEAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
        
        # Load real DCE data for testing
        loader = DCEDataLoader(self.raw_data_path)
        dce_data = loader.load_dce_data()
        preprocessor = DCEPreprocessor(dce_data)
        self.analyzer = DCEAnalyzer(preprocessor)
    
    def test_prepare_analysis_data(self):
        """Test preparing analysis data."""
        self.analyzer.prepare_analysis_data()
        
        self.assertIsNotNone(self.analyzer.choice_matrix)
        self.assertIsNotNone(self.analyzer.attribute_data)
        
        # Check choice matrix structure
        self.assertGreater(len(self.analyzer.choice_matrix), 0)
        self.assertIn('chosen', self.analyzer.choice_matrix.columns)
        
        # Check attribute data structure
        self.assertGreater(len(self.analyzer.attribute_data), 0)
        self.assertIn('chose_sugar_free', self.analyzer.attribute_data.columns)
    
    def test_analyze_attribute_preferences(self):
        """Test attribute preference analysis."""
        analysis_results = self.analyzer.analyze_attribute_preferences()
        
        # Check structure
        self.assertIn('sugar_free_preference', analysis_results)
        self.assertIn('health_label_preference', analysis_results)
        self.assertIn('price_analysis', analysis_results)
        
        # Check sugar-free preference
        sugar_pref = analysis_results['sugar_free_preference']
        self.assertIn('preference_rate', sugar_pref)
        self.assertIn('sugar_free_choices', sugar_pref)
        self.assertIn('regular_sugar_choices', sugar_pref)
        self.assertGreaterEqual(sugar_pref['preference_rate'], 0)
        self.assertLessEqual(sugar_pref['preference_rate'], 1)
        
        # Check health label preference
        label_pref = analysis_results['health_label_preference']
        self.assertIn('preference_rate', label_pref)
        self.assertGreaterEqual(label_pref['preference_rate'], 0)
        self.assertLessEqual(label_pref['preference_rate'], 1)
        
        # Check price analysis
        price_analysis = analysis_results['price_analysis']
        self.assertIn('mean_chosen_price', price_analysis)
        self.assertIn('median_chosen_price', price_analysis)
        self.assertIn('price_distribution', price_analysis)
    
    def test_get_choice_patterns(self):
        """Test choice pattern analysis."""
        choice_patterns = self.analyzer.get_choice_patterns()
        
        # Check structure
        self.assertIsInstance(choice_patterns, pd.DataFrame)
        self.assertGreater(len(choice_patterns), 0)
        
        # Check columns
        expected_columns = ['respondent_id', 'total_choices', 'sugar_free_choices', 'health_label_choices', 'avg_chosen_price']
        self.assertEqual(set(choice_patterns.columns), set(expected_columns))
        
        # Check that we have data for all respondents who made choices
        unique_respondents = len(choice_patterns['respondent_id'].unique())
        self.assertGreater(unique_respondents, 0)


class TestDCEProcessor(unittest.TestCase):
    """Test cases for DCEProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
        
        self.processor = DCEProcessor(self.raw_data_path)
    
    def test_init(self):
        """Test DCEProcessor initialization."""
        self.assertIsNotNone(self.processor.loader)
        self.assertIsNone(self.processor.data)
        self.assertIsNone(self.processor.preprocessor)
        self.assertIsNone(self.processor.analyzer)
    
    def test_load_and_prepare_data(self):
        """Test loading and preparing data."""
        self.processor.load_and_prepare_data()
        
        self.assertIsNotNone(self.processor.data)
        self.assertIsNotNone(self.processor.preprocessor)
        self.assertIsNotNone(self.processor.analyzer)
        self.assertEqual(self.processor.data.shape[0], 300)  # 300 respondents (numbering corrected)
    
    def test_get_validation_report(self):
        """Test getting validation report."""
        self.processor.load_and_prepare_data()
        
        validation_report = self.processor.get_validation_report()
        self.assertIsInstance(validation_report, dict)
        self.assertIn('total_responses', validation_report)
        self.assertIn('question_stats', validation_report)
    
    def test_get_choice_summary(self):
        """Test getting choice summary."""
        self.processor.load_and_prepare_data()
        
        summary = self.processor.get_choice_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 6)
    
    def test_get_choice_matrix(self):
        """Test getting choice matrix."""
        self.processor.load_and_prepare_data()
        
        choice_matrix = self.processor.get_choice_matrix()
        self.assertIsInstance(choice_matrix, pd.DataFrame)
        self.assertGreater(len(choice_matrix), 0)
    
    def test_get_attribute_analysis(self):
        """Test getting attribute analysis."""
        self.processor.load_and_prepare_data()
        
        analysis = self.processor.get_attribute_analysis()
        self.assertIsInstance(analysis, dict)
        self.assertIn('sugar_free_preference', analysis)
        self.assertIn('health_label_preference', analysis)
    
    def test_export_dce_data(self):
        """Test exporting DCE data."""
        self.processor.load_and_prepare_data()
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            self.processor.export_dce_data(temp_dir)
            
            # Check that files were created
            temp_path = Path(temp_dir)
            csv_files = list(temp_path.glob("*.csv"))
            
            # Should have multiple files
            self.assertGreater(len(csv_files), 0)
            
            # Check specific files exist
            expected_files = [
                "dce_raw_data.csv",
                "dce_choice_summary.csv", 
                "dce_choice_matrix.csv",
                "dce_choice_sets_config.csv",
                "dce_validation_report.csv"
            ]
            
            for expected_file in expected_files:
                self.assertTrue((temp_path / expected_file).exists())


class TestIntegration(unittest.TestCase):
    """Integration tests using real DCE data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.raw_data_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
        
        # Check if raw data file exists
        if not Path(self.raw_data_path).exists():
            self.skipTest(f"Raw data file not found: {self.raw_data_path}")
    
    def test_full_dce_workflow(self):
        """Test the complete DCE processing workflow."""
        # Initialize processor
        processor = DCEProcessor(self.raw_data_path)
        
        # Load and prepare data
        processor.load_and_prepare_data()
        
        # Get validation report
        validation = processor.get_validation_report()
        self.assertGreater(validation['total_responses'], 0)
        
        # Get choice summary
        summary = processor.get_choice_summary()
        self.assertEqual(len(summary), 6)
        
        # Get attribute analysis
        analysis = processor.get_attribute_analysis()
        self.assertIn('sugar_free_preference', analysis)
        
        # Test data integrity
        choice_matrix = processor.get_choice_matrix()
        self.assertEqual(len(choice_matrix), 300 * 6 * 3)  # 300 respondents * 6 questions * 3 alternatives


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
