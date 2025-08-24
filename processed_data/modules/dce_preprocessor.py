"""
DCE (Discrete Choice Experiment) Data Preprocessor Module

This module provides functionality to preprocess and analyze DCE data from survey responses.
DCE data represents choices between product alternatives with different attributes.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SugarType(Enum):
    """Sugar type attribute."""
    REGULAR = "일반당"
    SUGAR_FREE = "무설탕"


class HealthLabel(Enum):
    """Health label attribute."""
    WITH_LABEL = "건강라벨 있음"
    WITHOUT_LABEL = "건강라벨 없음"


class Choice(Enum):
    """Choice options in DCE."""
    PRODUCT_A = 1
    PRODUCT_B = 2
    NEITHER = 3


@dataclass
class ProductAttributes:
    """Product attributes for DCE analysis."""
    sugar_type: SugarType
    health_label: HealthLabel
    price: int
    
    def __str__(self):
        return f"{self.sugar_type.value}, {self.health_label.value}, {self.price}원"


@dataclass
class ChoiceSet:
    """A choice set containing two product alternatives."""
    question_id: str
    product_a: ProductAttributes
    product_b: ProductAttributes
    
    def __str__(self):
        return f"{self.question_id}: A({self.product_a}) vs B({self.product_b})"


class DCEConfig:
    """Configuration class for DCE experiment design and product attributes."""
    
    # DCE Choice Sets Definition
    CHOICE_SETS = {
        'q21': ChoiceSet(
            question_id='q21',
            product_a=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITH_LABEL, 2500),
            product_b=ProductAttributes(SugarType.REGULAR, HealthLabel.WITHOUT_LABEL, 2000)
        ),
        'q22': ChoiceSet(
            question_id='q22',
            product_a=ProductAttributes(SugarType.REGULAR, HealthLabel.WITHOUT_LABEL, 3000),
            product_b=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITH_LABEL, 2500)
        ),
        'q23': ChoiceSet(
            question_id='q23',
            product_a=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITHOUT_LABEL, 2000),
            product_b=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITH_LABEL, 3000)
        ),
        'q24': ChoiceSet(
            question_id='q24',
            product_a=ProductAttributes(SugarType.REGULAR, HealthLabel.WITH_LABEL, 2000),
            product_b=ProductAttributes(SugarType.REGULAR, HealthLabel.WITHOUT_LABEL, 2500)
        ),
        'q25': ChoiceSet(
            question_id='q25',
            product_a=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITHOUT_LABEL, 2500),
            product_b=ProductAttributes(SugarType.REGULAR, HealthLabel.WITH_LABEL, 3000)
        ),
        'q26': ChoiceSet(
            question_id='q26',
            product_a=ProductAttributes(SugarType.REGULAR, HealthLabel.WITH_LABEL, 3000),
            product_b=ProductAttributes(SugarType.SUGAR_FREE, HealthLabel.WITHOUT_LABEL, 2000)
        )
    }
    
    @classmethod
    def get_choice_set(cls, question_id: str) -> ChoiceSet:
        """Get choice set for a specific question."""
        if question_id not in cls.CHOICE_SETS:
            raise ValueError(f"Unknown question ID: {question_id}")
        return cls.CHOICE_SETS[question_id]
    
    @classmethod
    def get_all_questions(cls) -> List[str]:
        """Get all DCE question IDs."""
        return list(cls.CHOICE_SETS.keys())
    
    @classmethod
    def get_choice_sets_summary(cls) -> pd.DataFrame:
        """Get summary of all choice sets."""
        summary_data = []
        for q_id, choice_set in cls.CHOICE_SETS.items():
            summary_data.append({
                'question_id': q_id,
                'product_a_sugar': choice_set.product_a.sugar_type.value,
                'product_a_label': choice_set.product_a.health_label.value,
                'product_a_price': choice_set.product_a.price,
                'product_b_sugar': choice_set.product_b.sugar_type.value,
                'product_b_label': choice_set.product_b.health_label.value,
                'product_b_price': choice_set.product_b.price
            })
        return pd.DataFrame(summary_data)


class DCEDataLoader:
    """Handles loading and basic validation of DCE data."""
    
    def __init__(self, file_path: str, sheet_name: str = 'DATA'):
        """
        Initialize DCEDataLoader.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet containing data
        """
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        self.dce_questions = DCEConfig.get_all_questions()
        self._validate_file()
    
    def _validate_file(self) -> None:
        """Validate that the file exists and is accessible."""
        if not self.file_path.suffix.lower() in ['.xlsx', '.xls']:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
    
    def load_dce_data(self) -> pd.DataFrame:
        """
        Load DCE data from Excel file.
        
        Returns:
            DataFrame containing DCE data with respondent ID
        """
        try:
            logger.info(f"Loading DCE data from {self.file_path}, sheet: {self.sheet_name}")
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            
            # Select DCE columns and respondent ID
            columns_to_select = ['no'] + self.dce_questions
            missing_cols = [col for col in columns_to_select if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing columns in data: {missing_cols}")
            
            dce_data = df[columns_to_select].copy()

            # Check for and handle duplicate respondent IDs (numbering errors)
            if dce_data['no'].duplicated().any():
                duplicates = dce_data[dce_data['no'].duplicated(keep=False)]['no'].unique()
                logger.warning(f"Found duplicate respondent IDs (numbering errors): {duplicates}")
                logger.info("Correcting respondent numbering...")

                # Fix numbering by assigning sequential IDs
                for dup_id in duplicates:
                    dup_indices = dce_data[dce_data['no'] == dup_id].index
                    if len(dup_indices) > 1:
                        # Keep the first occurrence with original ID
                        # Assign new ID to subsequent occurrences
                        for i, idx in enumerate(dup_indices[1:], 1):
                            # Find next available ID
                            max_id = dce_data['no'].max()
                            new_id = max_id + i
                            dce_data.loc[idx, 'no'] = new_id
                            logger.info(f"Changed duplicate ID {dup_id} to {new_id} at index {idx}")

            logger.info(f"DCE data loaded successfully. Shape: {dce_data.shape}")

            return dce_data
            
        except Exception as e:
            logger.error(f"Error loading DCE data: {e}")
            raise
    
    def validate_dce_responses(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DCE response data.
        
        Args:
            data: DCE data DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'total_responses': len(data),
            'valid_responses': 0,
            'invalid_responses': 0,
            'missing_responses': 0,
            'question_stats': {}
        }
        
        for question in self.dce_questions:
            if question in data.columns:
                q_data = data[question]
                valid_choices = q_data.isin([1, 2, 3])
                missing_choices = q_data.isna()
                
                validation_results['question_stats'][question] = {
                    'valid_count': valid_choices.sum(),
                    'invalid_count': (~valid_choices & ~missing_choices).sum(),
                    'missing_count': missing_choices.sum(),
                    'choice_distribution': q_data.value_counts().to_dict()
                }
        
        # Calculate overall statistics
        total_valid = sum([stats['valid_count'] for stats in validation_results['question_stats'].values()])
        total_invalid = sum([stats['invalid_count'] for stats in validation_results['question_stats'].values()])
        total_missing = sum([stats['missing_count'] for stats in validation_results['question_stats'].values()])
        
        validation_results['valid_responses'] = total_valid
        validation_results['invalid_responses'] = total_invalid
        validation_results['missing_responses'] = total_missing
        
        return validation_results


class DCEPreprocessor:
    """Handles preprocessing of DCE choice data for analysis."""
    
    def __init__(self, dce_data: pd.DataFrame):
        """
        Initialize DCEPreprocessor.
        
        Args:
            dce_data: DataFrame containing DCE choice data
        """
        self.dce_data = dce_data.copy()
        self.dce_questions = DCEConfig.get_all_questions()
        self._validate_data()
    
    def _validate_data(self) -> None:
        """Validate DCE data structure."""
        if self.dce_data.empty:
            raise ValueError("DCE data is empty")
        
        missing_questions = [q for q in self.dce_questions if q not in self.dce_data.columns]
        if missing_questions:
            raise ValueError(f"Missing DCE questions: {missing_questions}")
    
    def create_choice_matrix(self) -> pd.DataFrame:
        """
        Create a choice matrix with expanded format for analysis.
        
        Returns:
            DataFrame with choice data in expanded format
        """
        choice_records = []
        
        for _, row in self.dce_data.iterrows():
            respondent_id = row['no']
            
            for question in self.dce_questions:
                choice_value = row[question]
                choice_set = DCEConfig.get_choice_set(question)
                
                # Create record for Product A
                choice_records.append({
                    'respondent_id': respondent_id,
                    'question_id': question,
                    'alternative': 'A',
                    'chosen': 1 if choice_value == 1 else 0,
                    'sugar_type': choice_set.product_a.sugar_type.value,
                    'health_label': choice_set.product_a.health_label.value,
                    'price': choice_set.product_a.price,
                    'sugar_free': 1 if choice_set.product_a.sugar_type == SugarType.SUGAR_FREE else 0,
                    'has_health_label': 1 if choice_set.product_a.health_label == HealthLabel.WITH_LABEL else 0
                })
                
                # Create record for Product B
                choice_records.append({
                    'respondent_id': respondent_id,
                    'question_id': question,
                    'alternative': 'B',
                    'chosen': 1 if choice_value == 2 else 0,
                    'sugar_type': choice_set.product_b.sugar_type.value,
                    'health_label': choice_set.product_b.health_label.value,
                    'price': choice_set.product_b.price,
                    'sugar_free': 1 if choice_set.product_b.sugar_type == SugarType.SUGAR_FREE else 0,
                    'has_health_label': 1 if choice_set.product_b.health_label == HealthLabel.WITH_LABEL else 0
                })
                
                # Create record for "Neither" option
                choice_records.append({
                    'respondent_id': respondent_id,
                    'question_id': question,
                    'alternative': 'Neither',
                    'chosen': 1 if choice_value == 3 else 0,
                    'sugar_type': 'None',
                    'health_label': 'None',
                    'price': 0,
                    'sugar_free': 0,
                    'has_health_label': 0
                })
        
        return pd.DataFrame(choice_records)
    
    def create_attribute_effects_data(self) -> pd.DataFrame:
        """
        Create data for analyzing attribute effects.
        
        Returns:
            DataFrame with attribute-level analysis data
        """
        attribute_records = []
        
        for _, row in self.dce_data.iterrows():
            respondent_id = row['no']
            
            for question in self.dce_questions:
                choice_value = row[question]
                choice_set = DCEConfig.get_choice_set(question)
                
                if choice_value in [1, 2]:  # Exclude "neither" choices
                    chosen_product = choice_set.product_a if choice_value == 1 else choice_set.product_b
                    
                    attribute_records.append({
                        'respondent_id': respondent_id,
                        'question_id': question,
                        'chosen_sugar_type': chosen_product.sugar_type.value,
                        'chosen_health_label': chosen_product.health_label.value,
                        'chosen_price': chosen_product.price,
                        'chose_sugar_free': 1 if chosen_product.sugar_type == SugarType.SUGAR_FREE else 0,
                        'chose_health_label': 1 if chosen_product.health_label == HealthLabel.WITH_LABEL else 0,
                        'choice_value': choice_value
                    })
        
        return pd.DataFrame(attribute_records)
    
    def get_choice_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each choice question.
        
        Returns:
            DataFrame with choice summary statistics
        """
        summary_data = []
        
        for question in self.dce_questions:
            if question in self.dce_data.columns:
                q_data = self.dce_data[question]
                choice_counts = q_data.value_counts().sort_index()
                total_responses = len(q_data.dropna())
                
                choice_set = DCEConfig.get_choice_set(question)
                
                summary_data.append({
                    'question_id': question,
                    'total_responses': total_responses,
                    'product_a_choices': choice_counts.get(1, 0),
                    'product_b_choices': choice_counts.get(2, 0),
                    'neither_choices': choice_counts.get(3, 0),
                    'product_a_share': choice_counts.get(1, 0) / total_responses * 100 if total_responses > 0 else 0,
                    'product_b_share': choice_counts.get(2, 0) / total_responses * 100 if total_responses > 0 else 0,
                    'neither_share': choice_counts.get(3, 0) / total_responses * 100 if total_responses > 0 else 0,
                    'product_a_description': str(choice_set.product_a),
                    'product_b_description': str(choice_set.product_b)
                })
        
        return pd.DataFrame(summary_data)


class DCEAnalyzer:
    """Provides analysis capabilities for DCE data."""
    
    def __init__(self, preprocessor: DCEPreprocessor):
        """
        Initialize DCEAnalyzer.
        
        Args:
            preprocessor: DCEPreprocessor instance with loaded data
        """
        self.preprocessor = preprocessor
        self.choice_matrix = None
        self.attribute_data = None
    
    def prepare_analysis_data(self) -> None:
        """Prepare data for analysis."""
        logger.info("Preparing DCE analysis data...")
        self.choice_matrix = self.preprocessor.create_choice_matrix()
        self.attribute_data = self.preprocessor.create_attribute_effects_data()
        logger.info("Analysis data prepared successfully")
    
    def analyze_attribute_preferences(self) -> Dict[str, Any]:
        """
        Analyze preferences for different attributes.
        
        Returns:
            Dictionary containing attribute preference analysis
        """
        if self.attribute_data is None:
            self.prepare_analysis_data()
        
        analysis_results = {}
        
        # Sugar type preference
        sugar_pref = self.attribute_data['chose_sugar_free'].mean()
        analysis_results['sugar_free_preference'] = {
            'preference_rate': sugar_pref,
            'sugar_free_choices': self.attribute_data['chose_sugar_free'].sum(),
            'regular_sugar_choices': len(self.attribute_data) - self.attribute_data['chose_sugar_free'].sum()
        }
        
        # Health label preference
        label_pref = self.attribute_data['chose_health_label'].mean()
        analysis_results['health_label_preference'] = {
            'preference_rate': label_pref,
            'with_label_choices': self.attribute_data['chose_health_label'].sum(),
            'without_label_choices': len(self.attribute_data) - self.attribute_data['chose_health_label'].sum()
        }
        
        # Price analysis
        price_stats = self.attribute_data['chosen_price'].describe()
        analysis_results['price_analysis'] = {
            'mean_chosen_price': price_stats['mean'],
            'median_chosen_price': price_stats['50%'],
            'price_distribution': self.attribute_data['chosen_price'].value_counts().to_dict()
        }
        
        return analysis_results
    
    def get_choice_patterns(self) -> pd.DataFrame:
        """
        Analyze choice patterns across respondents.
        
        Returns:
            DataFrame with choice pattern analysis
        """
        if self.choice_matrix is None:
            self.prepare_analysis_data()
        
        # Aggregate choices by respondent
        respondent_patterns = self.choice_matrix.groupby('respondent_id').agg({
            'chosen': 'sum',
            'sugar_free': lambda x: (x * self.choice_matrix.loc[x.index, 'chosen']).sum(),
            'has_health_label': lambda x: (x * self.choice_matrix.loc[x.index, 'chosen']).sum(),
            'price': lambda x: (x * self.choice_matrix.loc[x.index, 'chosen']).sum() / max(1, self.choice_matrix.loc[x.index, 'chosen'].sum())
        }).reset_index()
        
        respondent_patterns.columns = ['respondent_id', 'total_choices', 'sugar_free_choices', 'health_label_choices', 'avg_chosen_price']
        
        return respondent_patterns


class DCEProcessor:
    """Main class for DCE data processing and analysis."""
    
    def __init__(self, file_path: str, sheet_name: str = 'DATA'):
        """
        Initialize DCEProcessor.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet containing data
        """
        self.loader = DCEDataLoader(file_path, sheet_name)
        self.data = None
        self.preprocessor = None
        self.analyzer = None
    
    def load_and_prepare_data(self) -> None:
        """Load and prepare DCE data for analysis."""
        self.data = self.loader.load_dce_data()
        self.preprocessor = DCEPreprocessor(self.data)
        self.analyzer = DCEAnalyzer(self.preprocessor)
        logger.info("DCE data loaded and prepared for analysis")
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get data validation report."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.loader.validate_dce_responses(self.data)
    
    def get_choice_summary(self) -> pd.DataFrame:
        """Get choice summary statistics."""
        if self.preprocessor is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.preprocessor.get_choice_summary()
    
    def get_choice_matrix(self) -> pd.DataFrame:
        """Get choice data in matrix format."""
        if self.analyzer is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        if self.analyzer.choice_matrix is None:
            self.analyzer.prepare_analysis_data()
        
        return self.analyzer.choice_matrix
    
    def get_attribute_analysis(self) -> Dict[str, Any]:
        """Get attribute preference analysis."""
        if self.analyzer is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        return self.analyzer.analyze_attribute_preferences()
    
    def export_dce_data(self, output_dir: str = 'dce_processed_data') -> None:
        """
        Export DCE data to various formats for analysis.
        
        Args:
            output_dir: Directory to save the processed files
        """
        if self.preprocessor is None or self.analyzer is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare_data() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export original DCE data
        self.data.to_csv(output_path / "dce_raw_data.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Raw DCE data exported to {output_path / 'dce_raw_data.csv'}")
        
        # Export choice summary
        choice_summary = self.get_choice_summary()
        choice_summary.to_csv(output_path / "dce_choice_summary.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Choice summary exported to {output_path / 'dce_choice_summary.csv'}")
        
        # Export choice matrix
        choice_matrix = self.get_choice_matrix()
        choice_matrix.to_csv(output_path / "dce_choice_matrix.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Choice matrix exported to {output_path / 'dce_choice_matrix.csv'}")
        
        # Export attribute data
        if self.analyzer.attribute_data is not None:
            self.analyzer.attribute_data.to_csv(output_path / "dce_attribute_data.csv", index=False, encoding='utf-8-sig')
            logger.info(f"Attribute data exported to {output_path / 'dce_attribute_data.csv'}")
        
        # Export choice sets configuration
        choice_sets_config = DCEConfig.get_choice_sets_summary()
        choice_sets_config.to_csv(output_path / "dce_choice_sets_config.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Choice sets configuration exported to {output_path / 'dce_choice_sets_config.csv'}")
        
        # Export validation report
        validation_report = self.get_validation_report()
        validation_df = pd.DataFrame([validation_report])
        validation_df.to_csv(output_path / "dce_validation_report.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Validation report exported to {output_path / 'dce_validation_report.csv'}")


# Example usage
if __name__ == "__main__":
    # Example usage of the DCE processor
    file_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
    
    try:
        # Initialize processor
        processor = DCEProcessor(file_path)
        
        # Load and prepare data
        processor.load_and_prepare_data()
        
        # Get choice summary
        summary = processor.get_choice_summary()
        print("DCE Choice Summary:")
        print(summary[['question_id', 'product_a_share', 'product_b_share', 'neither_share']].round(2))
        
        # Get attribute analysis
        attr_analysis = processor.get_attribute_analysis()
        print(f"\nSugar-free preference rate: {attr_analysis['sugar_free_preference']['preference_rate']:.3f}")
        print(f"Health label preference rate: {attr_analysis['health_label_preference']['preference_rate']:.3f}")
        
        # Export all data
        processor.export_dce_data()
        
    except Exception as e:
        logger.error(f"Error in DCE processing: {e}")
