"""
Example Usage of Survey Data Preprocessor

This file demonstrates how to use the survey_data_preprocessor module
to load and preprocess survey data for statistical analysis.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from survey_data_preprocessor import SurveyDataPreprocessor, FactorConfig
import pandas as pd

def main():
    """Main function demonstrating the usage of the preprocessor."""
    
    # 1. Initialize the preprocessor with the raw data file
    print("=== Survey Data Preprocessor Example ===\n")
    
    file_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
    preprocessor = SurveyDataPreprocessor(file_path)
    
    # 2. Load and prepare the data
    print("1. Loading and preparing data...")
    preprocessor.load_and_prepare_data()
    print("   ✓ Data loaded successfully\n")
    
    # 3. Get summary of all factors
    print("2. Factor Summary:")
    summary = preprocessor.get_summary()
    print(summary.to_string(index=False))
    print()
    
    # 4. Get data for specific factors
    print("3. Getting data for specific factors:")
    
    # Health concern factor
    health_data = preprocessor.get_factor_data('health_concern')
    print(f"   Health Concern Factor:")
    print(f"   - Shape: {health_data.shape}")
    print(f"   - Columns: {list(health_data.columns)}")
    print(f"   - Sample data:")
    print(health_data.head(3).to_string(index=False))
    print()
    
    # Purchase intention factor
    purchase_data = preprocessor.get_factor_data('purchase_intention')
    print(f"   Purchase Intention Factor:")
    print(f"   - Shape: {purchase_data.shape}")
    print(f"   - Columns: {list(purchase_data.columns)}")
    print(f"   - Sample data:")
    print(purchase_data.head(3).to_string(index=False))
    print()
    
    # 5. Get all factors data at once
    print("4. Getting all factors data:")
    all_factors = preprocessor.get_all_factors_data()
    for factor_name, factor_data in all_factors.items():
        print(f"   {factor_name}: {factor_data.shape}")
    print()
    
    # 6. Export data to CSV files
    print("5. Exporting factor data to CSV files...")
    preprocessor.export_factor_data('processed_data')
    print("   ✓ All factor data exported to 'processed_data' directory\n")
    
    # 7. Demonstrate individual factor usage for statistical analysis
    print("6. Example: Preparing data for statistical analysis")
    
    # Example: Calculate descriptive statistics for health concern
    health_stats = health_data.iloc[:, 1:].describe()  # Exclude 'no' column
    print("   Health Concern Descriptive Statistics:")
    print(health_stats.round(2).to_string())
    print()
    
    # Example: Calculate correlation matrix for perceived benefit
    benefit_data = preprocessor.get_factor_data('perceived_benefit')
    benefit_corr = benefit_data.iloc[:, 1:].corr()  # Exclude 'no' column
    print("   Perceived Benefit Correlation Matrix:")
    print(benefit_corr.round(3).to_string())
    print()
    
    # 8. Show available factor configurations
    print("7. Available Factor Configurations:")
    for factor in FactorConfig.get_all_factors():
        description = FactorConfig.get_factor_description(factor)
        questions = FactorConfig.get_factor_questions(factor)
        print(f"   {factor}:")
        print(f"     Description: {description}")
        print(f"     Questions: {questions}")
        print()

def demonstrate_advanced_usage():
    """Demonstrate advanced usage patterns."""
    
    print("\n=== Advanced Usage Examples ===\n")
    
    # Initialize preprocessor
    preprocessor = SurveyDataPreprocessor("../Raw data/Sugar_substitue_Raw data_250730.xlsx")
    preprocessor.load_and_prepare_data()
    
    # Example 1: Combine multiple factors for analysis
    print("1. Combining multiple factors:")
    health_data = preprocessor.get_factor_data('health_concern')
    benefit_data = preprocessor.get_factor_data('perceived_benefit')
    
    # Merge on 'no' column
    combined_data = pd.merge(health_data, benefit_data, on='no')
    print(f"   Combined Health + Benefit data shape: {combined_data.shape}")
    print(f"   Columns: {list(combined_data.columns)}")
    print()
    
    # Example 2: Filter data for specific analysis
    print("2. Filtering data for analysis:")
    # Get only respondents with high health concern (assuming scale 1-5, high = 4-5)
    health_cols = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11']
    health_mean = health_data[health_cols].mean(axis=1)
    high_health_concern = health_data[health_mean >= 4]
    print(f"   Respondents with high health concern: {len(high_health_concern)}")
    print()
    
    # Example 3: Prepare data for factor analysis
    print("3. Preparing data for factor analysis:")
    nutrition_data = preprocessor.get_factor_data('nutrition_knowledge')
    nutrition_matrix = nutrition_data.iloc[:, 1:].values  # Exclude 'no' column
    print(f"   Nutrition knowledge matrix shape: {nutrition_matrix.shape}")
    print(f"   Ready for factor analysis (e.g., using sklearn.decomposition.FactorAnalysis)")
    print()

if __name__ == "__main__":
    try:
        main()
        demonstrate_advanced_usage()
        print("=== Example completed successfully! ===")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the raw data file exists and is accessible.")
