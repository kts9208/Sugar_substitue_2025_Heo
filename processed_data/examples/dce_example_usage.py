"""
Example Usage of DCE (Discrete Choice Experiment) Preprocessor

This file demonstrates how to use the dce_preprocessor module
to analyze DCE data for sugar substitute research.

Author: Sugar Substitute Research Team
Date: 2025-08-24
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
from dce_preprocessor import DCEProcessor, DCEConfig
import pandas as pd

def main():
    """Main function demonstrating the usage of the DCE processor."""
    
    print("=== DCE (Discrete Choice Experiment) Preprocessor Example ===\n")
    
    file_path = "../Raw data/Sugar_substitue_Raw data_250730.xlsx"
    processor = DCEProcessor(file_path)
    
    # 1. Load and prepare the data
    print("1. Loading and preparing DCE data...")
    processor.load_and_prepare_data()
    print("   ✓ DCE data loaded successfully\n")
    
    # 2. Show DCE experiment design
    print("2. DCE Experiment Design:")
    choice_sets_config = DCEConfig.get_choice_sets_summary()
    print(choice_sets_config.to_string(index=False))
    print()
    
    # 3. Get validation report
    print("3. Data Validation Report:")
    validation = processor.get_validation_report()
    print(f"   Total responses: {validation['total_responses']}")
    print(f"   Valid responses: {validation['valid_responses']}")
    print(f"   Invalid responses: {validation['invalid_responses']}")
    print(f"   Missing responses: {validation['missing_responses']}")
    print()
    
    # 4. Get choice summary
    print("4. Choice Summary by Question:")
    summary = processor.get_choice_summary()
    display_summary = summary[['question_id', 'product_a_share', 'product_b_share', 'neither_share']].round(2)
    print(display_summary.to_string(index=False))
    print()
    
    # 5. Analyze attribute preferences
    print("5. Attribute Preference Analysis:")
    attr_analysis = processor.get_attribute_analysis()
    
    # Sugar type preference
    sugar_pref = attr_analysis['sugar_free_preference']
    print(f"   Sugar-free preference:")
    print(f"     - Preference rate: {sugar_pref['preference_rate']:.3f} ({sugar_pref['preference_rate']*100:.1f}%)")
    print(f"     - Sugar-free choices: {sugar_pref['sugar_free_choices']}")
    print(f"     - Regular sugar choices: {sugar_pref['regular_sugar_choices']}")
    print()
    
    # Health label preference
    label_pref = attr_analysis['health_label_preference']
    print(f"   Health label preference:")
    print(f"     - Preference rate: {label_pref['preference_rate']:.3f} ({label_pref['preference_rate']*100:.1f}%)")
    print(f"     - With health label choices: {label_pref['with_label_choices']}")
    print(f"     - Without health label choices: {label_pref['without_label_choices']}")
    print()
    
    # Price analysis
    price_analysis = attr_analysis['price_analysis']
    print(f"   Price analysis:")
    print(f"     - Mean chosen price: {price_analysis['mean_chosen_price']:.0f}원")
    print(f"     - Median chosen price: {price_analysis['median_chosen_price']:.0f}원")
    print(f"     - Price distribution: {price_analysis['price_distribution']}")
    print()
    
    # 6. Export all DCE data
    print("6. Exporting DCE data to files...")
    processor.export_dce_data('dce_processed_data')
    print("   ✓ All DCE data exported to 'dce_processed_data' directory\n")
    
    # 7. Show detailed analysis for specific questions
    print("7. Detailed Analysis Examples:")
    
    # Example: Question q21 analysis
    print("   Question q21 Analysis:")
    q21_data = summary[summary['question_id'] == 'q21'].iloc[0]
    print(f"     Product A: {q21_data['product_a_description']} - {q21_data['product_a_share']:.1f}%")
    print(f"     Product B: {q21_data['product_b_description']} - {q21_data['product_b_share']:.1f}%")
    print(f"     Neither: {q21_data['neither_share']:.1f}%")
    print()
    
    # Example: Question q22 analysis
    print("   Question q22 Analysis:")
    q22_data = summary[summary['question_id'] == 'q22'].iloc[0]
    print(f"     Product A: {q22_data['product_a_description']} - {q22_data['product_a_share']:.1f}%")
    print(f"     Product B: {q22_data['product_b_description']} - {q22_data['product_b_share']:.1f}%")
    print(f"     Neither: {q22_data['neither_share']:.1f}%")
    print()

def demonstrate_advanced_analysis():
    """Demonstrate advanced DCE analysis capabilities."""
    
    print("\n=== Advanced DCE Analysis Examples ===\n")
    
    # Initialize processor
    processor = DCEProcessor("../Raw data/Sugar_substitue_Raw data_250730.xlsx")
    processor.load_and_prepare_data()
    
    # Example 1: Choice matrix analysis
    print("1. Choice Matrix Analysis:")
    choice_matrix = processor.get_choice_matrix()
    print(f"   Choice matrix shape: {choice_matrix.shape}")
    print(f"   Columns: {list(choice_matrix.columns)}")
    
    # Analyze choice patterns
    chosen_alternatives = choice_matrix[choice_matrix['chosen'] == 1]
    alt_distribution = chosen_alternatives['alternative'].value_counts()
    print(f"   Alternative distribution:")
    for alt, count in alt_distribution.items():
        percentage = count / len(chosen_alternatives) * 100
        print(f"     {alt}: {count} ({percentage:.1f}%)")
    print()
    
    # Example 2: Attribute correlation analysis
    print("2. Attribute Correlation Analysis:")
    # Get only chosen alternatives (excluding "Neither")
    product_choices = chosen_alternatives[chosen_alternatives['alternative'] != 'Neither']
    
    if len(product_choices) > 0:
        # Correlation between sugar-free and health label
        correlation = product_choices[['sugar_free', 'has_health_label']].corr()
        print(f"   Correlation between sugar-free and health label: {correlation.iloc[0,1]:.3f}")
        
        # Price sensitivity by attribute
        sugar_free_prices = product_choices[product_choices['sugar_free'] == 1]['price']
        regular_prices = product_choices[product_choices['sugar_free'] == 0]['price']
        
        print(f"   Average price of chosen sugar-free products: {sugar_free_prices.mean():.0f}원")
        print(f"   Average price of chosen regular products: {regular_prices.mean():.0f}원")
    print()
    
    # Example 3: Individual respondent analysis
    print("3. Individual Respondent Pattern Analysis:")
    
    # Analyze respondents who consistently choose sugar-free
    attribute_data = processor.analyzer.attribute_data
    if attribute_data is not None:
        respondent_patterns = attribute_data.groupby('respondent_id').agg({
            'chose_sugar_free': 'mean',
            'chose_health_label': 'mean',
            'chosen_price': 'mean'
        }).round(3)
        
        # Find respondents with strong sugar-free preference
        strong_sugar_free = respondent_patterns[respondent_patterns['chose_sugar_free'] >= 0.8]
        print(f"   Respondents with strong sugar-free preference (≥80%): {len(strong_sugar_free)}")
        
        # Find respondents with strong health label preference
        strong_health_label = respondent_patterns[respondent_patterns['chose_health_label'] >= 0.8]
        print(f"   Respondents with strong health label preference (≥80%): {len(strong_health_label)}")
        
        # Price-sensitive respondents (consistently choose lower prices)
        avg_price_by_respondent = respondent_patterns['chosen_price']
        price_sensitive = avg_price_by_respondent[avg_price_by_respondent <= 2200]  # Below median
        print(f"   Price-sensitive respondents (avg choice ≤2200원): {len(price_sensitive)}")
    print()
    
    # Example 4: Market simulation
    print("4. Market Share Simulation:")
    print("   Simulating market share for hypothetical products:")
    
    # Simulate market share based on observed preferences
    sugar_pref_rate = processor.get_attribute_analysis()['sugar_free_preference']['preference_rate']
    label_pref_rate = processor.get_attribute_analysis()['health_label_preference']['preference_rate']
    
    # Hypothetical products
    products = [
        {"name": "Product X", "sugar_free": True, "health_label": True, "price": 2500},
        {"name": "Product Y", "sugar_free": False, "health_label": True, "price": 2000},
        {"name": "Product Z", "sugar_free": True, "health_label": False, "price": 2200}
    ]
    
    for product in products:
        # Simple utility calculation (this is a simplified example)
        utility = 0
        if product["sugar_free"]:
            utility += sugar_pref_rate
        if product["health_label"]:
            utility += label_pref_rate
        # Price penalty (simplified)
        utility -= (product["price"] - 2000) / 1000 * 0.1
        
        print(f"   {product['name']}: Estimated utility = {utility:.3f}")
    print()

if __name__ == "__main__":
    try:
        main()
        demonstrate_advanced_analysis()
        print("=== DCE Analysis completed successfully! ===")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the raw data file exists and is accessible.")
