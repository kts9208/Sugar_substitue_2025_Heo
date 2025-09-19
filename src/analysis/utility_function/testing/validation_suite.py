"""
Validation suite for utility function calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationSuite:
    """
    Comprehensive validation suite for utility function calculations.
    
    Validates data quality, model assumptions, and result consistency.
    """
    
    def __init__(self):
        """Initialize validation suite."""
        self.validation_results = {}
        
    def validate_dce_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DCE data quality and structure.
        
        Args:
            data: DCE data to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating DCE data...")
        
        validation = {
            'data_structure': {},
            'data_quality': {},
            'choice_patterns': {},
            'attribute_distributions': {}
        }
        
        # Data structure validation
        required_columns = ['respondent_id', 'sugar_free', 'health_label', 'price_normalized']
        validation['data_structure']['has_required_columns'] = all(col in data.columns for col in required_columns)
        validation['data_structure']['n_observations'] = len(data)
        validation['data_structure']['n_variables'] = len(data.columns)
        
        # Data quality validation
        validation['data_quality']['has_missing_values'] = data.isnull().any().any()
        validation['data_quality']['missing_value_count'] = data.isnull().sum().sum()
        validation['data_quality']['duplicate_rows'] = data.duplicated().sum()
        
        # Binary variable validation
        if 'sugar_free' in data.columns:
            validation['data_quality']['sugar_free_valid'] = data['sugar_free'].isin([0, 1]).all()
            
        if 'health_label' in data.columns:
            validation['data_quality']['health_label_valid'] = data['health_label'].isin([0, 1]).all()
            
        # Price validation
        if 'price_normalized' in data.columns:
            validation['data_quality']['price_range_valid'] = (
                (data['price_normalized'] >= 0).all() and 
                (data['price_normalized'] <= 1).all()
            )
            
        # Choice patterns validation
        if 'choice_value' in data.columns:
            validation['choice_patterns']['choice_rate'] = data['choice_value'].mean()
            validation['choice_patterns']['choice_variance'] = data['choice_value'].var()
            validation['choice_patterns']['has_choice_variation'] = data['choice_value'].nunique() > 1
            
        # Attribute distributions
        for col in ['sugar_free', 'health_label']:
            if col in data.columns:
                validation['attribute_distributions'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'distribution': data[col].value_counts().to_dict()
                }
                
        if 'price_normalized' in data.columns:
            validation['attribute_distributions']['price_normalized'] = {
                'mean': float(data['price_normalized'].mean()),
                'std': float(data['price_normalized'].std()),
                'min': float(data['price_normalized'].min()),
                'max': float(data['price_normalized'].max())
            }
            
        return validation
    
    def validate_sem_results(self, sem_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SEM results structure and content.
        
        Args:
            sem_results: SEM results to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating SEM results...")
        
        validation = {
            'structure': {},
            'factor_effects': {},
            'statistical_validity': {}
        }
        
        # Structure validation
        validation['structure']['has_factor_effects'] = 'factor_effects' in sem_results
        
        if 'factor_effects' in sem_results:
            factor_effects = sem_results['factor_effects']
            expected_factors = ['health_benefit', 'nutrition_knowledge', 'perceived_price']
            
            validation['structure']['has_expected_factors'] = all(
                factor in factor_effects for factor in expected_factors
            )
            validation['structure']['n_factors'] = len(factor_effects)
            
            # Factor effects validation
            for factor_name, factor_info in factor_effects.items():
                if isinstance(factor_info, dict):
                    validation['factor_effects'][factor_name] = {
                        'has_coefficient': 'coefficient' in factor_info,
                        'has_p_value': 'p_value' in factor_info,
                        'coefficient_value': factor_info.get('coefficient', None),
                        'p_value': factor_info.get('p_value', None),
                        'is_significant': factor_info.get('p_value', 1.0) < 0.05
                    }
                    
        # Statistical validity
        if 'factor_effects' in sem_results:
            coefficients = []
            p_values = []
            
            for factor_info in sem_results['factor_effects'].values():
                if isinstance(factor_info, dict):
                    if 'coefficient' in factor_info:
                        coefficients.append(factor_info['coefficient'])
                    if 'p_value' in factor_info:
                        p_values.append(factor_info['p_value'])
                        
            if coefficients:
                validation['statistical_validity']['coefficient_range'] = {
                    'min': min(coefficients),
                    'max': max(coefficients),
                    'mean': np.mean(coefficients)
                }
                
            if p_values:
                validation['statistical_validity']['significance'] = {
                    'n_significant': sum(1 for p in p_values if p < 0.05),
                    'n_total': len(p_values),
                    'proportion_significant': sum(1 for p in p_values if p < 0.05) / len(p_values)
                }
                
        return validation
    
    def validate_utility_results(self, utility_values: pd.Series, 
                                data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate utility calculation results.
        
        Args:
            utility_values: Calculated utility values
            data: Original data (optional)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating utility results...")
        
        validation = {
            'basic_properties': {},
            'distribution_properties': {},
            'consistency_checks': {},
            'relationship_checks': {}
        }
        
        # Basic properties
        validation['basic_properties']['n_observations'] = len(utility_values)
        validation['basic_properties']['has_missing_values'] = utility_values.isnull().any()
        validation['basic_properties']['has_infinite_values'] = np.isinf(utility_values).any()
        validation['basic_properties']['all_finite'] = np.isfinite(utility_values).all()
        
        # Distribution properties
        validation['distribution_properties']['mean'] = float(utility_values.mean())
        validation['distribution_properties']['std'] = float(utility_values.std())
        validation['distribution_properties']['min'] = float(utility_values.min())
        validation['distribution_properties']['max'] = float(utility_values.max())
        validation['distribution_properties']['range'] = float(utility_values.max() - utility_values.min())
        validation['distribution_properties']['skewness'] = float(stats.skew(utility_values))
        validation['distribution_properties']['kurtosis'] = float(stats.kurtosis(utility_values))
        
        # Check for reasonable utility range
        validation['distribution_properties']['reasonable_range'] = (
            utility_values.min() > -10 and utility_values.max() < 10
        )
        
        # Consistency checks
        validation['consistency_checks']['has_variation'] = utility_values.nunique() > 1
        validation['consistency_checks']['std_positive'] = utility_values.std() > 0
        
        # Relationship checks with original data
        if data is not None:
            validation['relationship_checks'] = self._validate_utility_relationships(utility_values, data)
            
        return validation
    
    def _validate_utility_relationships(self, utility_values: pd.Series, 
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate relationships between utility and original data.
        
        Args:
            utility_values: Utility values
            data: Original data
            
        Returns:
            Dictionary with relationship validation results
        """
        relationships = {}
        
        # Check correlation with choice outcomes
        if 'choice_value' in data.columns:
            correlation = utility_values.corr(data['choice_value'])
            relationships['choice_correlation'] = {
                'correlation': float(correlation),
                'positive_correlation': correlation > 0,
                'strong_correlation': abs(correlation) > 0.3
            }
            
        # Check relationships with attributes
        for attr in ['sugar_free', 'health_label', 'price_normalized']:
            if attr in data.columns:
                correlation = utility_values.corr(data[attr])
                relationships[f'{attr}_correlation'] = {
                    'correlation': float(correlation),
                    'expected_sign': self._get_expected_sign(attr),
                    'sign_correct': self._check_sign_correctness(correlation, attr)
                }
                
        return relationships
    
    def _get_expected_sign(self, attribute: str) -> str:
        """Get expected sign for attribute correlation with utility."""
        if attribute in ['sugar_free', 'health_label']:
            return 'positive'  # Generally expected to increase utility
        elif 'price' in attribute:
            return 'negative'  # Higher price should decrease utility
        else:
            return 'unknown'
            
    def _check_sign_correctness(self, correlation: float, attribute: str) -> bool:
        """Check if correlation sign matches expectation."""
        expected_sign = self._get_expected_sign(attribute)
        
        if expected_sign == 'positive':
            return correlation >= 0
        elif expected_sign == 'negative':
            return correlation <= 0
        else:
            return True  # Unknown expectation, so any sign is acceptable
    
    def validate_component_contributions(self, contributions: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate utility component contributions.
        
        Args:
            contributions: DataFrame with component contributions
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating component contributions...")
        
        validation = {
            'structure': {},
            'contribution_properties': {},
            'balance_checks': {}
        }
        
        # Structure validation
        component_columns = [col for col in contributions.columns if col != 'total_utility']
        validation['structure']['n_components'] = len(component_columns)
        validation['structure']['has_total_utility'] = 'total_utility' in contributions.columns
        validation['structure']['component_names'] = component_columns
        
        # Component properties
        for component in component_columns:
            if component in contributions.columns:
                comp_values = contributions[component]
                validation['contribution_properties'][component] = {
                    'mean': float(comp_values.mean()),
                    'std': float(comp_values.std()),
                    'min': float(comp_values.min()),
                    'max': float(comp_values.max()),
                    'has_variation': comp_values.nunique() > 1,
                    'all_finite': np.isfinite(comp_values).all()
                }
                
        # Balance checks
        if 'total_utility' in contributions.columns and len(component_columns) > 0:
            # Check if components sum to total (approximately)
            component_sum = contributions[component_columns].sum(axis=1)
            total_utility = contributions['total_utility']
            
            difference = np.abs(component_sum - total_utility)
            validation['balance_checks']['components_sum_to_total'] = (difference < 1e-10).all()
            validation['balance_checks']['max_difference'] = float(difference.max())
            validation['balance_checks']['mean_difference'] = float(difference.mean())
            
        return validation
    
    def validate_model_assumptions(self, data: pd.DataFrame, 
                                 utility_values: pd.Series) -> Dict[str, Any]:
        """
        Validate key model assumptions.
        
        Args:
            data: Input data
            utility_values: Calculated utility values
            
        Returns:
            Dictionary with assumption validation results
        """
        logger.info("Validating model assumptions...")
        
        validation = {
            'independence': {},
            'linearity': {},
            'homoscedasticity': {},
            'normality': {}
        }
        
        # Independence assumption (check for autocorrelation)
        if len(utility_values) > 10:
            # Simple lag-1 autocorrelation
            lag1_corr = utility_values.autocorr(lag=1)
            validation['independence']['lag1_autocorrelation'] = float(lag1_corr) if not np.isnan(lag1_corr) else 0.0
            validation['independence']['independence_satisfied'] = abs(lag1_corr) < 0.3 if not np.isnan(lag1_corr) else True
            
        # Linearity assumption (check relationships with continuous variables)
        for attr in ['price_normalized']:
            if attr in data.columns:
                # Check if relationship is approximately linear
                correlation_linear = utility_values.corr(data[attr])
                correlation_squared = utility_values.corr(data[attr] ** 2)
                
                validation['linearity'][attr] = {
                    'linear_correlation': float(correlation_linear),
                    'quadratic_correlation': float(correlation_squared),
                    'linearity_satisfied': abs(correlation_linear) > abs(correlation_squared) * 0.8
                }
                
        # Homoscedasticity (constant variance)
        if 'choice_value' in data.columns:
            # Check variance across choice groups
            utility_chosen = utility_values[data['choice_value'] == 1]
            utility_not_chosen = utility_values[data['choice_value'] == 0]
            
            if len(utility_chosen) > 1 and len(utility_not_chosen) > 1:
                var_ratio = utility_chosen.var() / utility_not_chosen.var()
                validation['homoscedasticity']['variance_ratio'] = float(var_ratio)
                validation['homoscedasticity']['homoscedasticity_satisfied'] = 0.5 < var_ratio < 2.0
                
        # Normality of residuals (approximate check)
        if len(utility_values) > 8:  # Minimum for Shapiro-Wilk test
            try:
                # Test normality of utility values (proxy for residuals)
                sample_size = min(5000, len(utility_values))  # Shapiro-Wilk limitation
                sample_utilities = utility_values.sample(sample_size) if len(utility_values) > sample_size else utility_values
                
                shapiro_stat, shapiro_p = stats.shapiro(sample_utilities)
                validation['normality']['shapiro_statistic'] = float(shapiro_stat)
                validation['normality']['shapiro_p_value'] = float(shapiro_p)
                validation['normality']['normality_satisfied'] = shapiro_p > 0.05
                
            except Exception as e:
                logger.warning(f"Could not perform normality test: {str(e)}")
                validation['normality']['test_failed'] = True
                
        return validation
    
    def generate_validation_report(self, dce_data: pd.DataFrame,
                                 sem_results: Dict[str, Any],
                                 utility_values: pd.Series,
                                 contributions: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            dce_data: DCE data
            sem_results: SEM results
            utility_values: Utility values
            contributions: Component contributions (optional)
            
        Returns:
            Comprehensive validation report
        """
        logger.info("Generating comprehensive validation report...")
        
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'dce_data_validation': self.validate_dce_data(dce_data),
            'sem_results_validation': self.validate_sem_results(sem_results),
            'utility_results_validation': self.validate_utility_results(utility_values, dce_data),
            'model_assumptions_validation': self.validate_model_assumptions(dce_data, utility_values)
        }
        
        if contributions is not None:
            report['component_contributions_validation'] = self.validate_component_contributions(contributions)
            
        # Generate overall validation summary
        report['validation_summary'] = self._generate_validation_summary(report)
        
        return report
    
    def _generate_validation_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of validation results.
        
        Args:
            report: Validation report
            
        Returns:
            Validation summary
        """
        summary = {
            'overall_status': 'PASS',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check for critical issues
        dce_validation = report.get('dce_data_validation', {})
        if dce_validation.get('data_quality', {}).get('has_missing_values', False):
            summary['critical_issues'].append("DCE data contains missing values")
            summary['overall_status'] = 'FAIL'
            
        utility_validation = report.get('utility_results_validation', {})
        if not utility_validation.get('basic_properties', {}).get('all_finite', True):
            summary['critical_issues'].append("Utility values contain infinite or NaN values")
            summary['overall_status'] = 'FAIL'
            
        # Check for warnings
        if not utility_validation.get('distribution_properties', {}).get('reasonable_range', True):
            summary['warnings'].append("Utility values outside reasonable range (-10, 10)")
            
        assumptions_validation = report.get('model_assumptions_validation', {})
        if not assumptions_validation.get('independence', {}).get('independence_satisfied', True):
            summary['warnings'].append("Independence assumption may be violated (high autocorrelation)")
            
        # Generate recommendations
        if dce_validation.get('choice_patterns', {}).get('choice_rate', 0.5) < 0.1 or \
           dce_validation.get('choice_patterns', {}).get('choice_rate', 0.5) > 0.9:
            summary['recommendations'].append("Consider balancing choice outcomes in DCE data")
            
        if not utility_validation.get('relationship_checks', {}).get('choice_correlation', {}).get('positive_correlation', True):
            summary['recommendations'].append("Investigate negative correlation between utility and choice outcomes")
            
        return summary
