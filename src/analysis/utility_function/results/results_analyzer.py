"""
Results analyzer for utility function calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """
    Analyzes utility function calculation results.
    
    Provides comprehensive analysis including model performance,
    component effects, and choice prediction accuracy.
    """
    
    def __init__(self):
        """Initialize results analyzer."""
        self.analysis_cache = {}
        
    def analyze_utility_distribution(self, utility_values: pd.Series) -> Dict[str, Any]:
        """
        Analyze the distribution of utility values.
        
        Args:
            utility_values: Series with utility values
            
        Returns:
            Dictionary with distribution analysis
        """
        logger.info("Analyzing utility distribution...")
        
        analysis = {
            'descriptive_stats': {
                'count': len(utility_values),
                'mean': float(utility_values.mean()),
                'std': float(utility_values.std()),
                'min': float(utility_values.min()),
                'max': float(utility_values.max()),
                'skewness': float(stats.skew(utility_values)),
                'kurtosis': float(stats.kurtosis(utility_values))
            },
            'percentiles': {
                '1st': float(utility_values.quantile(0.01)),
                '5th': float(utility_values.quantile(0.05)),
                '10th': float(utility_values.quantile(0.10)),
                '25th': float(utility_values.quantile(0.25)),
                '50th': float(utility_values.quantile(0.50)),
                '75th': float(utility_values.quantile(0.75)),
                '90th': float(utility_values.quantile(0.90)),
                '95th': float(utility_values.quantile(0.95)),
                '99th': float(utility_values.quantile(0.99))
            }
        }
        
        # Test for normality
        try:
            shapiro_stat, shapiro_p = stats.shapiro(utility_values.sample(min(5000, len(utility_values))))
            analysis['normality_test'] = {
                'shapiro_wilk_statistic': float(shapiro_stat),
                'shapiro_wilk_p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        except Exception as e:
            logger.warning(f"Could not perform normality test: {str(e)}")
            
        return analysis
    
    def analyze_component_effects(self, decomposition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the effects of individual utility components.
        
        Args:
            decomposition: Utility decomposition results
            
        Returns:
            Dictionary with component effects analysis
        """
        logger.info("Analyzing component effects...")
        
        if 'contributions' not in decomposition:
            return {}
            
        contributions = decomposition['contributions']
        
        # Exclude total utility column for component analysis
        component_columns = [col for col in contributions.columns if col != 'total_utility']
        
        analysis = {
            'component_statistics': {},
            'component_correlations': {},
            'relative_importance': {}
        }
        
        # Analyze each component
        for component in component_columns:
            if component in contributions.columns:
                component_values = contributions[component]
                
                analysis['component_statistics'][component] = {
                    'mean': float(component_values.mean()),
                    'std': float(component_values.std()),
                    'min': float(component_values.min()),
                    'max': float(component_values.max()),
                    'contribution_range': float(component_values.max() - component_values.min()),
                    'coefficient_of_variation': float(component_values.std() / abs(component_values.mean())) if component_values.mean() != 0 else np.inf
                }
                
        # Calculate component correlations
        if len(component_columns) > 1:
            corr_matrix = contributions[component_columns].corr()
            analysis['component_correlations'] = corr_matrix.to_dict()
            
        # Calculate relative importance
        if 'total_utility' in contributions.columns:
            total_variance = contributions['total_utility'].var()
            for component in component_columns:
                if component in contributions.columns:
                    component_variance = contributions[component].var()
                    analysis['relative_importance'][component] = float(component_variance / total_variance) if total_variance > 0 else 0.0
                    
        return analysis
    
    def analyze_choice_prediction_accuracy(self, utility_values: pd.Series, 
                                         actual_choices: pd.Series,
                                         threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze choice prediction accuracy.
        
        Args:
            utility_values: Predicted utility values
            actual_choices: Actual choice outcomes (0/1)
            threshold: Threshold for converting utilities to binary predictions
            
        Returns:
            Dictionary with prediction accuracy analysis
        """
        logger.info("Analyzing choice prediction accuracy...")
        
        # Convert utilities to probabilities
        probabilities = 1 / (1 + np.exp(-utility_values))
        
        # Convert probabilities to binary predictions
        predictions = (probabilities > threshold).astype(int)
        
        analysis = {
            'threshold': threshold,
            'prediction_stats': {
                'n_observations': len(actual_choices),
                'n_predicted_positive': int(predictions.sum()),
                'n_actual_positive': int(actual_choices.sum()),
                'prediction_rate': float(predictions.mean()),
                'actual_choice_rate': float(actual_choices.mean())
            }
        }
        
        # Calculate accuracy metrics
        try:
            accuracy = (predictions == actual_choices).mean()
            analysis['accuracy_metrics'] = {
                'accuracy': float(accuracy),
                'classification_report': classification_report(actual_choices, predictions, output_dict=True)
            }
            
            # Confusion matrix
            cm = confusion_matrix(actual_choices, predictions)
            analysis['confusion_matrix'] = {
                'matrix': cm.tolist(),
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
            
            # ROC AUC
            if len(np.unique(actual_choices)) > 1:
                auc = roc_auc_score(actual_choices, probabilities)
                analysis['auc'] = float(auc)
                
        except Exception as e:
            logger.warning(f"Could not calculate accuracy metrics: {str(e)}")
            
        return analysis
    
    def analyze_attribute_effects(self, data: pd.DataFrame, utility_values: pd.Series) -> Dict[str, Any]:
        """
        Analyze effects of DCE attributes on utility.
        
        Args:
            data: Input data with DCE attributes
            utility_values: Calculated utility values
            
        Returns:
            Dictionary with attribute effects analysis
        """
        logger.info("Analyzing DCE attribute effects...")
        
        analysis = {}
        
        # Analyze sugar effect
        if 'sugar_free' in data.columns:
            sugar_analysis = self._analyze_binary_attribute_effect(
                data['sugar_free'], utility_values, 'sugar_free'
            )
            analysis['sugar_effect'] = sugar_analysis
            
        # Analyze health label effect
        if 'health_label' in data.columns:
            label_analysis = self._analyze_binary_attribute_effect(
                data['health_label'], utility_values, 'health_label'
            )
            analysis['health_label_effect'] = label_analysis
            
        # Analyze price effect
        price_columns = ['price', 'price_normalized', 'chosen_price']
        for price_col in price_columns:
            if price_col in data.columns:
                price_analysis = self._analyze_continuous_attribute_effect(
                    data[price_col], utility_values, price_col
                )
                analysis['price_effect'] = price_analysis
                break
                
        return analysis
    
    def _analyze_binary_attribute_effect(self, attribute: pd.Series, 
                                       utility: pd.Series, 
                                       attribute_name: str) -> Dict[str, Any]:
        """
        Analyze effect of binary attribute on utility.
        
        Args:
            attribute: Binary attribute values
            utility: Utility values
            attribute_name: Name of attribute
            
        Returns:
            Dictionary with binary attribute analysis
        """
        analysis = {
            'attribute_name': attribute_name,
            'attribute_type': 'binary'
        }
        
        # Split utility by attribute levels
        utility_0 = utility[attribute == 0]
        utility_1 = utility[attribute == 1]
        
        analysis['utility_by_level'] = {
            'level_0': {
                'n_observations': len(utility_0),
                'mean_utility': float(utility_0.mean()) if len(utility_0) > 0 else 0.0,
                'std_utility': float(utility_0.std()) if len(utility_0) > 0 else 0.0
            },
            'level_1': {
                'n_observations': len(utility_1),
                'mean_utility': float(utility_1.mean()) if len(utility_1) > 0 else 0.0,
                'std_utility': float(utility_1.std()) if len(utility_1) > 0 else 0.0
            }
        }
        
        # Calculate effect size
        if len(utility_0) > 0 and len(utility_1) > 0:
            mean_diff = utility_1.mean() - utility_0.mean()
            pooled_std = np.sqrt(((len(utility_0) - 1) * utility_0.var() + 
                                 (len(utility_1) - 1) * utility_1.var()) / 
                                (len(utility_0) + len(utility_1) - 2))
            
            analysis['effect_size'] = {
                'mean_difference': float(mean_diff),
                'cohens_d': float(mean_diff / pooled_std) if pooled_std > 0 else 0.0
            }
            
            # Statistical test
            try:
                t_stat, p_value = stats.ttest_ind(utility_1, utility_0)
                analysis['statistical_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Could not perform t-test for {attribute_name}: {str(e)}")
                
        return analysis
    
    def _analyze_continuous_attribute_effect(self, attribute: pd.Series, 
                                           utility: pd.Series, 
                                           attribute_name: str) -> Dict[str, Any]:
        """
        Analyze effect of continuous attribute on utility.
        
        Args:
            attribute: Continuous attribute values
            utility: Utility values
            attribute_name: Name of attribute
            
        Returns:
            Dictionary with continuous attribute analysis
        """
        analysis = {
            'attribute_name': attribute_name,
            'attribute_type': 'continuous'
        }
        
        # Basic correlation
        correlation = attribute.corr(utility)
        analysis['correlation'] = float(correlation)
        
        # Regression analysis
        try:
            from sklearn.linear_model import LinearRegression
            
            X = attribute.values.reshape(-1, 1)
            y = utility.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            analysis['regression'] = {
                'coefficient': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(model.score(X, y))
            }
            
        except ImportError:
            logger.warning("sklearn not available for regression analysis")
            
        # Quartile analysis
        quartiles = attribute.quantile([0.25, 0.5, 0.75])
        analysis['quartile_effects'] = {}
        
        for i, (q_name, q_value) in enumerate(zip(['Q1', 'Q2', 'Q3', 'Q4'], 
                                                 [0, quartiles.iloc[0], quartiles.iloc[1], quartiles.iloc[2]])):
            if i == 0:
                mask = attribute <= quartiles.iloc[0]
            elif i == 3:
                mask = attribute > quartiles.iloc[2]
            else:
                mask = (attribute > quartiles.iloc[i-1]) & (attribute <= quartiles.iloc[i])
                
            if mask.sum() > 0:
                quartile_utility = utility[mask]
                analysis['quartile_effects'][q_name] = {
                    'n_observations': int(mask.sum()),
                    'mean_utility': float(quartile_utility.mean()),
                    'std_utility': float(quartile_utility.std())
                }
                
        return analysis
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            results: Complete utility calculation results
            
        Returns:
            Dictionary with comprehensive analysis report
        """
        logger.info("Generating comprehensive analysis report...")
        
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'summary': {}
        }
        
        # Utility distribution analysis
        if 'total_utility' in results:
            report['utility_distribution'] = self.analyze_utility_distribution(results['total_utility'])
            
        # Component effects analysis
        if 'decomposition' in results:
            report['component_effects'] = self.analyze_component_effects(results['decomposition'])
            
        # Choice prediction accuracy
        if 'total_utility' in results and 'data' in results:
            data = results['data']
            if 'choice_value' in data.columns:
                report['prediction_accuracy'] = self.analyze_choice_prediction_accuracy(
                    results['total_utility'], data['choice_value']
                )
                
        # Attribute effects analysis
        if 'total_utility' in results and 'data' in results:
            report['attribute_effects'] = self.analyze_attribute_effects(
                results['data'], results['total_utility']
            )
            
        # Generate summary
        report['summary'] = self._generate_analysis_summary(report)
        
        logger.info("Comprehensive analysis report generated")
        return report
    
    def _generate_analysis_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of analysis results.
        
        Args:
            report: Analysis report
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        # Utility summary
        if 'utility_distribution' in report:
            util_dist = report['utility_distribution']
            summary['utility_summary'] = {
                'mean_utility': util_dist['descriptive_stats']['mean'],
                'utility_range': util_dist['descriptive_stats']['max'] - util_dist['descriptive_stats']['min'],
                'is_normally_distributed': util_dist.get('normality_test', {}).get('is_normal', False)
            }
            
        # Model performance summary
        if 'prediction_accuracy' in report:
            pred_acc = report['prediction_accuracy']
            summary['model_performance'] = {
                'accuracy': pred_acc.get('accuracy_metrics', {}).get('accuracy', 0.0),
                'auc': pred_acc.get('auc', 0.0)
            }
            
        # Key findings
        summary['key_findings'] = []
        
        # Add findings based on analysis
        if 'attribute_effects' in report:
            attr_effects = report['attribute_effects']
            
            # Sugar effect finding
            if 'sugar_effect' in attr_effects:
                sugar_effect = attr_effects['sugar_effect']
                if 'effect_size' in sugar_effect:
                    effect_size = sugar_effect['effect_size']['cohens_d']
                    if abs(effect_size) > 0.5:
                        summary['key_findings'].append(
                            f"Strong sugar preference effect detected (Cohen's d = {effect_size:.3f})"
                        )
                        
            # Health label effect finding
            if 'health_label_effect' in attr_effects:
                label_effect = attr_effects['health_label_effect']
                if 'effect_size' in label_effect:
                    effect_size = label_effect['effect_size']['cohens_d']
                    if abs(effect_size) > 0.3:
                        summary['key_findings'].append(
                            f"Moderate health label effect detected (Cohen's d = {effect_size:.3f})"
                        )
                        
        return summary
