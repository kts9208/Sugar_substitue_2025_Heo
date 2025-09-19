"""
Perceived price factor integrator for SEM results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .factor_integrator import FactorIntegrator
from ..config.settings import PERCEIVED_PRICE_FILE

logger = logging.getLogger(__name__)


class PerceivedPriceIntegrator(FactorIntegrator):
    """
    Integrator for perceived price factor from SEM results.
    
    Handles the integration of perceived price levels into utility calculations,
    including effects on choice behavior and interactions with actual prices.
    """
    
    def __init__(self):
        """Initialize perceived price integrator."""
        super().__init__(factor_name="perceived_price")
        self.price_coefficients = {}
        
    def integrate_factor_effects(self, dce_data: pd.DataFrame, 
                                sem_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate perceived price effects into DCE data.
        
        Args:
            dce_data: DCE choice data
            sem_results: SEM analysis results
            
        Returns:
            Enhanced DCE data with perceived price effects
        """
        logger.info("Integrating perceived price effects...")
        
        # Extract path coefficients
        self.extract_path_coefficients(sem_results)
        
        # Load factor scores if not already loaded
        if self.factor_scores is None:
            self._load_price_perception_scores()
            
        # Merge factor scores with DCE data
        enhanced_data = self._merge_factor_scores(dce_data)
        
        # Add price perception utility components
        enhanced_data = self._add_price_perception_utility_components(enhanced_data)
        
        self.is_fitted = True
        logger.info("Perceived price integration completed")
        
        return enhanced_data
    
    def _load_price_perception_scores(self):
        """Load perceived price factor scores."""
        try:
            price_data = pd.read_csv(PERCEIVED_PRICE_FILE)
            
            # Calculate factor scores from price perception items
            price_columns = [col for col in price_data.columns 
                           if col.startswith('q') and col not in ['respondent_id']]
            
            if price_columns:
                # For price perception items, calculate mean score
                price_scores = price_data[price_columns].mean(axis=1)
                
                self.factor_scores = pd.DataFrame({
                    'respondent_id': price_data.get('respondent_id', price_data.index),
                    'perceived_price_score': price_scores
                })
                
                logger.info(f"Loaded price perception scores for {len(self.factor_scores)} respondents")
            else:
                logger.warning("No price perception items found in data")
                
        except Exception as e:
            logger.error(f"Error loading price perception scores: {str(e)}")
            
    def _merge_factor_scores(self, dce_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge factor scores with DCE data.
        
        Args:
            dce_data: DCE choice data
            
        Returns:
            Data with merged factor scores
        """
        if self.factor_scores is None:
            logger.warning("No factor scores available for merging")
            return dce_data
            
        # Merge on respondent_id
        enhanced_data = dce_data.merge(
            self.factor_scores, 
            on='respondent_id', 
            how='left'
        )
        
        # Fill missing values with mean
        if 'perceived_price_score' in enhanced_data.columns:
            mean_score = enhanced_data['perceived_price_score'].mean()
            enhanced_data['perceived_price_score'].fillna(mean_score, inplace=True)
            
        return enhanced_data
    
    def _add_price_perception_utility_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price perception utility components to data.
        
        Args:
            data: Data with factor scores
            
        Returns:
            Data with additional utility components
        """
        data = data.copy()
        
        if 'perceived_price_score' not in data.columns:
            return data
            
        # Direct price perception effect
        data['price_perception_direct_utility'] = data['perceived_price_score']
        
        # Price perception × Actual Price interaction
        price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
        if price_col in data.columns:
            data['price_perception_actual_interaction'] = (
                data['perceived_price_score'] * data[price_col]
            )
            
        # Price perception × Sugar interaction (price sensitivity may vary by sugar type)
        if 'sugar_free' in data.columns:
            data['price_perception_sugar_interaction'] = (
                data['perceived_price_score'] * data['sugar_free']
            )
            
        # Price perception × Health Label interaction
        if 'health_label' in data.columns:
            data['price_perception_label_interaction'] = (
                data['perceived_price_score'] * data['health_label']
            )
            
        # Create price sensitivity categories
        if 'perceived_price_score' in data.columns:
            price_scores = data['perceived_price_score']
            data['price_sensitivity_level'] = pd.cut(
                price_scores,
                bins=[0, 0.33, 0.67, 1.0],
                labels=['Low_Sensitivity', 'Medium_Sensitivity', 'High_Sensitivity'],
                include_lowest=True
            )
            
        return data
    
    def calculate_factor_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from perceived price factor.
        
        Args:
            data: Data with price perception information
            **kwargs: Additional parameters including coefficients
            
        Returns:
            Series with price perception utility values
        """
        if not self.validate_factor_data(data):
            logger.warning("Invalid factor data, returning zero utility")
            return pd.Series(0.0, index=data.index)
            
        # Get coefficients from kwargs or use defaults from SEM results
        direct_coeff = kwargs.get('price_perception_direct_coeff', 
                                 self.path_coefficients.get('perceived_price_to_purchase_intention', 0.0))
        
        actual_price_interaction_coeff = kwargs.get('price_perception_actual_coeff', 0.0)
        sugar_interaction_coeff = kwargs.get('price_perception_sugar_coeff', 0.0)
        label_interaction_coeff = kwargs.get('price_perception_label_coeff', 0.0)
        
        # Calculate utility components
        utility = pd.Series(0.0, index=data.index)
        
        # Direct price perception effect
        if 'perceived_price_score' in data.columns:
            utility += direct_coeff * data['perceived_price_score']
            
        # Interaction effects
        if 'price_perception_actual_interaction' in data.columns:
            utility += actual_price_interaction_coeff * data['price_perception_actual_interaction']
            
        if 'price_perception_sugar_interaction' in data.columns:
            utility += sugar_interaction_coeff * data['price_perception_sugar_interaction']
            
        if 'price_perception_label_interaction' in data.columns:
            utility += label_interaction_coeff * data['price_perception_label_interaction']
            
        logger.debug(f"Perceived price utility calculated for {len(data)} observations")
        return utility
    
    def _get_factor_columns(self, survey_data: pd.DataFrame) -> List[str]:
        """
        Get column names for perceived price factor.
        
        Args:
            survey_data: Survey data
            
        Returns:
            List of price perception column names
        """
        # Perceived price items are typically q27-q29
        price_columns = []
        for col in survey_data.columns:
            if col.startswith('q') and col[1:].isdigit():
                q_num = int(col[1:])
                if 27 <= q_num <= 29:  # Price perception items
                    price_columns.append(col)
                    
        return price_columns
    
    def get_price_perception_effects_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of price perception effects in the data.
        
        Args:
            data: Data with price perception information
            
        Returns:
            Dictionary with price perception effects summary
        """
        summary = self.get_factor_summary(data)
        
        # Add price perception specific statistics
        if 'perceived_price_score' in data.columns:
            price_scores = data['perceived_price_score']
            
            # Correlation with choice if available
            if 'choice_value' in data.columns:
                summary['price_perception_choice_correlation'] = price_scores.corr(data['choice_value'])
                
            # Price sensitivity distribution
            if 'price_sensitivity_level' in data.columns:
                summary['price_sensitivity_distribution'] = data['price_sensitivity_level'].value_counts().to_dict()
                
            # Correlation with actual price
            price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
            if price_col in data.columns:
                summary['perceived_actual_price_correlation'] = price_scores.corr(data[price_col])
                
            # Price perception by DCE attributes
            if 'sugar_free' in data.columns:
                summary['price_perception_by_sugar'] = {
                    'sugar_free_choices': data[data['sugar_free'] == 1]['perceived_price_score'].mean(),
                    'regular_sugar_choices': data[data['sugar_free'] == 0]['perceived_price_score'].mean()
                }
                
            if 'health_label' in data.columns:
                summary['price_perception_by_label'] = {
                    'with_label_choices': data[data['health_label'] == 1]['perceived_price_score'].mean(),
                    'without_label_choices': data[data['health_label'] == 0]['perceived_price_score'].mean()
                }
                
        return summary
    
    def analyze_price_sensitivity_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price sensitivity patterns in choice behavior.
        
        Args:
            data: Data with price perception and choice information
            
        Returns:
            Dictionary with price sensitivity analysis
        """
        if 'perceived_price_score' not in data.columns or 'choice_value' not in data.columns:
            return {}
            
        analysis = {}
        
        # Price elasticity by sensitivity level
        if 'price_sensitivity_level' in data.columns:
            price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
            if price_col in data.columns:
                elasticity_by_level = {}
                for level in data['price_sensitivity_level'].unique():
                    if pd.notna(level):
                        level_data = data[data['price_sensitivity_level'] == level]
                        if len(level_data) > 10:  # Minimum sample size
                            correlation = level_data[price_col].corr(level_data['choice_value'])
                            elasticity_by_level[level] = correlation
                            
                analysis['price_elasticity_by_sensitivity'] = elasticity_by_level
                
        # Choice patterns by price perception tertiles
        price_scores = data['perceived_price_score']
        tertiles = price_scores.quantile([0.33, 0.67])
        
        data_copy = data.copy()
        data_copy['price_perception_tertile'] = pd.cut(
            price_scores,
            bins=[0, tertiles.iloc[0], tertiles.iloc[1], price_scores.max()],
            labels=['Low_Perception', 'Medium_Perception', 'High_Perception'],
            include_lowest=True
        )
        
        choice_by_tertile = data_copy.groupby('price_perception_tertile')['choice_value'].agg(['mean', 'count'])
        analysis['choice_rates_by_price_perception'] = choice_by_tertile.to_dict()
        
        return analysis
    
    def set_price_perception_coefficients(self, direct_coeff: float = 0.0,
                                        actual_price_interaction_coeff: float = 0.0,
                                        sugar_interaction_coeff: float = 0.0,
                                        label_interaction_coeff: float = 0.0):
        """
        Set price perception coefficients.
        
        Args:
            direct_coeff: Direct price perception effect coefficient
            actual_price_interaction_coeff: Perceived × actual price interaction coefficient
            sugar_interaction_coeff: Price perception × sugar interaction coefficient
            label_interaction_coeff: Price perception × label interaction coefficient
        """
        self.price_coefficients = {
            'direct': direct_coeff,
            'actual_price_interaction': actual_price_interaction_coeff,
            'sugar_interaction': sugar_interaction_coeff,
            'label_interaction': label_interaction_coeff
        }
        
    def get_integrator_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the perceived price integrator.
        
        Returns:
            Dictionary with integrator information
        """
        info = super().get_integrator_info()
        info.update({
            'price_coefficients': self.price_coefficients,
            'factor_file': str(PERCEIVED_PRICE_FILE)
        })
        return info
