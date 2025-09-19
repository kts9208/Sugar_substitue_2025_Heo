"""
Health benefit factor integrator for SEM results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .factor_integrator import FactorIntegrator
from ..config.settings import PERCEIVED_BENEFIT_FILE

logger = logging.getLogger(__name__)


class HealthBenefitIntegrator(FactorIntegrator):
    """
    Integrator for health benefit (perceived benefit) factor from SEM results.
    
    Handles the integration of perceived health benefits into utility calculations,
    including direct effects and interactions with DCE attributes.
    """
    
    def __init__(self):
        """Initialize health benefit integrator."""
        super().__init__(factor_name="perceived_benefit")
        self.benefit_coefficients = {}
        
    def integrate_factor_effects(self, dce_data: pd.DataFrame, 
                                sem_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate health benefit effects into DCE data.
        
        Args:
            dce_data: DCE choice data
            sem_results: SEM analysis results
            
        Returns:
            Enhanced DCE data with health benefit effects
        """
        logger.info("Integrating health benefit effects...")
        
        # Extract path coefficients
        self.extract_path_coefficients(sem_results)
        
        # Load factor scores if not already loaded
        if self.factor_scores is None:
            self._load_benefit_scores()
            
        # Merge factor scores with DCE data
        enhanced_data = self._merge_factor_scores(dce_data)
        
        # Add benefit-related utility components
        enhanced_data = self._add_benefit_utility_components(enhanced_data)
        
        self.is_fitted = True
        logger.info("Health benefit integration completed")
        
        return enhanced_data
    
    def _load_benefit_scores(self):
        """Load perceived benefit factor scores."""
        try:
            benefit_data = pd.read_csv(PERCEIVED_BENEFIT_FILE)
            
            # Calculate factor scores from benefit items
            benefit_columns = [col for col in benefit_data.columns 
                             if col.startswith('q') and col not in ['respondent_id']]
            
            if benefit_columns:
                benefit_scores = benefit_data[benefit_columns].mean(axis=1)
                
                self.factor_scores = pd.DataFrame({
                    'respondent_id': benefit_data.get('respondent_id', benefit_data.index),
                    'perceived_benefit_score': benefit_scores
                })
                
                logger.info(f"Loaded benefit scores for {len(self.factor_scores)} respondents")
            else:
                logger.warning("No benefit items found in data")
                
        except Exception as e:
            logger.error(f"Error loading benefit scores: {str(e)}")
            
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
        if 'perceived_benefit_score' in enhanced_data.columns:
            mean_score = enhanced_data['perceived_benefit_score'].mean()
            enhanced_data['perceived_benefit_score'].fillna(mean_score, inplace=True)
            
        return enhanced_data
    
    def _add_benefit_utility_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add benefit-related utility components to data.
        
        Args:
            data: Data with factor scores
            
        Returns:
            Data with additional utility components
        """
        data = data.copy()
        
        if 'perceived_benefit_score' not in data.columns:
            return data
            
        # Direct benefit effect
        data['benefit_direct_utility'] = data['perceived_benefit_score']
        
        # Benefit × Sugar interaction
        if 'sugar_free' in data.columns:
            data['benefit_sugar_interaction'] = (
                data['perceived_benefit_score'] * data['sugar_free']
            )
            
        # Benefit × Health Label interaction
        if 'health_label' in data.columns:
            data['benefit_label_interaction'] = (
                data['perceived_benefit_score'] * data['health_label']
            )
            
        # Benefit × Price interaction
        price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
        if price_col in data.columns:
            data['benefit_price_interaction'] = (
                data['perceived_benefit_score'] * data[price_col]
            )
            
        return data
    
    def calculate_factor_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from health benefit factor.
        
        Args:
            data: Data with benefit information
            **kwargs: Additional parameters including coefficients
            
        Returns:
            Series with benefit utility values
        """
        if not self.validate_factor_data(data):
            logger.warning("Invalid factor data, returning zero utility")
            return pd.Series(0.0, index=data.index)
            
        # Get coefficients from kwargs or use defaults from SEM results
        direct_coeff = kwargs.get('benefit_direct_coeff', 
                                 self.path_coefficients.get('perceived_benefit_to_purchase_intention', 0.0))
        
        sugar_interaction_coeff = kwargs.get('benefit_sugar_coeff', 0.0)
        label_interaction_coeff = kwargs.get('benefit_label_coeff', 0.0)
        price_interaction_coeff = kwargs.get('benefit_price_coeff', 0.0)
        
        # Calculate utility components
        utility = pd.Series(0.0, index=data.index)
        
        # Direct benefit effect
        if 'perceived_benefit_score' in data.columns:
            utility += direct_coeff * data['perceived_benefit_score']
            
        # Interaction effects
        if 'benefit_sugar_interaction' in data.columns:
            utility += sugar_interaction_coeff * data['benefit_sugar_interaction']
            
        if 'benefit_label_interaction' in data.columns:
            utility += label_interaction_coeff * data['benefit_label_interaction']
            
        if 'benefit_price_interaction' in data.columns:
            utility += price_interaction_coeff * data['benefit_price_interaction']
            
        logger.debug(f"Health benefit utility calculated for {len(data)} observations")
        return utility
    
    def _get_factor_columns(self, survey_data: pd.DataFrame) -> List[str]:
        """
        Get column names for perceived benefit factor.
        
        Args:
            survey_data: Survey data
            
        Returns:
            List of benefit-related column names
        """
        # Perceived benefit items are typically q12-q17
        benefit_columns = []
        for col in survey_data.columns:
            if col.startswith('q') and col[1:].isdigit():
                q_num = int(col[1:])
                if 12 <= q_num <= 17:  # Benefit items
                    benefit_columns.append(col)
                    
        return benefit_columns
    
    def get_benefit_effects_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of benefit effects in the data.
        
        Args:
            data: Data with benefit information
            
        Returns:
            Dictionary with benefit effects summary
        """
        summary = self.get_factor_summary(data)
        
        # Add benefit-specific statistics
        if 'perceived_benefit_score' in data.columns:
            benefit_scores = data['perceived_benefit_score']
            
            # Correlation with choice if available
            if 'choice_value' in data.columns:
                summary['benefit_choice_correlation'] = benefit_scores.corr(data['choice_value'])
                
            # Benefit levels by DCE attributes
            if 'sugar_free' in data.columns:
                summary['benefit_by_sugar'] = {
                    'sugar_free': data[data['sugar_free'] == 1]['perceived_benefit_score'].mean(),
                    'regular_sugar': data[data['sugar_free'] == 0]['perceived_benefit_score'].mean()
                }
                
            if 'health_label' in data.columns:
                summary['benefit_by_label'] = {
                    'with_label': data[data['health_label'] == 1]['perceived_benefit_score'].mean(),
                    'without_label': data[data['health_label'] == 0]['perceived_benefit_score'].mean()
                }
                
        return summary
    
    def set_benefit_coefficients(self, direct_coeff: float = 0.0,
                               sugar_interaction_coeff: float = 0.0,
                               label_interaction_coeff: float = 0.0,
                               price_interaction_coeff: float = 0.0):
        """
        Set benefit-related coefficients.
        
        Args:
            direct_coeff: Direct benefit effect coefficient
            sugar_interaction_coeff: Benefit × sugar interaction coefficient
            label_interaction_coeff: Benefit × label interaction coefficient
            price_interaction_coeff: Benefit × price interaction coefficient
        """
        self.benefit_coefficients = {
            'direct': direct_coeff,
            'sugar_interaction': sugar_interaction_coeff,
            'label_interaction': label_interaction_coeff,
            'price_interaction': price_interaction_coeff
        }
        
    def get_integrator_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the health benefit integrator.
        
        Returns:
            Dictionary with integrator information
        """
        info = super().get_integrator_info()
        info.update({
            'benefit_coefficients': self.benefit_coefficients,
            'factor_file': str(PERCEIVED_BENEFIT_FILE)
        })
        return info
