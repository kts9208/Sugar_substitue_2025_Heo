"""
Comprehensive SEM utility component that integrates all SEM factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .health_benefit_integrator import HealthBenefitIntegrator
from .nutrition_knowledge_integrator import NutritionKnowledgeIntegrator
from .perceived_price_integrator import PerceivedPriceIntegrator
from ..components.base_component import BaseUtilityComponent

logger = logging.getLogger(__name__)


class SEMUtilityComponent(BaseUtilityComponent):
    """
    Comprehensive utility component that integrates all SEM factors.
    
    Combines health benefits, nutrition knowledge, and perceived price effects
    from SEM analysis into a unified utility contribution.
    """
    
    def __init__(self):
        """Initialize SEM utility component."""
        super().__init__(name="sem_utility_component", coefficient=1.0)
        
        # Initialize factor integrators
        self.health_benefit_integrator = HealthBenefitIntegrator()
        self.nutrition_knowledge_integrator = NutritionKnowledgeIntegrator()
        self.perceived_price_integrator = PerceivedPriceIntegrator()
        
        self.sem_coefficients = {}
        self.integrated_data = None
        
    def integrate_all_sem_factors(self, dce_data: pd.DataFrame, 
                                 sem_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate all SEM factors into DCE data.
        
        Args:
            dce_data: DCE choice data
            sem_results: SEM analysis results
            
        Returns:
            Enhanced DCE data with all SEM factor effects
        """
        logger.info("Integrating all SEM factors...")
        
        # Start with original DCE data
        integrated_data = dce_data.copy()
        
        # Integrate each factor
        integrated_data = self.health_benefit_integrator.integrate_factor_effects(
            integrated_data, sem_results
        )
        
        integrated_data = self.nutrition_knowledge_integrator.integrate_factor_effects(
            integrated_data, sem_results
        )
        
        integrated_data = self.perceived_price_integrator.integrate_factor_effects(
            integrated_data, sem_results
        )
        
        # Store integrated data
        self.integrated_data = integrated_data
        
        # Extract and store SEM coefficients
        self._extract_sem_coefficients(sem_results)
        
        self.is_fitted = True
        logger.info("All SEM factors integrated successfully")
        
        return integrated_data
    
    def _extract_sem_coefficients(self, sem_results: Dict[str, Any]):
        """
        Extract SEM coefficients for utility calculation.
        
        Args:
            sem_results: SEM analysis results
        """
        if 'factor_effects' in sem_results:
            factor_effects = sem_results['factor_effects']
            
            # Health benefit effects
            if 'health_benefits' in factor_effects:
                self.sem_coefficients.update({
                    'health_benefit_to_purchase': factor_effects['health_benefits'].get('benefit_to_purchase', 0.0),
                    'health_to_benefit': factor_effects['health_benefits'].get('health_to_benefit', 0.0)
                })
                
            # Nutrition knowledge effects
            if 'nutrition_knowledge' in factor_effects:
                self.sem_coefficients.update({
                    'nutrition_to_benefit': factor_effects['nutrition_knowledge'].get('nutrition_to_benefit', 0.0),
                    'nutrition_to_purchase': factor_effects['nutrition_knowledge'].get('nutrition_to_purchase', 0.0)
                })
                
            # Perceived price effects
            if 'perceived_price' in factor_effects:
                self.sem_coefficients.update({
                    'price_to_benefit': factor_effects['perceived_price'].get('price_to_benefit', 0.0),
                    'price_to_purchase': factor_effects['perceived_price'].get('price_to_purchase', 0.0)
                })
                
    def calculate_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate total SEM utility contribution.
        
        Args:
            data: Data with SEM factor information
            **kwargs: Additional parameters including coefficients
            
        Returns:
            Series with total SEM utility values
        """
        if not self.is_fitted:
            logger.warning("SEM component not fitted, returning zero utility")
            return pd.Series(0.0, index=data.index)
            
        total_utility = pd.Series(0.0, index=data.index)
        
        # Health benefit utility
        try:
            benefit_utility = self.health_benefit_integrator.calculate_factor_utility(
                data, **kwargs
            )
            total_utility += benefit_utility
        except Exception as e:
            logger.warning(f"Could not calculate health benefit utility: {str(e)}")
            
        # Nutrition knowledge utility
        try:
            knowledge_utility = self.nutrition_knowledge_integrator.calculate_factor_utility(
                data, **kwargs
            )
            total_utility += knowledge_utility
        except Exception as e:
            logger.warning(f"Could not calculate nutrition knowledge utility: {str(e)}")
            
        # Perceived price utility
        try:
            price_utility = self.perceived_price_integrator.calculate_factor_utility(
                data, **kwargs
            )
            total_utility += price_utility
        except Exception as e:
            logger.warning(f"Could not calculate perceived price utility: {str(e)}")
            
        # Apply overall SEM coefficient
        total_utility *= self.coefficient
        
        logger.debug(f"Total SEM utility calculated for {len(data)} observations")
        return total_utility
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'SEMUtilityComponent':
        """
        Fit SEM utility component.
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # The fitting is done through the integration process
        # This method is here for consistency with base class
        if self.integrated_data is not None:
            self.is_fitted = True
        else:
            logger.warning("SEM factors not integrated yet. Call integrate_all_sem_factors first.")
            
        return self
    
    def get_sem_effects_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of all SEM effects.
        
        Args:
            data: Data with SEM factor information
            
        Returns:
            Dictionary with comprehensive SEM effects summary
        """
        summary = {
            'sem_coefficients': self.sem_coefficients,
            'health_benefit_effects': self.health_benefit_integrator.get_benefit_effects_summary(data),
            'nutrition_knowledge_effects': self.nutrition_knowledge_integrator.get_knowledge_effects_summary(data),
            'perceived_price_effects': self.perceived_price_integrator.get_price_perception_effects_summary(data)
        }
        
        # Add cross-factor correlations if all factors are present
        factor_columns = ['perceived_benefit_score', 'nutrition_knowledge_score', 'perceived_price_score']
        available_factors = [col for col in factor_columns if col in data.columns]
        
        if len(available_factors) > 1:
            factor_correlations = data[available_factors].corr()
            summary['factor_correlations'] = factor_correlations.to_dict()
            
        return summary
    
    def analyze_sem_mediation_effects(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze mediation effects between SEM factors.
        
        Args:
            data: Data with SEM factors and choice information
            
        Returns:
            Dictionary with mediation analysis results
        """
        if 'choice_value' not in data.columns:
            return {}
            
        mediation_analysis = {}
        
        # Health concern -> Perceived benefit -> Purchase intention
        if all(col in data.columns for col in ['health_concern_score', 'perceived_benefit_score', 'choice_value']):
            # Direct effect
            direct_effect = data['health_concern_score'].corr(data['choice_value'])
            
            # Indirect effect through perceived benefit
            health_to_benefit = data['health_concern_score'].corr(data['perceived_benefit_score'])
            benefit_to_choice = data['perceived_benefit_score'].corr(data['choice_value'])
            indirect_effect = health_to_benefit * benefit_to_choice
            
            mediation_analysis['health_to_choice_via_benefit'] = {
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'total_effect': direct_effect + indirect_effect
            }
            
        # Nutrition knowledge -> Perceived benefit -> Purchase intention
        if all(col in data.columns for col in ['nutrition_knowledge_score', 'perceived_benefit_score', 'choice_value']):
            direct_effect = data['nutrition_knowledge_score'].corr(data['choice_value'])
            
            knowledge_to_benefit = data['nutrition_knowledge_score'].corr(data['perceived_benefit_score'])
            benefit_to_choice = data['perceived_benefit_score'].corr(data['choice_value'])
            indirect_effect = knowledge_to_benefit * benefit_to_choice
            
            mediation_analysis['knowledge_to_choice_via_benefit'] = {
                'direct_effect': direct_effect,
                'indirect_effect': indirect_effect,
                'total_effect': direct_effect + indirect_effect
            }
            
        return mediation_analysis
    
    def set_sem_coefficients(self, coefficients: Dict[str, float]):
        """
        Set SEM coefficients manually.
        
        Args:
            coefficients: Dictionary of SEM coefficients
        """
        self.sem_coefficients.update(coefficients)
        
        # Update individual integrator coefficients
        if 'health_benefit_to_purchase' in coefficients:
            self.health_benefit_integrator.set_benefit_coefficients(
                direct_coeff=coefficients['health_benefit_to_purchase']
            )
            
        if 'nutrition_to_purchase' in coefficients:
            self.nutrition_knowledge_integrator.set_knowledge_coefficients(
                direct_coeff=coefficients['nutrition_to_purchase']
            )
            
        if 'price_to_purchase' in coefficients:
            self.perceived_price_integrator.set_price_perception_coefficients(
                direct_coeff=coefficients['price_to_purchase']
            )
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the SEM utility component.
        
        Returns:
            Dictionary with component information
        """
        info = super().get_component_info()
        info.update({
            'sem_coefficients': self.sem_coefficients,
            'health_benefit_integrator': self.health_benefit_integrator.get_integrator_info(),
            'nutrition_knowledge_integrator': self.nutrition_knowledge_integrator.get_integrator_info(),
            'perceived_price_integrator': self.perceived_price_integrator.get_integrator_info(),
            'has_integrated_data': self.integrated_data is not None
        })
        return info
