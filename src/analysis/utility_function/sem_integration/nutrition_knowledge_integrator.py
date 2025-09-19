"""
Nutrition knowledge factor integrator for SEM results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .factor_integrator import FactorIntegrator
from ..config.settings import NUTRITION_KNOWLEDGE_FILE

logger = logging.getLogger(__name__)


class NutritionKnowledgeIntegrator(FactorIntegrator):
    """
    Integrator for nutrition knowledge factor from SEM results.
    
    Handles the integration of nutrition knowledge levels into utility calculations,
    including effects on choice behavior and interactions with DCE attributes.
    """
    
    def __init__(self):
        """Initialize nutrition knowledge integrator."""
        super().__init__(factor_name="nutrition_knowledge")
        self.knowledge_coefficients = {}
        
    def integrate_factor_effects(self, dce_data: pd.DataFrame, 
                                sem_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate nutrition knowledge effects into DCE data.
        
        Args:
            dce_data: DCE choice data
            sem_results: SEM analysis results
            
        Returns:
            Enhanced DCE data with nutrition knowledge effects
        """
        logger.info("Integrating nutrition knowledge effects...")
        
        # Extract path coefficients
        self.extract_path_coefficients(sem_results)
        
        # Load factor scores if not already loaded
        if self.factor_scores is None:
            self._load_knowledge_scores()
            
        # Merge factor scores with DCE data
        enhanced_data = self._merge_factor_scores(dce_data)
        
        # Add knowledge-related utility components
        enhanced_data = self._add_knowledge_utility_components(enhanced_data)
        
        self.is_fitted = True
        logger.info("Nutrition knowledge integration completed")
        
        return enhanced_data
    
    def _load_knowledge_scores(self):
        """Load nutrition knowledge factor scores."""
        try:
            knowledge_data = pd.read_csv(NUTRITION_KNOWLEDGE_FILE)
            
            # Calculate factor scores from knowledge items
            knowledge_columns = [col for col in knowledge_data.columns 
                               if col.startswith('q') and col not in ['respondent_id']]
            
            if knowledge_columns:
                # For knowledge items, calculate proportion of correct answers
                # Assuming correct answers are coded as 1
                knowledge_scores = knowledge_data[knowledge_columns].mean(axis=1)
                
                self.factor_scores = pd.DataFrame({
                    'respondent_id': knowledge_data.get('respondent_id', knowledge_data.index),
                    'nutrition_knowledge_score': knowledge_scores
                })
                
                logger.info(f"Loaded knowledge scores for {len(self.factor_scores)} respondents")
            else:
                logger.warning("No knowledge items found in data")
                
        except Exception as e:
            logger.error(f"Error loading knowledge scores: {str(e)}")
            
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
        if 'nutrition_knowledge_score' in enhanced_data.columns:
            mean_score = enhanced_data['nutrition_knowledge_score'].mean()
            enhanced_data['nutrition_knowledge_score'].fillna(mean_score, inplace=True)
            
        return enhanced_data
    
    def _add_knowledge_utility_components(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add knowledge-related utility components to data.
        
        Args:
            data: Data with factor scores
            
        Returns:
            Data with additional utility components
        """
        data = data.copy()
        
        if 'nutrition_knowledge_score' not in data.columns:
            return data
            
        # Direct knowledge effect
        data['knowledge_direct_utility'] = data['nutrition_knowledge_score']
        
        # Knowledge × Sugar interaction (knowledgeable consumers may prefer sugar-free)
        if 'sugar_free' in data.columns:
            data['knowledge_sugar_interaction'] = (
                data['nutrition_knowledge_score'] * data['sugar_free']
            )
            
        # Knowledge × Health Label interaction (knowledgeable consumers may value labels differently)
        if 'health_label' in data.columns:
            data['knowledge_label_interaction'] = (
                data['nutrition_knowledge_score'] * data['health_label']
            )
            
        # Knowledge × Price interaction (knowledgeable consumers may be less price sensitive)
        price_col = 'price_normalized' if 'price_normalized' in data.columns else 'price'
        if price_col in data.columns:
            data['knowledge_price_interaction'] = (
                data['nutrition_knowledge_score'] * data[price_col]
            )
            
        # Create knowledge level categories for analysis
        if 'nutrition_knowledge_score' in data.columns:
            knowledge_scores = data['nutrition_knowledge_score']
            data['knowledge_level'] = pd.cut(
                knowledge_scores,
                bins=[0, 0.33, 0.67, 1.0],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
            
        return data
    
    def calculate_factor_utility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate utility contribution from nutrition knowledge factor.
        
        Args:
            data: Data with knowledge information
            **kwargs: Additional parameters including coefficients
            
        Returns:
            Series with knowledge utility values
        """
        if not self.validate_factor_data(data):
            logger.warning("Invalid factor data, returning zero utility")
            return pd.Series(0.0, index=data.index)
            
        # Get coefficients from kwargs or use defaults from SEM results
        direct_coeff = kwargs.get('knowledge_direct_coeff', 
                                 self.path_coefficients.get('nutrition_knowledge_to_purchase_intention', 0.0))
        
        sugar_interaction_coeff = kwargs.get('knowledge_sugar_coeff', 0.0)
        label_interaction_coeff = kwargs.get('knowledge_label_coeff', 0.0)
        price_interaction_coeff = kwargs.get('knowledge_price_coeff', 0.0)
        
        # Calculate utility components
        utility = pd.Series(0.0, index=data.index)
        
        # Direct knowledge effect
        if 'nutrition_knowledge_score' in data.columns:
            utility += direct_coeff * data['nutrition_knowledge_score']
            
        # Interaction effects
        if 'knowledge_sugar_interaction' in data.columns:
            utility += sugar_interaction_coeff * data['knowledge_sugar_interaction']
            
        if 'knowledge_label_interaction' in data.columns:
            utility += label_interaction_coeff * data['knowledge_label_interaction']
            
        if 'knowledge_price_interaction' in data.columns:
            utility += price_interaction_coeff * data['knowledge_price_interaction']
            
        logger.debug(f"Nutrition knowledge utility calculated for {len(data)} observations")
        return utility
    
    def _get_factor_columns(self, survey_data: pd.DataFrame) -> List[str]:
        """
        Get column names for nutrition knowledge factor.
        
        Args:
            survey_data: Survey data
            
        Returns:
            List of knowledge-related column names
        """
        # Nutrition knowledge items are typically q30-q49
        knowledge_columns = []
        for col in survey_data.columns:
            if col.startswith('q') and col[1:].isdigit():
                q_num = int(col[1:])
                if 30 <= q_num <= 49:  # Knowledge items
                    knowledge_columns.append(col)
                    
        return knowledge_columns
    
    def get_knowledge_effects_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of knowledge effects in the data.
        
        Args:
            data: Data with knowledge information
            
        Returns:
            Dictionary with knowledge effects summary
        """
        summary = self.get_factor_summary(data)
        
        # Add knowledge-specific statistics
        if 'nutrition_knowledge_score' in data.columns:
            knowledge_scores = data['nutrition_knowledge_score']
            
            # Correlation with choice if available
            if 'choice_value' in data.columns:
                summary['knowledge_choice_correlation'] = knowledge_scores.corr(data['choice_value'])
                
            # Knowledge levels distribution
            if 'knowledge_level' in data.columns:
                summary['knowledge_level_distribution'] = data['knowledge_level'].value_counts().to_dict()
                
            # Knowledge effects by DCE attributes
            if 'sugar_free' in data.columns:
                summary['knowledge_by_sugar'] = {
                    'sugar_free_choices': data[data['sugar_free'] == 1]['nutrition_knowledge_score'].mean(),
                    'regular_sugar_choices': data[data['sugar_free'] == 0]['nutrition_knowledge_score'].mean()
                }
                
            if 'health_label' in data.columns:
                summary['knowledge_by_label'] = {
                    'with_label_choices': data[data['health_label'] == 1]['nutrition_knowledge_score'].mean(),
                    'without_label_choices': data[data['health_label'] == 0]['nutrition_knowledge_score'].mean()
                }
                
            # Choice patterns by knowledge level
            if 'choice_value' in data.columns and 'knowledge_level' in data.columns:
                choice_by_knowledge = data.groupby('knowledge_level')['choice_value'].mean()
                summary['choice_rate_by_knowledge_level'] = choice_by_knowledge.to_dict()
                
        return summary
    
    def analyze_knowledge_choice_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze choice patterns by nutrition knowledge level.
        
        Args:
            data: Data with knowledge and choice information
            
        Returns:
            Dictionary with choice pattern analysis
        """
        if 'nutrition_knowledge_score' not in data.columns or 'choice_value' not in data.columns:
            return {}
            
        # Create knowledge tertiles
        knowledge_scores = data['nutrition_knowledge_score']
        tertiles = knowledge_scores.quantile([0.33, 0.67])
        
        data_copy = data.copy()
        data_copy['knowledge_tertile'] = pd.cut(
            knowledge_scores,
            bins=[0, tertiles.iloc[0], tertiles.iloc[1], 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        analysis = {}
        
        # Choice rates by knowledge tertile
        choice_by_tertile = data_copy.groupby('knowledge_tertile')['choice_value'].agg(['mean', 'count'])
        analysis['choice_rates_by_tertile'] = choice_by_tertile.to_dict()
        
        # Attribute preferences by knowledge level
        for attribute in ['sugar_free', 'health_label']:
            if attribute in data_copy.columns:
                attr_by_knowledge = data_copy.groupby('knowledge_tertile')[attribute].mean()
                analysis[f'{attribute}_preference_by_knowledge'] = attr_by_knowledge.to_dict()
                
        return analysis
    
    def set_knowledge_coefficients(self, direct_coeff: float = 0.0,
                                 sugar_interaction_coeff: float = 0.0,
                                 label_interaction_coeff: float = 0.0,
                                 price_interaction_coeff: float = 0.0):
        """
        Set knowledge-related coefficients.
        
        Args:
            direct_coeff: Direct knowledge effect coefficient
            sugar_interaction_coeff: Knowledge × sugar interaction coefficient
            label_interaction_coeff: Knowledge × label interaction coefficient
            price_interaction_coeff: Knowledge × price interaction coefficient
        """
        self.knowledge_coefficients = {
            'direct': direct_coeff,
            'sugar_interaction': sugar_interaction_coeff,
            'label_interaction': label_interaction_coeff,
            'price_interaction': price_interaction_coeff
        }
        
    def get_integrator_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the nutrition knowledge integrator.
        
        Returns:
            Dictionary with integrator information
        """
        info = super().get_integrator_info()
        info.update({
            'knowledge_coefficients': self.knowledge_coefficients,
            'factor_file': str(NUTRITION_KNOWLEDGE_FILE)
        })
        return info
