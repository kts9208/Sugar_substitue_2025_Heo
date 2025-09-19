"""
Results manager for utility function calculations.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging

from ..config.settings import OUTPUT_DIR, RESULTS_TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


class ResultsManager:
    """
    Manages storage and retrieval of utility function calculation results.
    
    Handles saving results in multiple formats and organizing output files.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize results manager.
        
        Args:
            output_dir: Directory for saving results (defaults to config setting)
        """
        self.output_dir = output_dir or OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.analysis_dir = self.output_dir / "analysis"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.data_dir, self.analysis_dir, self.visualizations_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.results_registry = {}
        
    def save_utility_results(self, results: Dict[str, Any], 
                            experiment_name: str = "utility_calculation",
                            include_timestamp: bool = True) -> str:
        """
        Save utility calculation results.
        
        Args:
            results: Dictionary containing utility results
            experiment_name: Name for this experiment
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            Experiment ID for referencing saved results
        """
        timestamp = datetime.now().strftime(RESULTS_TIMESTAMP_FORMAT)
        experiment_id = f"{experiment_name}_{timestamp}" if include_timestamp else experiment_name
        
        logger.info(f"Saving utility results for experiment: {experiment_id}")
        
        # Create experiment directory
        experiment_dir = self.data_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Save different components of results
        saved_files = {}
        
        # Save total utility
        if 'total_utility' in results:
            utility_file = experiment_dir / "total_utility.csv"
            utility_df = pd.DataFrame({
                'observation_id': range(len(results['total_utility'])),
                'total_utility': results['total_utility']
            })
            utility_df.to_csv(utility_file, index=False)
            saved_files['total_utility'] = str(utility_file)
            
        # Save utility decomposition if available
        if 'decomposition' in results:
            decomp_file = experiment_dir / "utility_decomposition.csv"
            decomposition = results['decomposition']
            if 'contributions' in decomposition:
                decomposition['contributions'].to_csv(decomp_file, index=False)
                saved_files['decomposition'] = str(decomp_file)
                
        # Save component importance
        if 'importance' in results:
            importance_file = experiment_dir / "component_importance.json"
            with open(importance_file, 'w') as f:
                json.dump(results['importance'], f, indent=2, default=str)
            saved_files['importance'] = str(importance_file)
            
        # Save input data
        if 'data' in results:
            data_file = experiment_dir / "input_data.csv"
            results['data'].to_csv(data_file, index=False)
            saved_files['input_data'] = str(data_file)
            
        # Save metadata
        metadata = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'saved_files': saved_files,
            'parameters': results.get('parameters', {}),
            'summary_stats': self._extract_summary_stats(results)
        }
        
        metadata_file = experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        # Save complete results as pickle for easy loading
        pickle_file = experiment_dir / "complete_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        saved_files['complete_results'] = str(pickle_file)
        
        # Update registry
        self.results_registry[experiment_id] = {
            'experiment_dir': str(experiment_dir),
            'metadata': metadata,
            'saved_files': saved_files
        }
        
        logger.info(f"Results saved successfully for experiment: {experiment_id}")
        return experiment_id
    
    def load_utility_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Load utility calculation results.
        
        Args:
            experiment_id: ID of experiment to load
            
        Returns:
            Dictionary containing loaded results
        """
        if experiment_id not in self.results_registry:
            # Try to find experiment directory
            experiment_dir = self.data_dir / experiment_id
            if not experiment_dir.exists():
                raise ValueError(f"Experiment '{experiment_id}' not found")
            self._load_experiment_registry(experiment_id, experiment_dir)
            
        logger.info(f"Loading utility results for experiment: {experiment_id}")
        
        experiment_info = self.results_registry[experiment_id]
        experiment_dir = Path(experiment_info['experiment_dir'])
        
        # Load complete results from pickle if available
        pickle_file = experiment_dir / "complete_results.pkl"
        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                results = pickle.load(f)
            logger.info("Loaded complete results from pickle file")
            return results
            
        # Otherwise, load individual components
        results = {}
        
        # Load total utility
        utility_file = experiment_dir / "total_utility.csv"
        if utility_file.exists():
            utility_df = pd.read_csv(utility_file)
            results['total_utility'] = utility_df['total_utility']
            
        # Load decomposition
        decomp_file = experiment_dir / "utility_decomposition.csv"
        if decomp_file.exists():
            results['decomposition'] = {'contributions': pd.read_csv(decomp_file)}
            
        # Load importance
        importance_file = experiment_dir / "component_importance.json"
        if importance_file.exists():
            with open(importance_file, 'r') as f:
                results['importance'] = json.load(f)
                
        # Load input data
        data_file = experiment_dir / "input_data.csv"
        if data_file.exists():
            results['data'] = pd.read_csv(data_file)
            
        # Load metadata
        metadata_file = experiment_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                results['metadata'] = json.load(f)
                
        logger.info("Results loaded successfully")
        return results
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all saved experiments.
        
        Returns:
            List of experiment information
        """
        experiments = []
        
        # Scan data directory for experiments
        for experiment_dir in self.data_dir.iterdir():
            if experiment_dir.is_dir():
                experiment_id = experiment_dir.name
                
                # Load metadata if available
                metadata_file = experiment_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    experiments.append(metadata)
                else:
                    # Basic info if no metadata
                    experiments.append({
                        'experiment_id': experiment_id,
                        'experiment_dir': str(experiment_dir),
                        'timestamp': experiment_dir.stat().st_mtime
                    })
                    
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment and all its files.
        
        Args:
            experiment_id: ID of experiment to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            experiment_dir = self.data_dir / experiment_id
            if experiment_dir.exists():
                import shutil
                shutil.rmtree(experiment_dir)
                
                # Remove from registry
                if experiment_id in self.results_registry:
                    del self.results_registry[experiment_id]
                    
                logger.info(f"Experiment '{experiment_id}' deleted successfully")
                return True
            else:
                logger.warning(f"Experiment '{experiment_id}' not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting experiment '{experiment_id}': {str(e)}")
            return False
    
    def _extract_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract summary statistics from results.
        
        Args:
            results: Results dictionary
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        if 'total_utility' in results:
            utility = results['total_utility']
            summary['utility_stats'] = {
                'n_observations': len(utility),
                'mean': float(utility.mean()),
                'std': float(utility.std()),
                'min': float(utility.min()),
                'max': float(utility.max())
            }
            
        if 'data' in results:
            summary['data_stats'] = {
                'n_observations': len(results['data']),
                'n_variables': len(results['data'].columns)
            }
            
        return summary
    
    def _load_experiment_registry(self, experiment_id: str, experiment_dir: Path):
        """
        Load experiment into registry.
        
        Args:
            experiment_id: Experiment ID
            experiment_dir: Path to experiment directory
        """
        metadata_file = experiment_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'experiment_id': experiment_id}
            
        # Find saved files
        saved_files = {}
        for file_path in experiment_dir.iterdir():
            if file_path.is_file():
                saved_files[file_path.stem] = str(file_path)
                
        self.results_registry[experiment_id] = {
            'experiment_dir': str(experiment_dir),
            'metadata': metadata,
            'saved_files': saved_files
        }
    
    def export_experiment_summary(self, experiment_id: str, 
                                 output_file: Optional[Path] = None) -> Path:
        """
        Export experiment summary to file.
        
        Args:
            experiment_id: ID of experiment to export
            output_file: Output file path (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if experiment_id not in self.results_registry:
            experiment_dir = self.data_dir / experiment_id
            if experiment_dir.exists():
                self._load_experiment_registry(experiment_id, experiment_dir)
            else:
                raise ValueError(f"Experiment '{experiment_id}' not found")
                
        if output_file is None:
            output_file = self.reports_dir / f"{experiment_id}_summary.json"
            
        experiment_info = self.results_registry[experiment_id]
        
        # Create comprehensive summary
        summary = {
            'experiment_info': experiment_info['metadata'],
            'files': experiment_info['saved_files'],
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Add detailed statistics if available
        try:
            results = self.load_utility_results(experiment_id)
            if 'total_utility' in results:
                utility = results['total_utility']
                summary['detailed_stats'] = {
                    'utility_distribution': {
                        'percentiles': {
                            '5th': float(utility.quantile(0.05)),
                            '25th': float(utility.quantile(0.25)),
                            '50th': float(utility.quantile(0.50)),
                            '75th': float(utility.quantile(0.75)),
                            '95th': float(utility.quantile(0.95))
                        }
                    }
                }
        except Exception as e:
            logger.warning(f"Could not load detailed stats: {str(e)}")
            
        # Save summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Experiment summary exported to: {output_file}")
        return output_file
    
    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get information about the results manager.
        
        Returns:
            Dictionary with manager information
        """
        return {
            'output_dir': str(self.output_dir),
            'subdirectories': {
                'data': str(self.data_dir),
                'analysis': str(self.analysis_dir),
                'visualizations': str(self.visualizations_dir),
                'reports': str(self.reports_dir)
            },
            'n_experiments': len(self.results_registry),
            'experiments': list(self.results_registry.keys())
        }
