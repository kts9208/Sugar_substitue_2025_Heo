"""
Results exporter for utility function calculations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class ResultsExporter:
    """
    Exports utility function results to various formats.
    
    Supports exporting to CSV, Excel, JSON, and generating reports.
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize results exporter.
        
        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = output_dir or Path("utility_function/outputs/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_csv(self, results: Dict[str, Any], 
                     filename: Optional[str] = None) -> List[Path]:
        """
        Export results to CSV files.
        
        Args:
            results: Results dictionary
            filename: Base filename (auto-generated if None)
            
        Returns:
            List of exported file paths
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"utility_results_{timestamp}"
            
        exported_files = []
        
        # Export total utility
        if 'total_utility' in results:
            utility_file = self.output_dir / f"{filename}_utility.csv"
            utility_df = pd.DataFrame({
                'observation_id': range(len(results['total_utility'])),
                'total_utility': results['total_utility']
            })
            utility_df.to_csv(utility_file, index=False)
            exported_files.append(utility_file)
            logger.info(f"Exported utility values to: {utility_file}")
            
        # Export utility decomposition
        if 'decomposition' in results and 'contributions' in results['decomposition']:
            decomp_file = self.output_dir / f"{filename}_decomposition.csv"
            results['decomposition']['contributions'].to_csv(decomp_file, index=False)
            exported_files.append(decomp_file)
            logger.info(f"Exported decomposition to: {decomp_file}")
            
        # Export input data
        if 'data' in results:
            data_file = self.output_dir / f"{filename}_data.csv"
            results['data'].to_csv(data_file, index=False)
            exported_files.append(data_file)
            logger.info(f"Exported input data to: {data_file}")
            
        return exported_files
    
    def export_to_excel(self, results: Dict[str, Any], 
                       filename: Optional[str] = None) -> Path:
        """
        Export results to Excel file with multiple sheets.
        
        Args:
            results: Results dictionary
            filename: Filename (auto-generated if None)
            
        Returns:
            Path to exported Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"utility_results_{timestamp}.xlsx"
        elif not filename.endswith('.xlsx'):
            filename += '.xlsx'
            
        excel_file = self.output_dir / filename
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Export total utility
            if 'total_utility' in results:
                utility_df = pd.DataFrame({
                    'observation_id': range(len(results['total_utility'])),
                    'total_utility': results['total_utility']
                })
                utility_df.to_excel(writer, sheet_name='Total_Utility', index=False)
                
            # Export utility decomposition
            if 'decomposition' in results and 'contributions' in results['decomposition']:
                results['decomposition']['contributions'].to_excel(
                    writer, sheet_name='Utility_Decomposition', index=False
                )
                
            # Export component importance
            if 'importance' in results:
                importance_df = pd.DataFrame.from_dict(results['importance'], orient='index')
                importance_df.to_excel(writer, sheet_name='Component_Importance')
                
            # Export input data (sample if too large)
            if 'data' in results:
                data = results['data']
                if len(data) > 10000:
                    data = data.sample(10000)  # Sample for Excel size limits
                data.to_excel(writer, sheet_name='Input_Data', index=False)
                
            # Export summary statistics
            if 'analysis' in results:
                analysis = results['analysis']
                if 'utility_distribution' in analysis:
                    dist_stats = analysis['utility_distribution']['descriptive_stats']
                    stats_df = pd.DataFrame.from_dict(dist_stats, orient='index', columns=['Value'])
                    stats_df.to_excel(writer, sheet_name='Summary_Statistics')
                    
        logger.info(f"Exported results to Excel: {excel_file}")
        return excel_file
    
    def export_to_json(self, results: Dict[str, Any], 
                      filename: Optional[str] = None) -> Path:
        """
        Export results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Filename (auto-generated if None)
            
        Returns:
            Path to exported JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"utility_results_{timestamp}.json"
        elif not filename.endswith('.json'):
            filename += '.json'
            
        json_file = self.output_dir / filename
        
        # Prepare results for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        logger.info(f"Exported results to JSON: {json_file}")
        return json_file
    
    def generate_summary_report(self, results: Dict[str, Any], 
                              analysis: Optional[Dict[str, Any]] = None,
                              filename: Optional[str] = None) -> Path:
        """
        Generate a comprehensive summary report.
        
        Args:
            results: Results dictionary
            analysis: Analysis results (optional)
            filename: Filename (auto-generated if None)
            
        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"utility_summary_report_{timestamp}.txt"
        elif not filename.endswith('.txt'):
            filename += '.txt'
            
        report_file = self.output_dir / filename
        
        with open(report_file, 'w') as f:
            f.write("UTILITY FUNCTION CALCULATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic information
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            if 'data' in results:
                data = results['data']
                f.write("DATA SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Number of observations: {len(data)}\n")
                f.write(f"Number of variables: {len(data.columns)}\n\n")
                
            # Utility summary
            if 'total_utility' in results:
                utility = results['total_utility']
                f.write("UTILITY SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Mean utility: {utility.mean():.4f}\n")
                f.write(f"Standard deviation: {utility.std():.4f}\n")
                f.write(f"Minimum utility: {utility.min():.4f}\n")
                f.write(f"Maximum utility: {utility.max():.4f}\n")
                f.write(f"Range: {utility.max() - utility.min():.4f}\n\n")
                
            # Component importance
            if 'importance' in results:
                f.write("COMPONENT IMPORTANCE\n")
                f.write("-" * 20 + "\n")
                for component, info in results['importance'].items():
                    if isinstance(info, dict) and 'relative_importance' in info:
                        f.write(f"{component}: {info['relative_importance']:.4f}\n")
                f.write("\n")
                
            # Analysis results
            if analysis:
                f.write("DETAILED ANALYSIS\n")
                f.write("-" * 20 + "\n")
                
                # Utility distribution
                if 'utility_distribution' in analysis:
                    dist = analysis['utility_distribution']
                    f.write("Utility Distribution:\n")
                    if 'descriptive_stats' in dist:
                        stats = dist['descriptive_stats']
                        f.write(f"  Skewness: {stats.get('skewness', 'N/A'):.4f}\n")
                        f.write(f"  Kurtosis: {stats.get('kurtosis', 'N/A'):.4f}\n")
                    if 'normality_test' in dist:
                        norm_test = dist['normality_test']
                        f.write(f"  Normal distribution: {norm_test.get('is_normal', 'N/A')}\n")
                    f.write("\n")
                    
                # Prediction accuracy
                if 'prediction_accuracy' in analysis:
                    pred = analysis['prediction_accuracy']
                    f.write("Prediction Accuracy:\n")
                    if 'accuracy_metrics' in pred:
                        acc = pred['accuracy_metrics']
                        f.write(f"  Overall accuracy: {acc.get('accuracy', 'N/A'):.4f}\n")
                    if 'auc' in pred:
                        f.write(f"  AUC: {pred['auc']:.4f}\n")
                    f.write("\n")
                    
                # Key findings
                if 'summary' in analysis and 'key_findings' in analysis['summary']:
                    f.write("Key Findings:\n")
                    for finding in analysis['summary']['key_findings']:
                        f.write(f"  - {finding}\n")
                    f.write("\n")
                    
        logger.info(f"Generated summary report: {report_file}")
        return report_file
    
    def export_for_publication(self, results: Dict[str, Any], 
                             analysis: Optional[Dict[str, Any]] = None,
                             format_type: str = 'academic') -> Dict[str, Path]:
        """
        Export results in publication-ready format.
        
        Args:
            results: Results dictionary
            analysis: Analysis results
            format_type: Type of publication format ('academic', 'business', 'technical')
            
        Returns:
            Dictionary with exported file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"utility_publication_{format_type}_{timestamp}"
        
        exported_files = {}
        
        # Export main results table
        if 'decomposition' in results and 'contributions' in results['decomposition']:
            contributions = results['decomposition']['contributions']
            
            # Create publication table
            pub_table = self._create_publication_table(contributions, format_type)
            table_file = self.output_dir / f"{base_filename}_table.csv"
            pub_table.to_csv(table_file, index=False)
            exported_files['main_table'] = table_file
            
        # Export summary statistics
        if 'total_utility' in results:
            summary_stats = self._create_summary_stats_table(results['total_utility'])
            stats_file = self.output_dir / f"{base_filename}_summary_stats.csv"
            summary_stats.to_csv(stats_file, index=False)
            exported_files['summary_stats'] = stats_file
            
        # Export analysis results if available
        if analysis:
            analysis_file = self.output_dir / f"{base_filename}_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            exported_files['analysis'] = analysis_file
            
        logger.info(f"Exported publication-ready files for {format_type} format")
        return exported_files
    
    def _prepare_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare results for JSON serialization.
        
        Args:
            results: Results dictionary
            
        Returns:
            JSON-serializable dictionary
        """
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, pd.Series):
                json_results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
                
        return json_results
    
    def _create_publication_table(self, contributions: pd.DataFrame, 
                                format_type: str) -> pd.DataFrame:
        """
        Create publication-ready table.
        
        Args:
            contributions: Utility contributions DataFrame
            format_type: Publication format type
            
        Returns:
            Publication-ready DataFrame
        """
        # Calculate summary statistics for each component
        component_columns = [col for col in contributions.columns if col != 'total_utility']
        
        pub_data = []
        for component in component_columns:
            if component in contributions.columns:
                values = contributions[component]
                pub_data.append({
                    'Component': component.replace('_', ' ').title(),
                    'Mean': f"{values.mean():.4f}",
                    'Std Dev': f"{values.std():.4f}",
                    'Min': f"{values.min():.4f}",
                    'Max': f"{values.max():.4f}"
                })
                
        return pd.DataFrame(pub_data)
    
    def _create_summary_stats_table(self, utility_values: pd.Series) -> pd.DataFrame:
        """
        Create summary statistics table.
        
        Args:
            utility_values: Utility values
            
        Returns:
            Summary statistics DataFrame
        """
        stats_data = [
            {'Statistic': 'Count', 'Value': f"{len(utility_values)}"},
            {'Statistic': 'Mean', 'Value': f"{utility_values.mean():.4f}"},
            {'Statistic': 'Standard Deviation', 'Value': f"{utility_values.std():.4f}"},
            {'Statistic': 'Minimum', 'Value': f"{utility_values.min():.4f}"},
            {'Statistic': 'Maximum', 'Value': f"{utility_values.max():.4f}"},
            {'Statistic': '25th Percentile', 'Value': f"{utility_values.quantile(0.25):.4f}"},
            {'Statistic': '50th Percentile', 'Value': f"{utility_values.quantile(0.50):.4f}"},
            {'Statistic': '75th Percentile', 'Value': f"{utility_values.quantile(0.75):.4f}"}
        ]
        
        return pd.DataFrame(stats_data)
