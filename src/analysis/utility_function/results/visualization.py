"""
Visualization module for utility function results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


class UtilityVisualizer:
    """
    Creates visualizations for utility function results.
    
    Generates plots for utility distributions, component effects,
    and model performance metrics.
    """
    
    def __init__(self, output_dir: Optional[Path] = None, 
                 figure_size: Tuple[int, int] = (10, 6),
                 dpi: int = 300):
        """
        Initialize utility visualizer.
        
        Args:
            output_dir: Directory for saving plots
            figure_size: Default figure size
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir or Path("utility_function/outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size
        self.dpi = dpi
        
    def plot_utility_distribution(self, utility_values: pd.Series, 
                                 title: str = "Utility Distribution",
                                 save_path: Optional[Path] = None) -> Path:
        """
        Plot utility value distribution.
        
        Args:
            utility_values: Series with utility values
            title: Plot title
            save_path: Path to save plot (auto-generated if None)
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Histogram
        axes[0, 0].hist(utility_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel('Utility Value')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(utility_values, vert=True)
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel('Utility Value')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(utility_values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal)')
        
        # Density plot
        utility_values.plot.density(ax=axes[1, 1], color='orange')
        axes[1, 1].set_title('Density Plot')
        axes[1, 1].set_xlabel('Utility Value')
        axes[1, 1].set_ylabel('Density')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "utility_distribution.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Utility distribution plot saved to: {save_path}")
        return save_path
    
    def plot_component_contributions(self, contributions: pd.DataFrame,
                                   title: str = "Component Contributions",
                                   save_path: Optional[Path] = None) -> Path:
        """
        Plot utility component contributions.
        
        Args:
            contributions: DataFrame with component contributions
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        # Exclude total utility column
        component_columns = [col for col in contributions.columns if col != 'total_utility']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Stacked bar plot of mean contributions
        mean_contributions = contributions[component_columns].mean()
        axes[0, 0].bar(range(len(mean_contributions)), mean_contributions.values, 
                      color=sns.color_palette("husl", len(mean_contributions)))
        axes[0, 0].set_title('Mean Component Contributions')
        axes[0, 0].set_xlabel('Component')
        axes[0, 0].set_ylabel('Mean Contribution')
        axes[0, 0].set_xticks(range(len(mean_contributions)))
        axes[0, 0].set_xticklabels([col.replace('_', ' ').title() for col in mean_contributions.index], 
                                  rotation=45, ha='right')
        
        # Box plot of component contributions
        contributions[component_columns].boxplot(ax=axes[0, 1])
        axes[0, 1].set_title('Component Contribution Distributions')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].set_ylabel('Contribution Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        if len(component_columns) > 1:
            corr_matrix = contributions[component_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Component Correlations')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient components\nfor correlation analysis', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Component Correlations')
        
        # Cumulative contribution plot
        if 'total_utility' in contributions.columns:
            sorted_indices = contributions['total_utility'].argsort()
            cumulative_data = contributions.iloc[sorted_indices][component_columns].cumsum(axis=1)
            
            for i, col in enumerate(component_columns):
                if i == 0:
                    axes[1, 1].fill_between(range(len(cumulative_data)), 0, cumulative_data[col], 
                                          alpha=0.7, label=col.replace('_', ' ').title())
                else:
                    axes[1, 1].fill_between(range(len(cumulative_data)), 
                                          cumulative_data[component_columns[i-1]], 
                                          cumulative_data[col], 
                                          alpha=0.7, label=col.replace('_', ' ').title())
            
            axes[1, 1].set_title('Cumulative Contributions (Sorted by Total Utility)')
            axes[1, 1].set_xlabel('Observation (Sorted)')
            axes[1, 1].set_ylabel('Cumulative Utility')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "component_contributions.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Component contributions plot saved to: {save_path}")
        return save_path
    
    def plot_attribute_effects(self, data: pd.DataFrame, utility_values: pd.Series,
                             title: str = "DCE Attribute Effects",
                             save_path: Optional[Path] = None) -> Path:
        """
        Plot effects of DCE attributes on utility.
        
        Args:
            data: Input data with DCE attributes
            utility_values: Utility values
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Sugar effect
        if 'sugar_free' in data.columns:
            sugar_data = pd.DataFrame({
                'sugar_free': data['sugar_free'],
                'utility': utility_values
            })
            sns.boxplot(data=sugar_data, x='sugar_free', y='utility', ax=axes[0, 0])
            axes[0, 0].set_title('Sugar Effect on Utility')
            axes[0, 0].set_xlabel('Sugar Free (0=No, 1=Yes)')
            axes[0, 0].set_ylabel('Utility')
        
        # Health label effect
        if 'health_label' in data.columns:
            label_data = pd.DataFrame({
                'health_label': data['health_label'],
                'utility': utility_values
            })
            sns.boxplot(data=label_data, x='health_label', y='utility', ax=axes[0, 1])
            axes[0, 1].set_title('Health Label Effect on Utility')
            axes[0, 1].set_xlabel('Health Label (0=No, 1=Yes)')
            axes[0, 1].set_ylabel('Utility')
        
        # Price effect
        price_columns = ['price', 'price_normalized', 'chosen_price']
        price_col = None
        for col in price_columns:
            if col in data.columns:
                price_col = col
                break
                
        if price_col:
            axes[1, 0].scatter(data[price_col], utility_values, alpha=0.6)
            axes[1, 0].set_title(f'{price_col.replace("_", " ").title()} Effect on Utility')
            axes[1, 0].set_xlabel(price_col.replace('_', ' ').title())
            axes[1, 0].set_ylabel('Utility')
            
            # Add trend line
            z = np.polyfit(data[price_col], utility_values, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(data[price_col], p(data[price_col]), "r--", alpha=0.8)
        
        # Interaction effects (if available)
        if 'sugar_free' in data.columns and 'health_label' in data.columns:
            interaction_data = pd.DataFrame({
                'sugar_free': data['sugar_free'],
                'health_label': data['health_label'],
                'utility': utility_values
            })
            
            # Create interaction categories
            interaction_data['category'] = (
                interaction_data['sugar_free'].astype(str) + '_' + 
                interaction_data['health_label'].astype(str)
            )
            
            category_labels = {
                '0_0': 'Sugar, No Label',
                '0_1': 'Sugar, Label',
                '1_0': 'Sugar-free, No Label',
                '1_1': 'Sugar-free, Label'
            }
            
            interaction_data['category_label'] = interaction_data['category'].map(category_labels)
            
            sns.boxplot(data=interaction_data, x='category_label', y='utility', ax=axes[1, 1])
            axes[1, 1].set_title('Sugar × Health Label Interaction')
            axes[1, 1].set_xlabel('Product Category')
            axes[1, 1].set_ylabel('Utility')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "attribute_effects.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attribute effects plot saved to: {save_path}")
        return save_path
    
    def plot_model_performance(self, utility_values: pd.Series, 
                             actual_choices: pd.Series,
                             title: str = "Model Performance",
                             save_path: Optional[Path] = None) -> Path:
        """
        Plot model performance metrics.
        
        Args:
            utility_values: Predicted utility values
            actual_choices: Actual choice outcomes
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Path to saved plot
        """
        # Convert utilities to probabilities
        probabilities = 1 / (1 + np.exp(-utility_values))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Utility vs actual choices
        choice_data = pd.DataFrame({
            'utility': utility_values,
            'choice': actual_choices
        })
        sns.boxplot(data=choice_data, x='choice', y='utility', ax=axes[0, 0])
        axes[0, 0].set_title('Utility by Actual Choice')
        axes[0, 0].set_xlabel('Actual Choice (0=Not Chosen, 1=Chosen)')
        axes[0, 0].set_ylabel('Utility')
        
        # Probability distribution by choice
        prob_data = pd.DataFrame({
            'probability': probabilities,
            'choice': actual_choices
        })
        sns.histplot(data=prob_data, x='probability', hue='choice', 
                    alpha=0.7, ax=axes[0, 1])
        axes[0, 1].set_title('Probability Distribution by Choice')
        axes[0, 1].set_xlabel('Predicted Probability')
        axes[0, 1].set_ylabel('Count')
        
        # ROC Curve
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(actual_choices, probabilities)
            roc_auc = auc(fpr, tpr)
            
            axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                           label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC Curve')
            axes[1, 0].legend(loc="lower right")
            
        except ImportError:
            axes[1, 0].text(0.5, 0.5, 'sklearn not available\nfor ROC curve', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('ROC Curve')
        
        # Calibration plot
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = actual_choices[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        axes[1, 1].plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        axes[1, 1].plot(bin_centers, bin_accuracies, 's-', label='Model')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title('Calibration Plot')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "model_performance.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model performance plot saved to: {save_path}")
        return save_path
    
    def create_comprehensive_dashboard(self, results: Dict[str, Any],
                                     analysis: Optional[Dict[str, Any]] = None,
                                     save_path: Optional[Path] = None) -> Path:
        """
        Create comprehensive visualization dashboard.
        
        Args:
            results: Complete utility calculation results
            analysis: Analysis results
            save_path: Path to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        saved_plots = []
        
        # Generate individual plots
        if 'total_utility' in results:
            plot_path = self.plot_utility_distribution(
                results['total_utility'], 
                save_path=self.output_dir / "dashboard_utility_dist.png"
            )
            saved_plots.append(plot_path)
            
        if 'decomposition' in results and 'contributions' in results['decomposition']:
            plot_path = self.plot_component_contributions(
                results['decomposition']['contributions'],
                save_path=self.output_dir / "dashboard_components.png"
            )
            saved_plots.append(plot_path)
            
        if 'total_utility' in results and 'data' in results:
            plot_path = self.plot_attribute_effects(
                results['data'], results['total_utility'],
                save_path=self.output_dir / "dashboard_attributes.png"
            )
            saved_plots.append(plot_path)
            
            # Model performance if choice data available
            if 'choice_value' in results['data'].columns:
                plot_path = self.plot_model_performance(
                    results['total_utility'], results['data']['choice_value'],
                    save_path=self.output_dir / "dashboard_performance.png"
                )
                saved_plots.append(plot_path)
        
        if save_path is None:
            save_path = self.output_dir / "utility_dashboard.html"
            
        # Create HTML dashboard
        self._create_html_dashboard(saved_plots, save_path, results, analysis)
        
        logger.info(f"Comprehensive dashboard created: {save_path}")
        return save_path
    
    def _create_html_dashboard(self, plot_paths: List[Path], 
                             save_path: Path,
                             results: Dict[str, Any],
                             analysis: Optional[Dict[str, Any]] = None):
        """
        Create HTML dashboard with embedded plots.
        
        Args:
            plot_paths: List of plot file paths
            save_path: Path to save HTML file
            results: Results dictionary
            analysis: Analysis results
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Utility Function Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .plot-container { margin: 20px 0; text-align: center; }
                .plot-container img { max-width: 100%; height: auto; }
                .summary { background-color: #f5f5f5; padding: 15px; margin: 20px 0; }
                .stats-table { border-collapse: collapse; width: 100%; }
                .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .stats-table th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Utility Function Analysis Dashboard</h1>
                <p>Generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        """
        
        # Add summary statistics
        if 'total_utility' in results:
            utility = results['total_utility']
            html_content += f"""
            <div class="summary">
                <h2>Summary Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Number of Observations</td><td>{len(utility)}</td></tr>
                    <tr><td>Mean Utility</td><td>{utility.mean():.4f}</td></tr>
                    <tr><td>Standard Deviation</td><td>{utility.std():.4f}</td></tr>
                    <tr><td>Minimum Utility</td><td>{utility.min():.4f}</td></tr>
                    <tr><td>Maximum Utility</td><td>{utility.max():.4f}</td></tr>
                </table>
            </div>
            """
        
        # Add plots
        for plot_path in plot_paths:
            plot_name = plot_path.stem.replace('dashboard_', '').replace('_', ' ').title()
            html_content += f"""
            <div class="plot-container">
                <h2>{plot_name}</h2>
                <img src="{plot_path.name}" alt="{plot_name}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
