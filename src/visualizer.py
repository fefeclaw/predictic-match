"""
Visualizer Module - Predictic Match
Professional data visualization for model evaluation and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Professional style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color palettes
COLOR_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}

MODEL_COLORS = {
    'XGBoost': '#E74C3C',
    'Random Forest': '#3498DB',
    'Logistic Regression': '#2ECC71',
    'LightGBM': '#9B59B6',
    'CatBoost': '#F39C12',
    'Neural Network': '#1ABC9C',
    'Ensemble': '#34495E'
}


def setup_plot(figsize: Tuple[int, int] = (12, 8), dpi: int = 150) -> plt.Figure:
    """Initialize a figure with professional styling."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FFFFFF')
    return fig


def save_plot(fig: plt.Figure, filename: str, output_dir: Optional[str] = None, 
              dpi: int = 300, bbox_inches: str = 'tight') -> str:
    """Save figure as high-quality PNG."""
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
    else:
        filepath = Path(filename)
    
    if not filepath.suffix.lower() == '.png':
        filepath = filepath.with_suffix('.png')
    
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                facecolor=fig.patch.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return str(filepath)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'log_loss'],
    title: str = 'Model Performance Comparison',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    show: bool = False
) -> str:
    """
    Create a bar chart comparing model performance across multiple metrics.
    
    Parameters
    ----------
    results : dict
        Dictionary with model names as keys and metric dictionaries as values.
        Example: {'XGBoost': {'accuracy': 0.85, 'log_loss': 0.42}, ...}
    metrics : list
        List of metrics to display.
    title : str
        Plot title.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size (width, height).
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    fig = setup_plot(figsize, dpi=150)
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    ncols = min(n_metrics, 3)
    nrows = (n_metrics + ncols - 1) // ncols
    
    if n_metrics == 1:
        axes = [fig.add_subplot(111)]
    else:
        axes = fig.subplots(nrows, ncols)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    models = list(results.keys())
    x_positions = np.arange(len(models))
    bar_width = 0.8 / max(len(models), 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx] if idx < len(axes) else fig.add_subplot(nrows, ncols, idx + 1)
        
        values = [results[model].get(metric, 0) for model in models]
        colors = [MODEL_COLORS.get(model, COLOR_PALETTE['primary']) for model in models]
        
        bars = ax.bar(x_positions, values, width=bar_width, color=colors, 
                      edgecolor='white', linewidth=1.5, alpha=0.9)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == 'log_loss':
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:
                ax.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} by Model', fontsize=12, fontweight='bold')
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis limits
        if metric == 'log_loss':
            ax.set_ylim(0, max(values) * 1.3)
        else:
            ax.set_ylim(0, min(1.0, max(values) * 1.15))
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = 'Confusion Matrix',
    normalize: bool = True,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = False
) -> str:
    """
    Create a heatmap confusion matrix with percentages.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels.
    title : str
        Plot title.
    normalize : bool
        Whether to display percentages.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    from sklearn.metrics import confusion_matrix
    
    fig = setup_plot(figsize, dpi=150)
    ax = fig.add_subplot(111)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]
    
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.1%' if normalize else 'd',
                cmap='YlOrRd', cbar=True, cbar_kws={'label': 'Percentage' if normalize else 'Count'},
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, linecolor='white', square=True)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 15,
    title: str = 'Feature Importance (Top 15)',
    model_name: str = 'Model',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = False
) -> str:
    """
    Create a horizontal bar chart of top feature importances.
    
    Parameters
    ----------
    model : object
        Trained model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to display.
    title : str
        Plot title.
    model_name : str
        Name of the model for display.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    fig = setup_plot(figsize, dpi=150)
    ax = fig.add_subplot(111)
    
    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance_'):  # LightGBM
        importances = model.feature_importance_
    else:
        raise ValueError("Model does not have feature importances")
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True).tail(top_n)
    
    # Color gradient based on importance
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
    
    # Create horizontal bar chart
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, importance_df['importance']):
        width = bar.get_width()
        ax.annotate(f'{val:.3f}', xy=(width, bar.get_y() + bar.get_height()/2),
                   ha='left', va='center', fontsize=9, fontweight='bold',
                   xytext=(5, 0), textcoords='offset points')
    
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\n{model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_axisbelow(True)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Invert y-axis to have highest importance at top
    ax.invert_yaxis()
    
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_names: Optional[List[str]] = None,
    title: str = 'Calibration Curve',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = False
) -> str:
    """
    Create calibration curves for probability predictions.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like or list of arrays
        Predicted probabilities. Can be a single array or list for multiple models.
    model_names : list, optional
        Names of models for legend.
    title : str
        Plot title.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    from sklearn.calibration import calibration_curve
    
    fig = setup_plot(figsize, dpi=150)
    ax = fig.add_subplot(111)
    
    # Handle single or multiple models
    if isinstance(y_prob, (list, tuple)) and len(y_prob) > 0 and isinstance(y_prob[0], (list, np.ndarray)):
        prob_list = y_prob
        names = model_names or [f'Model {i+1}' for i in range(len(prob_list))]
    else:
        prob_list = [y_prob]
        names = model_names or ['Model']
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2, alpha=0.7)
    
    # Plot calibration curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(prob_list)))
    
    for idx, (y_prob_single, name) in enumerate(zip(prob_list, names)):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob_single, n_bins=10, strategy='uniform'
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=name, color=colors[idx], linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


def plot_probability_divergence(
    bookmaker_probs: np.ndarray,
    polymarket_probs: np.ndarray,
    match_ids: Optional[np.ndarray] = None,
    title: str = 'Bookmaker vs Polymarket Probability Divergence',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = False
) -> str:
    """
    Create a scatter plot comparing bookmaker and Polymarket probabilities.
    
    Parameters
    ----------
    bookmaker_probs : array-like
        Probabilities from bookmakers.
    polymarket_probs : array-like
        Probabilities from Polymarket.
    match_ids : array-like, optional
        Match identifiers for annotation.
    title : str
        Plot title.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    fig = setup_plot(figsize, dpi=150)
    ax = fig.add_subplot(111)
    
    bookmaker_probs = np.asarray(bookmaker_probs)
    polymarket_probs = np.asarray(polymarket_probs)
    
    # Calculate divergence
    divergence = np.abs(bookmaker_probs - polymarket_probs)
    
    # Create scatter plot with color based on divergence
    scatter = ax.scatter(bookmaker_probs, polymarket_probs, 
                        c=divergence, cmap='RdYlGn_r', 
                        s=100, alpha=0.7, edgecolors='white', linewidth=0.5,
                        vmin=0, vmax=0.5)
    
    # Add diagonal line (perfect agreement)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect agreement')
    
    # Add threshold lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Divergence', fontsize=10, fontweight='bold')
    
    # Annotate points with high divergence
    high_divergence_mask = divergence > 0.15
    if match_ids is not None and np.any(high_divergence_mask):
        for i in np.where(high_divergence_mask)[0]:
            ax.annotate(f'{match_ids[i]}', 
                       xy=(bookmaker_probs[i], polymarket_probs[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                alpha=0.7, edgecolor='black'))
    
    ax.set_xlabel('Bookmaker Probability', fontsize=11, fontweight='bold')
    ax.set_ylabel('Polymarket Probability', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    # Add statistics text box
    stats_text = (f'Mean Divergence: {np.mean(divergence):.3f}\n'
                  f'Max Divergence: {np.max(divergence):.3f}\n'
                  f'Correlation: {np.corrcoef(bookmaker_probs, polymarket_probs)[0, 1]:.3f}')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


def plot_triple_layer_radar(
    data: Dict[str, Dict[str, float]],
    categories: List[str],
    title: str = 'Triple Layer Radar Comparison',
    source_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show: bool = False
) -> str:
    """
    Create a radar chart comparing three data sources across multiple categories.
    
    Parameters
    ----------
    data : dict
        Dictionary with source names as keys and category-value dicts as values.
        Example: {'Bookmaker': {'Home': 0.6, 'Draw': 0.2, 'Away': 0.2}, ...}
    categories : list
        List of category names.
    title : str
        Plot title.
    source_names : list, optional
        Names of sources for legend.
    output_path : str, optional
        Path to save the plot.
    figsize : tuple
        Figure size.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    str
        Path to saved file.
    """
    fig = setup_plot(figsize, dpi=150)
    
    # Number of variables
    N = len(categories)
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create polar subplot
    ax = fig.add_subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=11, weight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], 
               ['0.2', '0.4', '0.6', '0.8', '1.0'], 
               color="grey", size=9)
    plt.ylim(0, 1)
    
    # Colors for sources
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['tertiary']]
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Plot data for each source
    source_keys = list(data.keys())[:3]  # Limit to 3 sources
    names = source_names or source_keys
    
    for idx, (source_key, name) in enumerate(zip(source_keys, names)):
        values = [data[source_key].get(cat, 0) for cat in categories]
        values += values[:1]  # Close the loop
        
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        ax.plot(angles, values, linewidth=2.5, linestyle=linestyle, 
                label=name, color=color, marker=marker, markersize=8, alpha=0.8)
        ax.fill(angles, values, color=color, alpha=0.15)
    
    # Add title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, y=1.1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
    
    if show:
        plt.show()
    
    return output_path or ''


# Convenience function to create all visualizations at once
def create_all_visualizations(
    results: Dict,
    output_dir: str = './visualizations',
    prefix: str = ''
) -> Dict[str, str]:
    """
    Generate all visualization types and save them.
    
    Parameters
    ----------
    results : dict
        Dictionary containing all necessary data for visualizations.
    output_dir : str
        Directory to save plots.
    prefix : str
        Prefix for output filenames.
    
    Returns
    -------
    dict
        Dictionary mapping plot type to saved file path.
    """
    output_paths = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Model Comparison
    if 'model_results' in results:
        output_paths['model_comparison'] = plot_model_comparison(
            results['model_results'],
            title='Predictic Match - Model Performance',
            output_path=f'{output_dir}/{prefix}model_comparison.png'
        )
    
    # 2. Confusion Matrix
    if 'y_true' in results and 'y_pred' in results:
        output_paths['confusion_matrix'] = plot_confusion_matrix(
            results['y_true'],
            results['y_pred'],
            title='Predictic Match - Confusion Matrix',
            output_path=f'{output_dir}/{prefix}confusion_matrix.png'
        )
    
    # 3. Feature Importance
    if 'model' in results and 'feature_names' in results:
        output_paths['feature_importance'] = plot_feature_importance(
            results['model'],
            results['feature_names'],
            title='Predictic Match - Feature Importance',
            output_path=f'{output_dir}/{prefix}feature_importance.png'
        )
    
    # 4. Calibration Curve
    if 'y_true' in results and 'y_prob' in results:
        output_paths['calibration_curve'] = plot_calibration_curve(
            results['y_true'],
            results['y_prob'],
            title='Predictic Match - Calibration Curve',
            output_path=f'{output_dir}/{prefix}calibration_curve.png'
        )
    
    # 5. Probability Divergence
    if 'bookmaker_probs' in results and 'polymarket_probs' in results:
        output_paths['probability_divergence'] = plot_probability_divergence(
            results['bookmaker_probs'],
            results['polymarket_probs'],
            match_ids=results.get('match_ids'),
            title='Predictic Match - Probability Divergence',
            output_path=f'{output_dir}/{prefix}probability_divergence.png'
        )
    
    # 6. Triple Layer Radar
    if 'radar_data' in results and 'radar_categories' in results:
        output_paths['triple_layer_radar'] = plot_triple_layer_radar(
            results['radar_data'],
            results['radar_categories'],
            title='Predictic Match - Source Comparison',
            output_path=f'{output_dir}/{prefix}triple_layer_radar.png'
        )
    
    return output_paths


if __name__ == '__main__':
    # Demo/test code
    print("Predictic Match Visualizer Module")
    print("==================================")
    print("\nAvailable functions:")
    print("  - plot_model_comparison()")
    print("  - plot_confusion_matrix()")
    print("  - plot_feature_importance()")
    print("  - plot_calibration_curve()")
    print("  - plot_probability_divergence()")
    print("  - plot_triple_layer_radar()")
    print("  - create_all_visualizations()")
    print("\nImport and use in your prediction pipeline.")
