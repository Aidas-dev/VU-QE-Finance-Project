import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns


def plot_portfolio_allocation(weights, title, filename):
    """
    Create a professional pie chart of portfolio allocations
    
    Parameters:
    weights (Series): Portfolio weights/allocations (negative values indicate short positions)
    title (str): Chart title
    filename (str): Output filename
    """
    # Set professional font settings
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # Filter and sort weights
    significant_weights = weights[abs(weights) > 0.01]

    # Sort by absolute weight descending
    significant_weights = significant_weights.iloc[abs(significant_weights).argsort()[::-1]]
    labels = significant_weights.index

    # Create figure with 16:10 aspect ratio
    fig = plt.figure(figsize=(16, 10), dpi=500)
    ax1 = fig.add_subplot(111)  # Main plot area

    # Create pie chart without labels or percentages
    patches = ax1.pie(
        abs(significant_weights),
        labels=None,
        startangle=90,
        colors=sns.color_palette("mako", len(significant_weights)),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )[0]

    # Create detailed legend with allocation info
    legend_labels = [
        f"{label} ({'Long' if significant_weights[label] > 0 else 'Short'}): {abs(significant_weights[label]):.1%}"
        for label in labels
    ]
    legend = ax1.legend(
        patches,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=14,
        frameon=True,
        fancybox=True,
        framealpha=1,
        facecolor='white',
        edgecolor='#dddddd'
    )
    legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    # Main title and layout
    plt.title(title, fontsize=24, pad=25)
    plt.subplots_adjust(left=0.05, right=0.65, top=0.9)  # Shift plot left and ensure title visibility
    plt.tight_layout()

    # Save figure
    os.makedirs('Visualization Graphs', exist_ok=True)
    plt.savefig(
        f'Visualization Graphs/{filename}',
        dpi=500,
    )
    plt.close()
