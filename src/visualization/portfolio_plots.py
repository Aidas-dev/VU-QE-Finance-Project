import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

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

    # Separate significant weights (>1%) and small weights (0% to 1%)
    significant_weights = weights[abs(weights) > 0.01]  # > 1%
    small_weights = weights[(abs(weights) > 0) & (abs(weights) <= 0.01)]  # 0% to 1%

    # Create "Others" category if there are small weights
    if len(small_weights) > 0:
        others_total = small_weights.sum()
        if abs(others_total) > 0:
            # Add Others to significant weights for plotting
            significant_weights = significant_weights.copy()
            significant_weights['Others'] = others_total

    # Sort by absolute weight descending
    significant_weights = significant_weights.iloc[abs(significant_weights).argsort()[::-1]]
    labels = significant_weights.index

    # Create figure with adjusted margins for left legend
    fig = plt.figure(figsize=(12, 9), dpi=500)
    plt.subplots_adjust(left=0.25, right=0.95)  # Make space on left for legend

    ax1 = fig.add_subplot(111)  # Main plot area

    # Create pie chart without labels or percentages
    patches = ax1.pie(
        abs(significant_weights),
        labels=None,
        startangle=90,
        colors=sns.color_palette("mako", len(significant_weights)),
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )[0]

    # Create detailed legend with allocation info - SORTED DESCENDING
    legend_labels = [
        f"{label} ({'Long' if significant_weights[label] > 0 else 'Short'}): {abs(significant_weights[label]):.1%}"
        for label in labels
    ]

    # Sort patches and labels by absolute weight descending
    sorted_indices = abs(significant_weights).argsort()[::-1]
    sorted_patches = [patches[i] for i in sorted_indices]
    sorted_legend_labels = [legend_labels[i] for i in sorted_indices]

    legend = ax1.legend(
        sorted_patches,
        sorted_legend_labels,
        loc='center left',
        bbox_to_anchor=(-0.3, 0.5),  # Move legend to left side
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        facecolor='white',
        edgecolor='#dddddd'
    )
    legend.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    # Main title and layout
    fig.suptitle(title, fontsize=24, y=0.95)
    plt.tight_layout()

    # Save figure
    os.makedirs('Visualization Graphs', exist_ok=True)
    plt.savefig(
        f'Visualization Graphs/{filename}',
        dpi=500,
        bbox_inches='tight'
    )
    plt.close()