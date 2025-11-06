import matplotlib.pyplot as plt
import os

def plot_portfolio_allocation(weights, title, filename):
    """
    Plot portfolio allocation as a pie chart and save to file
    
    Parameters:
    weights (Series): Portfolio weights
    title (str): Plot title
    filename (str): Output filename (without extension)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot pie chart
    weights.plot.pie(
        ax=ax,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 8}
    )
    
    # Set title
    ax.set_title(title, fontsize=14, pad=20)
    
    # Remove y-label
    ax.set_ylabel('')
    
    # Ensure output directory exists
    os.makedirs('Visualization Graphs', exist_ok=True)
    
    # Save figure
    plt.savefig(f'Visualization Graphs/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()
