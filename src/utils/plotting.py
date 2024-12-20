import numpy as np
import matplotlib.pyplot as plt

def plot_circle_groups(q, s, groups, title="", subtitle="", figsize=(4, 4)):
    """
    Plot students arranged on a circle with their groupings.
    
    Args:
        q (np.array): (N,2) array of x,y positions on circle
        s (np.array): (N,1) array of sensitive attributes (0 or 1)
        groups: Either a 1D array of group assignments or list of lists with indices
        title (str): Main title for plot
        subtitle (str): Subtitle for additional information
        figsize (tuple): Figure size
    """
    # Setup figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot circle
    circle = plt.Circle((0, 0), 1, fill=False, color='#cccccc', linewidth=1)
    ax.add_artist(circle)
    
    # Colors for students based on sensitive attribute
    colors = ['#4A90E2', '#FF7676']  # Blue for 0, Red for 1
    
    # Plot students
    for i in range(len(q)):
        plt.scatter(q[i,0], q[i,1], c=colors[int(s[i])], s=100, zorder=3)
    
    # Convert group assignments to list of groups if necessary
    if isinstance(groups, np.ndarray) and groups.ndim == 1:
        unique_groups = np.unique(groups)
        groups_list = [np.where(groups == g)[0] for g in unique_groups]
    else:
        groups_list = groups
    
    # Plot group connections
    for idx, group in enumerate(groups_list):
        # Different line styles for different groups
        linestyles = ['-', '--', ':', '-.']
        style = linestyles[idx % len(linestyles)]
        
        # Get indices in this group
        indices = np.array(group)
        
        # Connect all points in the group
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                plt.plot([q[indices[i],0], q[indices[j],0]], 
                        [q[indices[i],1], q[indices[j],1]], 
                        '#45B7D1', linestyle='-', linewidth=1.5, alpha=0.7,
                        zorder=2)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    
    # Remove axes
    plt.axis('off')
    
    # Add titles
    plt.title(title + '\n' + subtitle if subtitle else title, 
              pad=20, fontsize=12)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c=colors[0], s=100, label='Without Attribute'),
        plt.scatter([], [], c=colors[1], s=100, label='With Attribute'),
        plt.Line2D([0], [0], color='#45B7D1', label='Group'),
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    return plt.gcf()