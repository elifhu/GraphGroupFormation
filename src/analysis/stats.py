import pandas as pd
import numpy as np
import os

def plot_affinity_representation(best_groups, data, save_path):
    """
    Plot representation of affinity groups within tutor groups.
    
    Args:
        best_groups (np.ndarray): Array of group assignments for each student
        data (dict): Dictionary containing affinity_groups for each student
        save_path (str): Path to save the generated plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    groups = np.array(best_groups)
    unique_groups = np.unique(groups)
    affinity_groups = data['affinity_groups']
    unique_affinities = np.unique(affinity_groups)
    n_affinities = len(unique_affinities)
    
    # Create count matrix: rows=tutor_groups, cols=affinity_groups
    count_matrix = np.zeros((len(unique_groups), n_affinities))
    percentage_matrix = np.zeros((len(unique_groups), n_affinities))
    
    for i, g in enumerate(unique_groups):
        group_mask = groups == g
        group_affinities = affinity_groups[group_mask]
        for j in range(n_affinities):
            count_matrix[i, j] = np.sum(group_affinities == j)
            percentage_matrix[i, j] = np.sum(group_affinities == j) / np.sum(group_mask) * 100

    # 1. Bar plot showing counts per group
    plt.figure(figsize=(12, 6))
    bar_width = 0.8 / n_affinities
    x = np.arange(len(unique_groups))
    
    for i in range(n_affinities):
        plt.bar(x + i*bar_width - 0.4 + bar_width/2, 
                count_matrix[:, i], 
                bar_width,
                label=f'Affinity {i}')
    
    plt.xlabel('Tutor Group')
    plt.ylabel('Number of Students')
    plt.title('Affinity Group Representation in Each Tutor Group')
    plt.legend()
    plt.xticks(x, [f'Group {i+1}' for i in range(len(unique_groups))])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'affinity_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap showing percentages
    plt.figure(figsize=(12, 6))
    sns.heatmap(percentage_matrix.T, 
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                xticklabels=[f'Group {i+1}' for i in range(len(unique_groups))],
                yticklabels=[f'Affinity {i}' for i in range(n_affinities)])
    plt.xlabel('Tutor Group')
    plt.ylabel('Affinity Group')
    plt.title('Affinity Group Percentages in Each Tutor Group')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'affinity_percentages.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add summary statistics to stats.csv
    stats_dict = {
        'metric': [],
        'value': [],
        'std': [],
        'category': []
    }
    
    # Add overall representation stats
    for i in range(n_affinities):
        stats_dict['metric'].extend([
            f'affinity_{i}_min_count',
            f'affinity_{i}_max_count',
            f'affinity_{i}_mean_count',
            f'affinity_{i}_min_percentage',
            f'affinity_{i}_max_percentage',
            f'affinity_{i}_mean_percentage'
        ])
        stats_dict['value'].extend([
            np.min(count_matrix[:, i]),
            np.max(count_matrix[:, i]),
            np.mean(count_matrix[:, i]),
            np.min(percentage_matrix[:, i]),
            np.max(percentage_matrix[:, i]),
            np.mean(percentage_matrix[:, i])
        ])
        stats_dict['std'].extend([np.nan] * 6)
        stats_dict['category'].extend(['affinity_representation'] * 6)
    
    # Add to existing stats file
    stats_df = pd.read_csv(os.path.join(save_path, 'stats.csv'))
    new_stats = pd.DataFrame(stats_dict)
    stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'), index=False)

def plot_group_distance_distribution(best_groups, data, save_path):
    """
    Plot histogram of pairwise distances and save histogram data for later use.
    
    Args:
        best_groups (np.ndarray): Array of group assignments for each student
        data (dict): Dictionary containing eigenmap coordinates in 'q'
        save_path (str): Path to save the generated plot and data
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist
    
    # Convert groups to numpy array if needed
    groups = np.array(best_groups)
    unique_groups = np.unique(groups)
    
    # Collect all pairwise distances within groups
    within_group_distances = []
    
    for g in unique_groups:
        group_mask = groups == g
        group_q = data['q'][group_mask]
        
        if len(group_q) > 1:
            distances = pdist(group_q)
            within_group_distances.extend(distances)
    
    # Create histogram
    counts, bin_edges, _ = plt.hist(within_group_distances, bins=30, edgecolor='black')
    
    # Calculate statistics
    mean_dist = np.mean(within_group_distances)
    std_dist = np.std(within_group_distances)
    median_dist = np.median(within_group_distances)
    
    # Add statistics to plot
    stats_text = f'Mean: {mean_dist:.3f}\nStd: {std_dist:.3f}\nMedian: {median_dist:.3f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Pairwise Distance in Eigenmap Space')
    plt.ylabel('Frequency')
    plt.title('Distribution of Within-Group Pairwise Distances')
    
    # Save plot
    plt.savefig(os.path.join(save_path, 'group_distances_hist.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save histogram data to numpy file
    histogram_data = {
        'bin_edges': bin_edges,
        'counts': counts,
        'raw_distances': np.array(within_group_distances),
        'statistics': {
            'mean': mean_dist,
            'std': std_dist,
            'median': median_dist
        }
    }
    np.save(os.path.join(save_path, 'histogram_data.npy'), histogram_data)
    
    # Add stats to stats.csv
    stats_df = pd.read_csv(os.path.join(save_path, 'stats.csv'))
    
    new_stats = pd.DataFrame({
        'metric': ['within_group_distance_mean', 
                  'within_group_distance_std',
                  'within_group_distance_median'],
        'value': [mean_dist, std_dist, median_dist],
        'std': [np.nan, np.nan, np.nan],
        'category': ['distances'] * 3
    })
    
    stats_df = pd.concat([stats_df, new_stats], ignore_index=True)
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'), index=False)

def get_optimal_solution_circle(n_pairs, use_balance_constraint=True):
    if n_pairs % 2 != 0:
        raise NotImplementedError("Only even number of pairs is supported.")
    angle_between_students = np.pi / n_pairs
    min_dist = 2 * np.sin(angle_between_students / 2)
    max_dist = 2 * np.sin((np.pi - np.pi/n_pairs) / 2) if use_balance_constraint else 2
    return min_dist * n_pairs * 2, max_dist * n_pairs * 2

def print_grouping_stats(best_groups, best_cost, data, config, save_path):
    """
    Print and save comprehensive statistics about the grouping solution.
    Creates both human-readable output and a CSV file.
    """

    # Initialize dictionary to store statistics
    stats_dict = {
        'metric': [],
        'value': [],
        'std': [],
        'category': []
    }

    # Add theoretical optimal for circle data
    if config['data']['name'] == 'circle':
        N = len(best_groups)
        n_pairs = N // 2
        min_cost, max_cost = get_optimal_solution_circle(n_pairs, config['optim']['params']['balance'] != 0.0)
        
        # Use min or max depending on optimization direction
        optimal_cost = min_cost if not config['optim']['params']['maximize'] else max_cost
        
        stats_dict['metric'].append('theoretical_optimal_cost')
        stats_dict['value'].append(optimal_cost)
        stats_dict['std'].append(np.nan)
        stats_dict['category'].append('optimization')
        
        # Add gap to optimal
        gap = abs(best_cost - optimal_cost) / abs(optimal_cost) * 100
        stats_dict['metric'].append('gap_to_optimal_percent')
        stats_dict['value'].append(gap)
        stats_dict['std'].append(np.nan)
        stats_dict['category'].append('optimization')
    
    # Convert group assignments to numpy array if needed
    groups = np.array(best_groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    # Basic group statistics
    group_sizes = [np.sum(groups == g) for g in unique_groups]
    stats_dict['metric'].extend(['number_of_groups', 'group_size_mean', 'group_size_min', 'group_size_max'])
    stats_dict['value'].extend([n_groups, np.mean(group_sizes), np.min(group_sizes), np.max(group_sizes)])
    stats_dict['std'].extend([np.nan, np.std(group_sizes), np.nan, np.nan])
    stats_dict['category'].extend(['basic'] * 4)
    
    # Skill diversity statistics using eigenmap
    within_group_vars = []
    group_means = []
    
    for g in unique_groups:
        group_mask = groups == g
        group_q = data['q'][group_mask]
        within_var = np.var(group_q, axis=0)
        within_group_vars.append(within_var)
        group_means.append(np.mean(group_q, axis=0))
    
    stats_dict['metric'].extend(['within_group_variance', 'between_group_variance'])
    stats_dict['value'].extend([np.mean(within_group_vars), np.var(group_means, axis=0).mean()])
    stats_dict['std'].extend([np.std(within_group_vars), np.std(np.var(group_means, axis=0))])
    stats_dict['category'].extend(['skill_diversity'] * 2)
    
    # Sensitive attribute statistics if available
    if 's' in data and data['s'] is not None:
        sensitive_ratios = []
        for g in unique_groups:
            group_mask = groups == g
            group_s = data['s'][group_mask]
            ratios = np.mean(group_s, axis=0)
            sensitive_ratios.append(ratios)
        
        sensitive_ratios = np.array(sensitive_ratios)
        overall_ratios = np.mean(data['s'], axis=0)
        
        for i in range(data['s'].shape[1]):
            stats_dict['metric'].extend([f'sensitive_attr_{i}_population_mean', 
                                       f'sensitive_attr_{i}_group_ratio_mean'])
            stats_dict['value'].extend([overall_ratios[i], 
                                      np.mean(sensitive_ratios[:, i])])
            stats_dict['std'].extend([np.nan, 
                                    np.std(sensitive_ratios[:, i])])
            stats_dict['category'].extend(['fairness'] * 2)
    
    # Course marks statistics if available
    if config['data']['name'] == 'real' and 'df' in data:
        df = data['df']
        non_course_cols = ['No.']
        course_cols = [col for col in df.columns if col not in non_course_cols]
        
        if course_cols:
            marks_matrix = df[course_cols].values
            
            # Correlation statistics
            within_corrs = []
            between_corrs = []
            
            for g in unique_groups:
                group_mask = groups == g
                group_marks = marks_matrix[group_mask]
                
                if len(group_marks) > 1:
                    group_corr = np.corrcoef(group_marks)
                    triu_indices = np.triu_indices_from(group_corr, k=1)
                    within_corrs.extend(group_corr[triu_indices])
                
                other_mask = ~group_mask
                if np.any(other_mask):
                    other_marks = marks_matrix[other_mask]
                    between_corr = np.corrcoef(group_marks, other_marks)
                    between_corrs.extend(between_corr[:len(group_marks), len(group_marks):].flatten())
            
            stats_dict['metric'].extend(['within_group_correlation', 'between_group_correlation'])
            stats_dict['value'].extend([np.mean(within_corrs), np.mean(between_corrs)])
            stats_dict['std'].extend([np.std(within_corrs), np.std(between_corrs)])
            stats_dict['category'].extend(['marks_correlation'] * 2)
            
            # Course-specific statistics
            for course in course_cols:
                group_means = []
                group_vars = []
                
                for g in unique_groups:
                    group_mask = groups == g
                    group_marks = df.loc[group_mask, course]
                    group_means.append(group_marks.mean())
                    group_vars.append(group_marks.var())
                
                stats_dict['metric'].extend([f'{course}_within_group_variance',
                                           f'{course}_between_group_variance'])
                stats_dict['value'].extend([np.mean(group_vars), np.var(group_means)])
                stats_dict['std'].extend([np.std(group_vars), np.nan])
                stats_dict['category'].extend(['course_marks'] * 2)
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(stats_dict)
    
    # Add run configuration information
    if config['optim']['params']['balance'] is None:
        config['optim']['params']['balance'] = 0.0

    config_info = pd.DataFrame({
        'metric': ['maximize', 'min_group_size', 'max_group_size', 'balance'],
        'value': [config['optim']['params']['maximize'],
                 config['optim']['params']['min_group_size'],
                 config['optim']['params']['max_group_size'],
                 config['optim']['params']['balance']],
        'std': [np.nan] * 4,
        'category': ['config'] * 4
    })
    
    stats_df = pd.concat([config_info, stats_df], ignore_index=True)
    
    # Save to CSV
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'), index=False)
    
    # Also print to console in a readable format
    print("\n=== Grouping Statistics ===")
    for category in stats_df['category'].unique():
        print(f"\n{category.upper()}:")
        category_stats = stats_df[stats_df['category'] == category]
        for _, row in category_stats.iterrows():
            if pd.isna(row['std']):
                print(f"{row['metric']}: {row['value']:.4f}")
            else:
                print(f"{row['metric']}: {row['value']:.4f} ± {row['std']:.4f}")

    if config['data']['name'] == 'simulated':
        plot_group_distance_distribution(best_groups, data, save_path)
        plot_affinity_representation(best_groups, data, save_path)
        np.save(os.path.join(save_path, 'affinity_groups.npy'), data['affinity_groups'])

