import os
import numpy as np
import pandas as pd

def save_run_data(data, best_groups, trajectory_df, cfg, save_path):
    """
    Save all relevant data from a run including input data, results, and configuration.
    
    Args:
        data (dict): Dictionary containing input data
        best_groups (np.array): Array of group assignments
        cfg (DictConfig): Hydra configuration
        save_path (str): Directory to save files
    """
    # save input data
    data_dir = os.path.join(save_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'eigenmap.npy'), data['q'])
    
    # save sensitive attributes if they exist
    if 's' in data and data['s'] is not None:
        np.save(os.path.join(data_dir, 'sensitive_attributes.npy'), data['s'])

    # save loss curve
    if trajectory_df is not None:
        trajectory_df.to_csv(os.path.join(save_path, 'loss_curve.csv'), index=False)
    
    # save a group assignments CSV
    group_df = pd.DataFrame({
        'student_id': range(len(best_groups)),
        'group': best_groups
    })

    if 'df' in data:
        group_df = pd.concat([group_df, data['df']], axis=1)    
    group_df.to_csv(os.path.join(save_path, 'group_assignments.csv'), index=False)