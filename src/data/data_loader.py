# src/data/data_loader.py

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from src.eigenmap.eigenmap import load_eigenmaps_pandas, compute_spectral_colors

class RealStudentLoader:
    """Class to handle all data loading and processing operations"""
    
    def __init__(self, config):
        self.config = config
        self.df_eee = None
        self.df_eie = None
        self.q_eee = None
        self.q_eie = None
        self.colors_eee = None
        self.colors_eie = None
    
    def load_all(self):
        """Load and process all data"""
        self._load_raw_data()
        self._compute_eigenmaps()
        return self.get_cohort_data()
    
    def _load_raw_data(self):
        """Load raw data from file"""
        from src.data.imperial import load_data
        self.df_eee, self.df_eie = load_data(
            self.config['save_path'],
            self.config['data_params']['fn_name'],
            self.config['data_params']
        )
    
    def _compute_eigenmaps(self):
        """Compute eigenmaps for both cohorts"""
        c_thresh = self.config['c_thresh']
        denom = self.config['denom']
        files_suffix = f'_{c_thresh}_{denom}'
        
        qs = load_eigenmaps_pandas(
            [self.df_eee, self.df_eie],
            [f"{self.config['save_path']}/eee{files_suffix}.npy",
             f"{self.config['save_path']}/eie{files_suffix}.npy"],
            c_thresh,
            denom
        )
        
        self.q_eee, self.q_eie = qs[0], qs[1]
        self.colors_eee = compute_spectral_colors(self.q_eee)
        self.colors_eie = compute_spectral_colors(self.q_eie)
    
    def get_cohort_data(self):
        """Get data for specified cohort"""
        cohort = self.config['data_params']['cohort']
        
        if cohort == 'eee':
            q = self.q_eee
            df = self.df_eee
            colors = self.colors_eee
        else:
            q = self.q_eie
            df = self.df_eie
            colors = self.colors_eie
            
        # Create sensitive attributes
        N = q.shape[0]
        s = np.zeros((N, 1)).astype(int)
        s[::2] = 1
        s = s.flatten()[:, None]
        
        return {
            'q': q,
            's': s,
            'df': df,
            'colors': colors,
            'distances': cdist(q, q).flatten()
        }
    
class CircleStudentLoader:
    """ load data where pairs of students are generated opposite each other on a circle in the eigenmap space, 
    that is, students lay on circle x^2 + y^2 = 1"""
    def __init__(self, config):
        self.config = config
        self.q = None
        self.colors = None
    
    def load_all(self):
        """Load and process all data"""
        return self._create_circle_data(self.config['data_params']['n_pairs'])
    
    def _create_circle_data(self, n_pairs):
        """Create benchmark with n_pairs of students (total N = 2*n_pairs).
        Students placed on unit circle with same sensitive attributes placed 
        opposite each other (180 degrees apart).
        
        For example, with n_pairs=4 (N=8):
        - 4 students with attribute 1 placed at angles [0, 90, 180, 270]
        - 4 students with attribute 0 placed at angles [45, 135, 225, 315]
        """
        N = 2 * n_pairs
        # Create two sets of evenly spaced angles for each attribute
        # First set (e.g., [0, 90, 180, 270] for n_pairs=4)
        theta1 = np.linspace(0, 2*np.pi, n_pairs, endpoint=False)
        # Second set offset by pi/n_pairs (e.g., [45, 135, 225, 315])
        theta2 = theta1 + np.pi/n_pairs
        
        # Combine and sort angles
        theta = np.sort(np.concatenate([theta1, theta2]))
        
        # Create coordinates on unit circle
        x = np.cos(theta)
        y = np.sin(theta)
        q = np.column_stack([x, y])
        
        # Create sensitive attributes
        # First half of students (on theta1) get attribute 1
        # Second half (on theta2) get attribute 0
        s = np.zeros((N, 1))
        s[::2] = 1  # Students at theta1 angles get attribute 1
        
        return {
            'q': q,
            's': s,
            'distances': cdist(q, q).flatten()
        }