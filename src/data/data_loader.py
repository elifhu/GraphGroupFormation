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
        """Load raw data from Excel"""
        path = self.config['data_params']['f']
        include_components = self.config['data_params'].get('include_components', False)

        # First try normal header
        df = pd.read_excel(path, header=0)
        df.columns = [
            ' '.join(
                str(c).replace('\n', ' ').replace('_x000D_', ' ').strip().split()
            )
            for c in df.columns
        ]

        # If this looks like the original raw Year1 file with wrong header, retry with header=1
        if ('CID' not in df.columns) and ('No.' not in df.columns):
            df = pd.read_excel(path, header=1)
            df.columns = [
                ' '.join(
                    str(c).replace('\n', ' ').replace('_x000D_', ' ').strip().split()
                )
                for c in df.columns
            ]

        print("COLUMNS:", df.columns.tolist())

        # Use either No. or CID as student identifier
        if 'No.' in df.columns:
            id_col = 'No.'
        elif 'CID' in df.columns:
            id_col = 'CID'
            df = df.rename(columns={'CID': 'No.'})
            id_col = 'No.'
        else:
            raise KeyError(f"Could not find student ID column. Columns were: {df.columns.tolist()}")

        df['No.'] = df['No.'].astype(str).str.strip()

        base_cols = [
            'No.',
            'ELEC40002 Analysis & Design of Circuits',
            'ELEC40003 Digital Electronic & Comp Arch',
            'ELEC40004 Programming for Engineers',
            'ELEC40006 Electronics Design Project 1',
            'ELEC40009 Topics in Elec Engineering',
            'ELEC40012 Mathematics 1',
        ]

        component_cols = [
            'ELEC40002 exam',
            'ELEC40002 MT Test 1',
            'ELEC40002 Oral 1',
            'ELEC40002 MT Test 2',
            'ELEC40002 Oral 2',
            'ELEC40003 exam',
            'ELEC40003 MT Test 1',
            'ELEC40003 Oral 1',
            'ELEC40003 MT Test 2',
            'ELEC40003 Oral 2',
            'ELEC40004 MT Tests',
            'ELEC40004 CW',
            'ELEC40012A exam',
            'ELEC40012A cw',
            'ELEC40012B exam',
            'ELEC40012B cw',
        ]

        # For the model, keep only ID + numeric mark columns
        keep_cols = base_cols + (component_cols if include_components else [])
        keep_cols = [c for c in keep_cols if c in df.columns]

        print("KEEP_COLS:", keep_cols)

        if 'No.' not in keep_cols:
            raise ValueError("No usable student ID column found after filtering.")

        mark_cols = [c for c in keep_cols if c != 'No.']

        if len(mark_cols) == 0:
            raise ValueError(f"No usable mark columns found. Available columns: {df.columns.tolist()}")

        df = df[keep_cols].copy()

        for c in mark_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop rows with no marks at all
        df = df.dropna(subset=mark_cols, how='all').reset_index(drop=True)

        # Fill remaining missing values with column means
        df[mark_cols] = df[mark_cols].fillna(df[mark_cols].mean())

        # Since each input file is already cohort-specific, we can store the same df
        self.df_eee = df.copy()
        self.df_eie = df.copy()


    
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


class SimulatedStudentLoader:
    """Simulate course marks from a normal distribution"""
    
    def __init__(self, config):
        self.config = config
        self.num_students = config['data_params']['num_students']
        self.num_courses = config['data_params']['num_courses']
        self.num_affinities = config['data_params']['num_affinities']
        self.affinity_mark = config['data_params']['affinity_mark']
        self.non_affinity_mark = config['data_params']['non_affinity_mark']
        self.affinity_mark_range = config['data_params']['affinity_mark_range']
        self.corr_range = config['data_params']['corr_range']
        self.high_corr = config['data_params']['high_corr']
        self.low_corr = config['data_params']['low_corr']

    def load_all(self):
        """Load and process all data"""
        self._simulate_data()
        self._compute_eigenmaps()
        return self._get_data()
    
    def _simulate_data(self):
        """Simulate data according to multivariate gaussian for different affinity groups"""

        # Initialize arrays to store all student data
        all_students = []
        student_ids = []
        affinity_groups = []

        # Calculate how many students per affinity group (roughly equal split)
        students_per_affinity = self.num_students // self.num_affinities
        remainder = self.num_students % self.num_affinities

        for affinity_id in range(self.num_affinities):
            # Add remainder students to first few groups
            n_students = students_per_affinity + (1 if affinity_id < remainder else 0)
            
            # Create mean vector for this affinity group
            # Each group has higher means in their affinity subjects
            # and lower means in other subjects
            mean = np.zeros(self.num_courses)
            affinity_courses = np.array_split(range(self.num_courses), self.num_affinities)[affinity_id]
            mean[affinity_courses] = self.affinity_mark + np.random.randn(len(affinity_courses)) * self.affinity_mark_range  # High mean for affinity courses
            other_courses = list(set(range(self.num_courses)) - set(affinity_courses))
            mean[other_courses] = self.non_affinity_mark + np.random.randn(len(other_courses)) * self.affinity_mark_range  # Lower mean for other courses
            
            # Create covariance matrix
            # Use high correlation if both subjects are in affinity group
            # Use low correlation otherwise
            cov = np.eye(self.num_courses) * 100  # Base variance on diagonal
            
            for i in range(self.num_courses):
                for j in range(i+1, self.num_courses):
                    corr = self.high_corr if (i in affinity_courses and j in affinity_courses) else self.low_corr
                    
                    # Convert correlation to covariance
                    cov[i,j] = corr * np.sqrt(cov[i,i] * cov[j,j])
                    cov[j,i] = cov[i,j]  # Ensure symmetry
            
            # Generate student marks using multivariate normal distribution
            marks = np.random.multivariate_normal(mean, cov, size=n_students)
            
            # Clip marks to valid range [0, 100]
            marks = np.clip(marks, 0, 100)
            
            # Store student data
            all_students.append(marks)
            student_ids.extend([f"student_{i}" for i in range(
                len(all_students[-1]))])
            affinity_groups.extend([affinity_id] * n_students)

        # Combine all students into single array
        all_marks = np.vstack(all_students)

        # create df
        df_dict = {
            'No.': student_ids
        }
        for i in range(self.num_courses):
            df_dict[f'course_{i}'] = all_marks[:, i]
        self.df = pd.DataFrame(df_dict)
        self.affinity_groups = np.array(affinity_groups)
    
    def _compute_eigenmaps(self):
        """Compute eigenmaps for both cohorts"""
        c_thresh = self.config['c_thresh']
        denom = self.config['denom']
        files_suffix = f'_n{self.num_students}_c{self.num_courses}_a{self.num_affinities}_{c_thresh}_{denom}'
        
        qs = load_eigenmaps_pandas(
            [self.df],
            [f"{self.config['save_path']}/sim{files_suffix}.npy"],
            c_thresh,
            denom)
    
        self.q = qs[0]
        self.colors = compute_spectral_colors(self.q)
    
    def _get_data(self):
        """Get data for specified cohort"""
            
        # Create sensitive attributes
        N = self.q.shape[0]
        s = np.zeros((N, 1)).astype(int)

        return {
            'q': self.q,
            's': s,
            'df': self.df,
            'colors': self.colors,
            'distances': cdist(self.q, self.q).flatten(),
            'affinity_groups': self.affinity_groups
        }
