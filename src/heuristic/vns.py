import itertools
import numpy as np
import time
import pandas as pd

from functools import wraps
from scipy.spatial.distance import cdist
from tqdm import tqdm
from src.utils.misc import find_sum_combinations

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f'Starting {func.__name__}...')
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'Finished {func.__name__}. Total time: {end_time - start_time:.4f} seconds')
        return result
    return wrapper

class GroupOptimizer:
    def __init__(self, q, s, min_group_size, max_group_size, lower_balance, seed=42, maximize=True, balance_reg_weight=0.0, size_reg_weight=0.0):
        self.q = q
        self.N = q.shape[0]
        print(f'Initializing optimizer with {self.N} students...')
        
        self.s = s
        self.d = cdist(q, q).flatten()
        self.seed = seed
        self.check_constraints(min_group_size, max_group_size, lower_balance)
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.lower_balance = lower_balance
        self.maximize = maximize
        self.balance_reg_weight = balance_reg_weight
        self.size_reg_weight = size_reg_weight
        
        self.current_solution = None
        self.best_solution = None
        self.best_objective = float('-inf')

    def check_constraints(self, min_group_size, max_group_size, balance):
        assert min_group_size <= max_group_size, 'min_group_size must be less than or equal to max_group_size'
        if balance is not None:
            assert balance < 1, 'balance must be less than 1'
            assert balance >= 0, 'balance must be greater than 0'
            self.use_balance = True
        else:
            self.use_balance = False
    
    def base_objective(self, solution):
        return np.sum(solution.flatten() * self.d)
    
    def evaluate_solution(self, solution):
        base_obj = self.base_objective(solution)
        balance_penalty = 0.0 if not self.use_balance else self.balance_reg_weight * self._balance_l2_penalty(solution)
        size_penalty = self.size_reg_weight * self._size_l2_penalty(solution)
        if self.maximize:
            return base_obj - balance_penalty - size_penalty
        return base_obj + balance_penalty + size_penalty
        
    def check_feasibility(self, solution):
        size_feasible = self._check_sizes(solution)
        if not size_feasible:
            return False
        
        if self.use_balance:
            balance_feasible = self._check_balance(solution)
            if not balance_feasible:
                return False
                
        triangle_feasible = self._check_triangle(solution)
        if not triangle_feasible:
            return False

        return True

    def _compute_group_sizes(self, solution):
        group_sizes = np.sum(solution, axis=1)
        return group_sizes

    def _check_sizes(self, solution):
        group_sizes = self._compute_group_sizes(solution)
        size_feasible = all(self.min_group_size <= size <= self.max_group_size 
                          for size in group_sizes if size > 0)
        return size_feasible
    
    def _size_l2_penalty(self, solution):
        group_sizes = self._compute_group_sizes(solution)
        penalty = 0
        for size in group_sizes:
            size_outside_bound_amount = max(0, size - self.max_group_size) + max(0, self.min_group_size - size)
            penalty += size_outside_bound_amount ** 2
        return penalty

    def _check_triangle(self, solution):
        row_expanded = solution[:, np.newaxis, :]
        col_expanded = solution[np.newaxis, :, :]
        return np.all(solution + row_expanded - col_expanded <= 1)

    def _compute_balance(self, solution):
        group_sizes = np.sum(solution, axis=1)
        sensitive_in_groups = (solution @ self.s)
        R_group = sensitive_in_groups / (group_sizes[:, None] + 1e-10)
        R_pop = np.sum(self.s, axis=0) / self.N
        R_group_to_pop = R_group / R_pop
        balance = np.minimum(R_group_to_pop, 1 / (R_group_to_pop + 1e-10))
        return balance

    def _check_balance(self, solution):
        balance = self._compute_balance(solution)
        if np.any(balance < self.lower_balance):
            return False
        return True
    
    def _balance_l2_penalty(self, solution):
        balance = self._compute_balance(solution)
        balance_outside_bound_amount = np.maximum(0, self.lower_balance - balance)
        penalty = np.sum((1+balance_outside_bound_amount) ** 2)
        return penalty

    def get_groups(self, solution):
        return np.unique(solution, axis=0, return_inverse=True)[1]

    def initialize_solution(self):
        """ random initial solution """
        np.random.seed(self.seed)
        group_size_combos = self._find_valid_group_sizes()
        np.random.shuffle(group_size_combos)

        # Random initial solution which satisfies the group size constraints
        group_sizes = group_size_combos[0]
        group_assignments = np.zeros((self.N, self.N)).astype(int)
        
        start_ind = 0
        for group_size in group_sizes:
            end_ind = start_ind + group_size
            group_assignments[start_ind:end_ind, start_ind:end_ind] = 1
            start_ind = end_ind

        student_indices = np.arange(self.N)
        np.random.shuffle(student_indices)
        return group_assignments[student_indices][:, student_indices]

    def _find_valid_group_sizes(self):
        combos = find_sum_combinations(self.min_group_size, self.max_group_size, self.N)
        if len(combos) <= 0:
            raise ValueError('Satisfying group sizes not possible, adjust min_group_size and max_group_size')
        return combos


class VNSGroupOptimizer(GroupOptimizer):
    def __init__(self, q, s, min_group_size, max_group_size, lower_balance, \
                 seed=42, maximize=True, pvns=[0.5, 0.5], max_k=2, space_n_sample=None, balance_reg_weight=1.0, size_reg_weight=1.0):
        super().__init__(q, s, min_group_size, max_group_size, lower_balance, seed, maximize, balance_reg_weight, size_reg_weight)
        
        self.pvns = pvns
        self.max_k = max_k
        self.space_n_sample = space_n_sample
        if self.lower_balance == 0.0:
            print('Warning: Lower balance is set to 0.0. Balance regularization will not be effective.')
            self.balance_reg_weight = 0.0

        if self.space_n_sample is None and self.N > 50:
            print('Warning: Large number of students. Consider setting space_n_sample to sample space of possible solutions for efficiency.')
        if self.min_group_size == self.max_group_size:
            print('Warning: min_group_size equals max_group_size. Single moves are not possible.')
            self.pvns = [0.0, 1.0]
    
    def get_neighborhood(self, solution, k):
        if k == 1:
            moves, objs = self._generate_single_moves(solution)
            move_type = 'single'
        elif k == 2:
            moves, objs = self._generate_pair_swaps(solution)
            move_type = 'swap'
        else:
            raise NotImplementedError
        
        if moves:
            picked_ind = -1 if self.maximize else 0
            best_ind = np.argsort(objs)[picked_ind]
            best_move = moves[best_ind]
            best_obj = objs[best_ind]
            return (move_type, best_move), best_obj
        return None

    def _generate_single_moves(self, solution):
        unique_groups = self.get_groups(solution)
        num_groups = len(np.unique(unique_groups))
        
        # Create all possible student-target group pairs
        students = np.arange(self.N)
        target_groups = np.arange(num_groups)
        
        # Create meshgrid of all possible combinations
        student_indices, group_indices = np.meshgrid(students, target_groups)
        
        # Flatten and stack to get all possible moves
        all_moves = np.stack([student_indices.flatten(), group_indices.flatten()], axis=1)
        
        # Filter out moves where student is already in target group
        valid_moves_mask = unique_groups[all_moves[:, 0]] != all_moves[:, 1]
        possible_moves = all_moves[valid_moves_mask]
        
        # Sample if necessary
        if self.space_n_sample is not None and len(possible_moves) > self.space_n_sample:
            sample_indices = np.random.choice(
                range(len(possible_moves)), 
                self.space_n_sample, 
                replace=False
            )
            possible_moves = possible_moves[sample_indices]
        
        valid_moves = []
        valid_objs = []
        moves_checked = 0
        
        # Check feasibility of sampled moves
        for student, target_group in possible_moves:
            moves_checked += 1
            test_solution = self._apply_single_move(solution, (student, target_group))
            valid_moves.append((student, target_group))
            valid_objs.append(self.evaluate_solution(test_solution))
        return valid_moves, valid_objs
        
    def _generate_pair_swaps(self, solution):
        unique_groups = self.get_groups(solution)
        
        i_indices, j_indices = np.meshgrid(np.arange(self.N), np.arange(self.N))
        mask = (j_indices > i_indices) & (unique_groups[i_indices] != unique_groups[j_indices])
        
        valid_pairs = np.stack([i_indices[mask], j_indices[mask]], axis=1)
        
        valid_swaps = []
        valid_objs = []

        if self.space_n_sample is not None and len(valid_pairs) > self.space_n_sample:
            sample_indices = np.random.choice(range(len(valid_pairs)), self.space_n_sample, replace=False)
            valid_pairs = valid_pairs[sample_indices]

        pairs_checked = 0
        for i, j in valid_pairs:
            pairs_checked += 1
            test_solution = solution.copy()
            np.fill_diagonal(test_solution, 0)
            test_solution[[i, j]] = test_solution[[j, i]]
            test_solution[:, [i, j]] = test_solution[:, [j, i]]
            np.fill_diagonal(test_solution, 1)

            valid_swaps.append((i, j))
            valid_objs.append(self.evaluate_solution(test_solution))

        return valid_swaps, valid_objs

    def _apply_single_move(self, solution, move):
        student, target_group = move
        new_solution = solution.copy()
        
        unique_patterns = self.get_groups(solution)
        target_group_students = np.where(unique_patterns == target_group)[0]
        
        new_solution[student, :] = 0
        new_solution[:, student] = 0
        
        new_solution[student, target_group_students] = 1
        new_solution[target_group_students, student] = 1
        np.fill_diagonal(new_solution, 1)
        return new_solution

    def _apply_single_swap(self, solution, move):
        i, j = move
        new_solution = solution.copy()
        np.fill_diagonal(new_solution, 0)
        
        temp = new_solution[i].copy()
        new_solution[i] = new_solution[j]
        new_solution[j] = temp
        
        temp = new_solution[:, i].copy()
        new_solution[:, i] = new_solution[:, j]
        new_solution[:, j] = temp
        np.fill_diagonal(new_solution, 1)
        return new_solution
    
    def local_search(self, solution, max_local_search_tries):
        current = solution.copy()
        current_obj = self.evaluate_solution(current)
        tries = 0
        improvements = 0
        available_k = [k for k in range(1, self.max_k + 1) if self.pvns[k - 1] > 0]
        
        while tries < max_local_search_tries:
            best_move = None
            best_obj = float('-inf') if self.maximize else float('inf')
            for k in available_k:
                n_out = self.get_neighborhood(current, k)
                if not n_out:
                    tries += 1
                    continue
                
                (move_type, move), new_obj = n_out
                if (self.maximize and new_obj > best_obj) or (not self.maximize and new_obj < best_obj):
                    best_move = (move_type, move)
                    best_obj = new_obj

            if (self.maximize and best_obj > current_obj) or (not self.maximize and best_obj < current_obj):
                move_type, move = best_move
                if move_type == 'single':
                    current = self._apply_single_move(current, move)
                elif move_type == 'swap':
                    current = self._apply_single_swap(current, move)
                current_obj = new_obj
                improvements += 1
            tries += 1
            
        return current
        
    def shake(self, solution):
        perturbed = solution.copy()
        
        move_type = np.random.choice(['single', 'swap'], p=self.pvns)
        
        if move_type == 'swap':
            valid_moves, _ = self._generate_pair_swaps(perturbed)
            if valid_moves:
                move = valid_moves[np.random.randint(len(valid_moves))]
                perturbed = self._apply_single_swap(perturbed, move)
                
        elif move_type == 'single':
            valid_moves, _ = self._generate_single_moves(perturbed)
            if valid_moves:
                move = valid_moves[np.random.randint(len(valid_moves))]
                perturbed = self._apply_single_move(perturbed, move)
    
        else:
            raise NotImplementedError

        return perturbed
        
    def optimize(self, max_optim_steps=20, max_no_improve=5, max_local_search_tries=10, return_groups_only=True, **kwargs):
        """
        Run optimization with tracking of objective values and timing.
        Returns optimization trajectory data alongside results.
        """
        # Initialize tracking lists with both objectives
        trajectory = {
            'iteration': [],
            'current_objective': [],          # Full objective with regularization
            'current_base_objective': [],     # Base objective without regularization
            'best_objective': [],             # Best full objective
            'best_base_objective': [],        # Best base objective
            'iteration_time': [],
            'cumulative_time': [],
            'improvement_flag': []
        }
        
        # Random initialization
        self.current_solution = self.initialize_solution()
        self.best_solution = self.current_solution.copy()
        self.best_objective = self.evaluate_solution(self.best_solution)  # Full objective
        self.best_base_objective = self.base_objective(self.best_solution)     # Base objective

        if self.maximize is None:
            if return_groups_only:
                return self.get_groups(self.best_solution), None
            return self.get_groups(self.best_solution), (self.best_solution, self.best_base_objective), None

        total_start_time = time.time()
        
        # Record initial state with both objectives
        trajectory['iteration'].append(0)
        trajectory['current_objective'].append(self.best_objective)
        trajectory['current_base_objective'].append(self.best_base_objective)
        trajectory['best_objective'].append(self.best_objective)
        trajectory['best_base_objective'].append(self.best_base_objective)
        trajectory['iteration_time'].append(0)
        trajectory['cumulative_time'].append(0)
        trajectory['improvement_flag'].append(True)
        
        iter_no_improve = 0
        iters = 0

        print('Starting optimization...')
        with tqdm(total=max_optim_steps) as pbar:
            while (iter_no_improve < max_no_improve) and (iters < max_optim_steps):
                iter_start_time = time.time()
                
                # Shake and optimize
                temp_solution = self.shake(self.current_solution)
                local_best = self.local_search(temp_solution, max_local_search_tries)
                local_obj = self.evaluate_solution(local_best)          # Full objective
                local_base_obj = self.base_objective(local_best)       # Base objective

                # Check improvement based on full objective
                if self.maximize:
                    solution_is_better = local_obj > self.best_objective
                else:
                    solution_is_better = local_obj < self.best_objective

                # Update if better
                if solution_is_better:
                    self.best_solution = local_best
                    self.best_objective = local_obj
                    self.best_base_objective = local_base_obj
                    self.current_solution = local_best
                    iter_no_improve = 0
                else:
                    iter_no_improve += 1

                # Record iteration stats with both objectives
                iter_end_time = time.time()
                iter_time = iter_end_time - iter_start_time
                cumulative_time = iter_end_time - total_start_time
                
                trajectory['iteration'].append(iters + 1)
                trajectory['current_objective'].append(local_obj)
                trajectory['current_base_objective'].append(local_base_obj)
                trajectory['best_objective'].append(self.best_objective)
                trajectory['best_base_objective'].append(self.best_base_objective)
                trajectory['iteration_time'].append(iter_time)
                trajectory['cumulative_time'].append(cumulative_time)
                trajectory['improvement_flag'].append(solution_is_better)
                
                # Update progress bar with both objectives
                pbar.set_postfix({
                    'Obj': f'{local_obj:.2f}',
                    'Best': f'{self.best_objective:.2f}',
                    'Base': f'{local_base_obj:.2f}',
                    'Time': f'{iter_time:.2f}s'
                })
                pbar.update(1)
                iters += 1
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Print final statistics
        print(f'\nOptimization complete in {total_time:.2f} seconds.')
        print(f'Final objective value: {self.best_objective}')
        print(f'Final base objective value: {self.best_base_objective}')
        print(f'Number of improvements: {sum(trajectory["improvement_flag"])}')
        
        # Convert trajectory to DataFrame
        trajectory_df = pd.DataFrame(trajectory)

        if return_groups_only:
            return self.get_groups(self.best_solution), trajectory_df
        return self.get_groups(self.best_solution), (self.best_solution, self.best_base_objective), trajectory_df

class ExactOptimizer(GroupOptimizer):
    """
    Exact solver for the group optimization problem through explicit enumeration.
    Enumerates all possible symmetric adjacency matrices with unit diagonal entries.
    """
    def __init__(self, q, s, min_group_size, max_group_size, lower_balance, seed=42, maximize=True):
        """
        Initialize ExactOptimizer with same parameters as GroupOptimizer.
        
        Args:
            q (np.array): (N,2) array of student coordinates
            s (np.array): (N,m) array of sensitive attributes
            min_group_size (int): Minimum group size
            max_group_size (int): Maximum group size
            lower_balance (float): Minimum balance ratio required
            seed (int): Random seed
            maximize (bool): If True, maximize objective; if False, minimize
        """
        super().__init__(q, s, min_group_size, max_group_size, lower_balance, seed, maximize)
        
        # Calculate total number of possible matrices for progress reporting
        n_free_entries = (self.N * (self.N-1)) // 2
        self.total_possible_matrices = 2**n_free_entries
        print(f"Problem size: {self.N} students, {self.total_possible_matrices} possible matrices")
        
    def optimize(self, max_time=None, return_groups_only=True, **kwargs):
        """
        Find optimal solution by enumerating all possible symmetric matrices.
        
        Args:
            max_time (float, optional): Maximum runtime in seconds
            
        Returns:
            tuple: (optimal_solution, optimal_objective)
        """
        start_time = time.time()
        
        # Only need to enumerate upper triangle (excluding diagonal)
        n_free_entries = (self.N * (self.N-1)) // 2
        
        best_solution = None
        best_obj = float('-inf') if self.maximize else float('inf')
        
        print(f"Searching {self.total_possible_matrices} possible matrices...")
        
        for i in tqdm(range(self.total_possible_matrices)):
            # Convert integer to binary sequence for upper triangle
            bin_seq = format(i, f'0{n_free_entries}b')
            
            # Create symmetric matrix with this configuration
            matrix = np.zeros((self.N, self.N), dtype=int)
            np.fill_diagonal(matrix, 1)  # Set diagonal to 1
            
            # Fill upper triangle
            idx = 0
            for row in range(self.N):
                for col in range(row + 1, self.N):
                    matrix[row, col] = int(bin_seq[idx])
                    matrix[col, row] = matrix[row, col]  # Symmetry
                    idx += 1
            
            # Check feasibility
            if not self.check_feasibility(matrix):
                continue
                
            # Evaluate objective
            obj = self.base_objective(matrix)
            
            # Update best if better
            if ((self.maximize and obj > best_obj) or 
                (not self.maximize and obj < best_obj)):
                best_solution = matrix.copy()
                best_obj = obj
            
            # Check time limit
            if max_time and (time.time() - start_time) > max_time:
                print(f"\nStopping: Time limit {max_time}s exceeded")
                break
        
        if best_solution is None:
            raise RuntimeError("No feasible solution found")
            
        runtime = time.time() - start_time
        print(f"\nOptimal solution found in {runtime:.2f} seconds")
        print(f"Objective value: {best_obj}")
        
        if return_groups_only:
            return self.get_groups(best_solution), None
        return self.get_groups(best_solution), (best_solution, best_obj), None