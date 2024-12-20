def find_sum_combinations(x_lower, x_higher, target_sum, current_combo=None, all_combos=None):
    """Find all combinations of numbers in range x_lower to x_higher that sum to target_sum"""
    if current_combo is None:
        current_combo = []
    if all_combos is None:
        all_combos = []
    
    # base cases
    current_sum = sum(current_combo)
    if current_sum == target_sum:
        all_combos.append(current_combo[:])
        return
    if current_sum > target_sum:
        return
    
    start = x_lower if not current_combo else max(x_lower, current_combo[-1])
    for num in range(start, x_higher + 1):
        current_combo.append(num)
        find_sum_combinations(x_lower, x_higher, target_sum, current_combo, all_combos)
        current_combo.pop()
    return all_combos