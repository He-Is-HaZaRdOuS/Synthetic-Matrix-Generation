from scipy.sparse import csr_matrix
import numpy as np
import copy

weights = {
    "bandwidth": 0.001,
    
    # Symmetry (pattern, numerical)
    "pattern_symmetry": 1,
    "numerical_symmetry": 1,
    
    # Nonzeros per row
    "nonzeros_per_row_min": 0.001,
    "nonzeros_per_row_max": 0.001,
    "nonzeros_per_row_avg": 0.001,
    "nonzeros_per_row_std": 0.001,
    
    # Nonzeros per column
    "nonzeros_per_col_min": 0.001,
    "nonzeros_per_col_max": 0.001,
    "nonzeros_per_col_avg": 0.001,
    "nonzeros_per_col_std": 0.001,
    
    # Nonzero values statistics
    "value_min": 1,
    "value_max": 1,
    "value_avg": 1,
    "value_std": 1,
    
    # Row-wise statistics
    "row_min_min": 1,
    "row_min_max": 1,
    "row_min_mean": 1,
    "row_min_std": 1,
    
    "row_max_min": 1,
    "row_max_max": 1,
    "row_max_mean": 1,
    "row_max_std": 1,
    
    "row_mean_min": 1,
    "row_mean_max": 1,
    "row_mean_mean": 1,
    "row_mean_std": 1,
                              
    "row_std_min": 1,
    "row_std_max": 1,
    "row_std_mean": 1,
    "row_std_std": 1,
    
    "row_median_min": 1,
    "row_median_max": 1,
    "row_median_mean": 1,
    "row_median_std": 1,
    
    # Column-wise statistics
    "col_min_min": 0.1,
    "col_min_max": 0.1,
    "col_min_mean": 0.1,
    "col_min_std": 0.1,
    
    "col_max_min": 0.1,
    "col_max_max": 0.1,
    "col_max_mean": 0.,
    "col_max_std": 0.1,
    
    "col_mean_min": 0.1,
    "col_mean_max": 0.1,
    "col_mean_mean": 0.1,
    "col_mean_std": 0.1,
    
    "col_std_min": 0.1,
    "col_std_max": 0.1,
    "col_std_mean": 0.1,
    "col_std_std": 0.1,
    
    "col_median_min": 0.1,
    "col_median_max": 0.1,
    "col_median_mean": 0.1,
    "col_median_std": 0.1,
    
    # Distance to diagonal
    "avg_distance_to_diagonal": 0.001,
    "num_diagonals_with_nonzeros": 0.0001,
    
    # Structural unsymmetry
    "num_structurally_unsymmetric_elements": 0.001,
    
    # Norms
    "norm_1": 0.0001,
    "norm_inf": 0.0001,
    "frobenius_norm": 0.01,
    
    # Condition number
    "estimated_condition_number": 0.001
}

def compute_matrix_properties(matrix: np.ndarray):
    props = {}
    
    # Basic dimensions
    num_rows, num_cols = matrix.shape
    
    # Count nonzero elements
    num_nonzeros = np.count_nonzero(matrix)
    # props["num_nonzeros"] = num_nonzeros
    total_elements = num_rows * num_cols
    density_percent = (num_nonzeros / total_elements * 100.0) if total_elements > 0 else 0.0
    # props["density_percent"] = density_percent
    
    # Symmetry checks
    # Pattern symmetry: Compare sets of nonzero indices
    nonzero_positions = set(zip(*np.nonzero(matrix)))
    transpose_nonzero_positions = set(zip(*np.nonzero(matrix.T)))
    pattern_symmetry = (nonzero_positions == transpose_nonzero_positions)
    props["pattern_symmetry"] = pattern_symmetry
    
    # Numerical symmetry: Check if matrix is approximately equal to its transpose
    numerical_symmetry = np.allclose(matrix, matrix.T, atol=1e-14)
    props["numerical_symmetry"] = numerical_symmetry

    props["pattern_symmetry"] = 1.0 if pattern_symmetry else 0.0
    props["numerical_symmetry"] = 1.0 if numerical_symmetry else 0.0

    
    # Nonzeros per row
    row_nnz = (matrix != 0).sum(axis=1)
    props["nonzeros_per_row_min"] = row_nnz.min() if num_rows > 0 else None
    props["nonzeros_per_row_max"] = row_nnz.max() if num_rows > 0 else None
    props["nonzeros_per_row_avg"] = row_nnz.mean() if num_rows > 0 else None
    props["nonzeros_per_row_std"] = row_nnz.std(ddof=1) if num_rows > 1 else None
    
    # Nonzeros per column
    col_nnz = (matrix != 0).sum(axis=0)
    props["nonzeros_per_col_min"] = col_nnz.min() if num_cols > 0 else None
    props["nonzeros_per_col_max"] = col_nnz.max() if num_cols > 0 else None
    props["nonzeros_per_col_avg"] = col_nnz.mean() if num_cols > 0 else None
    props["nonzeros_per_col_std"] = col_nnz.std(ddof=1) if num_cols > 1 else None
    
    # Nonzero values statistics
    nonzero_values = matrix[matrix != 0]
    if nonzero_values.size > 0:
        props["value_min"] = nonzero_values.min()
        props["value_max"] = nonzero_values.max()
        props["value_avg"] = nonzero_values.mean()
        props["value_std"] = nonzero_values.std(ddof=1) if nonzero_values.size > 1 else 0.0
    else:
        props["value_min"] = None
        props["value_max"] = None
        props["value_avg"] = None
        props["value_std"] = None
    
    # Row-wise statistics
    # Compute per-row min, max, mean, std, median
    if num_rows > 0 and num_cols > 0:
        row_mins = np.min(matrix, axis=1)
        row_maxs = np.max(matrix, axis=1)
        row_means = np.mean(matrix, axis=1)
        row_stds = np.std(matrix, axis=1, ddof=1) if num_cols > 1 else np.zeros(num_rows)
        row_medians = np.median(matrix, axis=1)
        
        # For each of these arrays, compute min, max, mean, std
        # row_min_*
        props["row_min_min"] = row_mins.min()
        props["row_min_max"] = row_mins.max()
        props["row_min_mean"] = row_mins.mean()
        props["row_min_std"] = row_mins.std(ddof=1) if num_rows > 1 else 0.0
        
        # row_max_*
        props["row_max_min"] = row_maxs.min()
        props["row_max_max"] = row_maxs.max()
        props["row_max_mean"] = row_maxs.mean()
        props["row_max_std"] = row_maxs.std(ddof=1) if num_rows > 1 else 0.0
        
        # row_mean_*
        props["row_mean_min"] = row_means.min()
        props["row_mean_max"] = row_means.max()
        props["row_mean_mean"] = row_means.mean()
        props["row_mean_std"] = row_means.std(ddof=1) if num_rows > 1 else 0.0
        
        # row_std_*
        props["row_std_min"] = row_stds.min() if row_stds.size > 0 else None
        props["row_std_max"] = row_stds.max() if row_stds.size > 0 else None
        props["row_std_mean"] = row_stds.mean() if row_stds.size > 0 else None
        props["row_std_std"] = row_stds.std(ddof=1) if num_rows > 1 else 0.0
        
        # row_median_*
        props["row_median_min"] = row_medians.min()
        props["row_median_max"] = row_medians.max()
        props["row_median_mean"] = row_medians.mean()
        props["row_median_std"] = row_medians.std(ddof=1) if num_rows > 1 else 0.0
    else:
        # No rows/columns: set these to None
        for stat in ["row_min","row_max","row_mean","row_std","row_median"]:
            for agg in ["min","max","mean","std"]:
                props[f"{stat}_{agg}"] = None
    
    # Column-wise statistics
    if num_cols > 0 and num_rows > 0:
        col_mins = np.min(matrix, axis=0)
        col_maxs = np.max(matrix, axis=0)
        col_means = np.mean(matrix, axis=0)
        col_stds = np.std(matrix, axis=0, ddof=1) if num_rows > 1 else np.zeros(num_cols)
        col_medians = np.median(matrix, axis=0)
        
        # col_min_*
        props["col_min_min"] = col_mins.min()
        props["col_min_max"] = col_mins.max()
        props["col_min_mean"] = col_mins.mean()
        props["col_min_std"] = col_mins.std(ddof=1) if num_cols > 1 else 0.0
        
        # col_max_*
        props["col_max_min"] = col_maxs.min()
        props["col_max_max"] = col_maxs.max()
        props["col_max_mean"] = col_maxs.mean()
        props["col_max_std"] = col_maxs.std(ddof=1) if num_cols > 1 else 0.0
        
        # col_mean_*
        props["col_mean_min"] = col_means.min()
        props["col_mean_max"] = col_means.max()
        props["col_mean_mean"] = col_means.mean()
        props["col_mean_std"] = col_means.std(ddof=1) if num_cols > 1 else 0.0
        
        # col_std_*
        props["col_std_min"] = col_stds.min() if col_stds.size > 0 else None
        props["col_std_max"] = col_stds.max() if col_stds.size > 0 else None
        props["col_std_mean"] = col_stds.mean() if col_stds.size > 0 else None
        props["col_std_std"] = col_stds.std(ddof=1) if num_cols > 1 else 0.0
        
        # col_median_*
        props["col_median_min"] = col_medians.min()
        props["col_median_max"] = col_medians.max()
        props["col_median_mean"] = col_medians.mean()
        props["col_median_std"] = col_medians.std(ddof=1) if num_cols > 1 else 0.0
    else:
        for stat in ["col_min","col_max","col_mean","col_std","col_median"]:
            for agg in ["min","max","mean","std"]:
                props[f"{stat}_{agg}"] = None
    
    # Distance to Diagonal
    # Distances for nonzero elements
    if num_nonzeros > 0:
        i_coords, j_coords = np.nonzero(matrix)
        distances = np.abs(i_coords - j_coords)
        props["avg_distance_to_diagonal"] = distances.mean()
        props["num_diagonals_with_nonzeros"] = len(np.unique(i_coords - j_coords))
        # bandwidth was computed as max(|i-j|)
        bandwidth = distances.max() if distances.size > 0 else 0
    else:
        props["avg_distance_to_diagonal"] = None
        props["num_diagonals_with_nonzeros"] = 0
        bandwidth = 0
    props["bandwidth"] = bandwidth
    
    # Structural Unsymmetry
    # Count how many entries do not have a symmetric counterpart
    # We define structural unsymmetry as the number of indices in A but not in A^T, plus vice versa.
    # Actually, since pattern_symmetry checks equality, we can measure the difference:
    diff_1 = nonzero_positions - transpose_nonzero_positions
    diff_2 = transpose_nonzero_positions - nonzero_positions
    num_structurally_unsymmetric_elements = len(diff_1.union(diff_2))
    props["num_structurally_unsymmetric_elements"] = num_structurally_unsymmetric_elements
    
    # Norms
    # 1-norm: max absolute column sum
    norm_1 = np.linalg.norm(matrix, 1) if total_elements > 0 else None
    props["norm_1"] = norm_1
    
    # Infinity norm: max absolute row sum
    # numpy doesn't have a direct norm_inf for 2D. 
    # Infinity norm can be computed as max over rows of sum of absolute values:
    if total_elements > 0:
        norm_inf = np.max(np.sum(np.abs(matrix), axis=1))
    else:
        norm_inf = None
    props["norm_inf"] = norm_inf
    
    # Frobenius norm
    fro_norm = np.linalg.norm(matrix, 'fro')
    props["frobenius_norm"] = fro_norm
    
    # Condition number (1-norm)
    # Can fail if matrix is singular. We'll catch and return None if that happens.
    try:
        estimated_condition_number = float(np.linalg.cond(matrix, 1))
    except np.linalg.LinAlgError:
        estimated_condition_number = None
    props["estimated_condition_number"] = estimated_condition_number

    
    return props

def perturb_matrix(csr_mat, step=1):
    data = csr_mat.data
    indices = csr_mat.indices
    indptr = csr_mat.indptr  # row pointers
    
    # Create a mask for the middle elements
    mask = np.ones_like(data, dtype=bool)
    
    for row in range(csr_mat.shape[0]):
        start = indptr[row]
        end = indptr[row+1]
        length = end - start
        
        if length > 1:
            mid_pos = start + length // 2
            mask[mid_pos] = False
    
    # Increment all elements except the middle ones
    data[mask] += step
    
    return csr_mat

def compute_scaling_params(list_of_prop_dicts, prop_names):
    """
    Given a list of property dictionaries and a list of property names,
    returns a dict with { prop_name: {"min": val, "max": val} }.
    """
    scaling_dict = {}
    
    for prop in prop_names:
        all_vals = []
        for props in list_of_prop_dicts:
            val = props.get(prop, None)
            # We'll only add numeric, non-None values
            if val is not None and isinstance(val, (int, float)):
                all_vals.append(val)
        
        if len(all_vals) == 0:
            # No valid numeric data for this property
            # We can set min/max to None or some default
            scaling_dict[prop] = {"min": None, "max": None}
        else:
            scaling_dict[prop] = {
                "min": min(all_vals),
                "max": max(all_vals)
            }
    return scaling_dict

def compute_property_loss_minmax(original_props: dict, 
                                 new_props: dict, 
                                 weights: dict, 
                                 scaling_dict: dict):
    """
    Compute property-based loss using min-max scaling for each property.
    scaling_dict is typically what you get from compute_scaling_params.
    """
    loss = 0.0
    epsilon = 1e-12  # to avoid zero division

    for prop_name, w in weights.items():
        # Get original/new values
        orig_val = original_props.get(prop_name, None)
        new_val  = new_props.get(prop_name, None)

        # If either is None, skip or handle them as you see fit
        if orig_val is None or new_val is None:
            # Option 1: Skip it
            # Option 2: Add some penalty for missing
            continue

        # Retrieve min and max from scaling_dict
        smin = scaling_dict[prop_name]["min"]
        smax = scaling_dict[prop_name]["max"]
        
        # If we have valid scaling bounds
        if smin is not None and smax is not None and smax != smin:
            scaled_orig_val = (orig_val - smin) / (smax - smin + epsilon)
            scaled_new_val  = (new_val - smin) / (smax - smin + epsilon)
        else:
            # Fallback: if smax == smin or we have None
            # we could treat everything as zero or skip
            scaled_orig_val = 0.0
            scaled_new_val  = 0.0

        # Compute absolute difference on the scaled values
        diff = np.abs(scaled_orig_val - scaled_new_val)
        loss += w * diff
    
    return loss

def cost_function(original_matrix, candidate_matrix, weights, scaling_dict):
    original_props = compute_matrix_properties(original_matrix)
    candidate_props = compute_matrix_properties(candidate_matrix)
    return compute_property_loss_minmax(original_props, candidate_props, weights, scaling_dict)

def random_perturb_matrix(csr_mat, step=1.0):
    data = csr_mat.data.copy()
    # Randomly pick some entries to change
    indices_to_perturb = np.random.choice(len(data), size=len(data)//5, replace=False)
    
    for idx in indices_to_perturb:
        # Decide increment or decrement
        sign = np.random.choice([-1, 1])
        # Apply random magnitude
        change = sign * np.random.uniform(0, step)
        data[idx] += change

    # Return a new matrix so you don’t modify in place
    new_csr = csr_matrix((data, csr_mat.indices, csr_mat.indptr), shape=csr_mat.shape)
    return new_csr

def local_search_optimization(original_matrix, 
                              init_matrix, 
                              weights, 
                              scaling_dict, 
                              max_iters=1000, 
                              step=1.0):
    """
    Performs a simple local search to minimize the property-based loss.
    """
    # Convert init_matrix to CSR for perturbations
    current_csr = csr_matrix(init_matrix)
    
    # Helper to compute cost
    def cost(mat):
        return cost_function(original_matrix, mat, weights, scaling_dict)
    
    current_loss = cost(current_csr.toarray())
    
    best_csr = copy.deepcopy(current_csr)
    best_loss = current_loss
    
    for iteration in range(max_iters):
        # Create a perturbed candidate
        candidate_csr = random_perturb_matrix(best_csr, step=step)
        candidate_loss = cost(candidate_csr.toarray())
        
        # If it’s better, accept it
        if candidate_loss < best_loss:
            best_csr = candidate_csr
            best_loss = candidate_loss
        
        # Optional: step-size scheduling, e.g. gradually reduce step
        step *= 0.999  # small decay
        
        # Debug/log
        if (iteration+1) % 100 == 0:
            print(f"Iteration {iteration+1}: Current Best Loss = {best_loss:.4f}")
    
    # Return the final best matrix as numpy array
    return best_csr.toarray(), best_loss
