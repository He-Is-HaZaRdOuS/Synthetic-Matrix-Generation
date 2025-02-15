import numpy as np
import copy
from scipy.sparse import csr_matrix

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

def compute_matrix_properties_csr(csr_mat: csr_matrix):
    """
    Computes matrix properties using only the CSR data, indices, and indptr.
    Avoids converting to a dense array.
    """
    props = {}
    num_rows, num_cols = csr_mat.shape
    data = csr_mat.data
    indices = csr_mat.indices
    indptr = csr_mat.indptr

    nnz = len(data)
    total_elements = num_rows * num_cols

    # Basic stats
    props["num_nonzeros"] = nnz
    props["density_percent"] = 100.0 * nnz / total_elements if total_elements > 0 else 0.0

    # Nonzero values stats
    if nnz > 0:
        props["value_min"] = data.min()
        props["value_max"] = data.max()
        props["value_avg"] = data.mean()
        props["value_std"] = data.std(ddof=1) if nnz > 1 else 0.0
    else:
        props["value_min"] = None
        props["value_max"] = None
        props["value_avg"] = None
        props["value_std"] = None

    # Nonzeros per row
    row_nnz = np.diff(indptr)  # length == num_rows
    if num_rows > 0:
        props["nonzeros_per_row_min"] = row_nnz.min()
        props["nonzeros_per_row_max"] = row_nnz.max()
        props["nonzeros_per_row_avg"] = row_nnz.mean()
        props["nonzeros_per_row_std"] = row_nnz.std(ddof=1) if num_rows > 1 else 0.0
    else:
        props["nonzeros_per_row_min"] = None
        props["nonzeros_per_row_max"] = None
        props["nonzeros_per_row_avg"] = None
        props["nonzeros_per_row_std"] = None

    # Nonzeros per col
    if num_cols > 0:
        col_counts = np.bincount(indices, minlength=num_cols)
        props["nonzeros_per_col_min"] = col_counts.min()
        props["nonzeros_per_col_max"] = col_counts.max()
        props["nonzeros_per_col_avg"] = col_counts.mean()
        props["nonzeros_per_col_std"] = col_counts.std(ddof=1) if num_cols > 1 else 0.0
    else:
        props["nonzeros_per_col_min"] = None
        props["nonzeros_per_col_max"] = None
        props["nonzeros_per_col_avg"] = None
        props["nonzeros_per_col_std"] = None

    # Pattern symmetry
    nonzero_positions = []
    for r in range(num_rows):
        start = indptr[r]
        end = indptr[r + 1]
        row_indices = indices[start:end]
        row_col_pairs = zip([r]*len(row_indices), row_indices)
        nonzero_positions.extend(row_col_pairs)

    nonzero_positions = set(nonzero_positions)
    transpose_positions = set((c, r) for (r, c) in nonzero_positions)
    pattern_symmetry = (nonzero_positions == transpose_positions)
    props["pattern_symmetry"] = 1.0 if pattern_symmetry else 0.0

    # For numerical symmetry, you could build a dict {(r, c): val} and compare val vs {(c, r): val2}
    # We'll skip the full check for brevity:
    props["numerical_symmetry"] = 0.0  # placeholder

    # Bandwidth / distance to diagonal
    if nnz > 0:
        all_rows = []
        for r in range(num_rows):
            count = indptr[r+1] - indptr[r]
            if count > 0:
                all_rows.append(np.full(count, r, dtype=int))
        all_rows = np.concatenate(all_rows) if all_rows else np.array([], dtype=int)
        all_cols = indices
        distances = np.abs(all_rows - all_cols)
        props["bandwidth"] = distances.max()
        props["avg_distance_to_diagonal"] = distances.mean()
        props["num_diagonals_with_nonzeros"] = len(np.unique(all_rows - all_cols))
    else:
        props["bandwidth"] = 0
        props["avg_distance_to_diagonal"] = 0
        props["num_diagonals_with_nonzeros"] = 0

    # Structural unsymmetry = difference in pattern
    diff_1 = nonzero_positions - transpose_positions
    diff_2 = transpose_positions - nonzero_positions
    props["num_structurally_unsymmetric_elements"] = len(diff_1.union(diff_2))

    # Norms
    # 1-norm: max column sum
    col_sums = np.zeros(num_cols)
    for r in range(num_rows):
        start = indptr[r]
        end = indptr[r+1]
        for idx in range(start, end):
            c = indices[idx]
            col_sums[c] += abs(data[idx])
    props["norm_1"] = col_sums.max() if nnz > 0 else 0

    # Inf norm: max row sum
    row_sums = np.zeros(num_rows)
    for r in range(num_rows):
        start = indptr[r]
        end = indptr[r+1]
        row_sums[r] = np.sum(np.abs(data[start:end]))
    props["norm_inf"] = row_sums.max() if nnz > 0 else 0

    # Frobenius norm
    props["frobenius_norm"] = np.sqrt(np.sum(data * data))

    # Condition number (skipped or approximate in sparse)
    props["estimated_condition_number"] = None

    # (Optional) row_min_min, row_mean_max, etc. would require row-based pass again or zero handling.
    # We'll skip or set them to None for brevity. If needed, replicate the logic above:
    props["row_min_min"] = None
    props["row_min_max"] = None
    props["row_min_mean"] = None
    props["row_min_std"] = None
    props["row_max_min"] = None
    props["row_max_max"] = None
    props["row_max_mean"] = None
    props["row_max_std"] = None
    props["row_mean_min"] = None
    props["row_mean_max"] = None
    props["row_mean_mean"] = None
    props["row_mean_std"] = None
    # Similarly for col_min_min, etc. (or you can compute them similarly)

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

def cost_function(original_csr: csr_matrix, candidate_csr: csr_matrix, weights, scaling_dict):
    original_props = compute_matrix_properties_csr(original_csr)  # CHANGED
    candidate_props = compute_matrix_properties_csr(candidate_csr)  # CHANGED
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
    Performs a simple local search to minimize the property-based loss, all in CSR form.
    """
    current_csr = csr_matrix(init_matrix)  # Convert the init_matrix to CSR
    best_csr = copy.deepcopy(current_csr)

    def cost(mat_csr):
        return cost_function(csr_matrix(original_matrix), mat_csr, weights, scaling_dict)

    current_loss = cost(best_csr)
    best_loss = current_loss

    for iteration in range(max_iters):
        # Create a perturbed candidate
        candidate_csr = copy.deepcopy(best_csr)
        candidate_csr = perturb_matrix(candidate_csr, step=step)
        candidate_loss = cost(candidate_csr)

        if candidate_loss < best_loss:
            best_csr = candidate_csr
            best_loss = candidate_loss

        step *= 0.999  # reduce step size

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration+1}: Current Best Loss = {best_loss:.4f}")

    return best_csr, best_loss
