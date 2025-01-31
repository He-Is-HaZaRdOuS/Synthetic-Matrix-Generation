from dynamic_matrix_expansion import load_matrix, expand_matrix, display_matrices
from compute_loss import compute_matrix_properties, compute_property_loss_minmax
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import scipy.io
import numpy as np
import os

from compute_loss import (
    compute_matrix_properties, 
    perturb_matrix, 
    weights  # We'll override the original compute_property_loss with a new approach
)

def compute_scaling_params(list_of_prop_dicts, prop_names):
    """
    Given a list of property dictionaries and a list of property names,
    returns a dict with { prop_name: {"min": val, "max": val} }.
    We only consider numeric, non-None values.
    """
    scaling_dict = {}
    
    for prop in prop_names:
        all_vals = []
        for props in list_of_prop_dicts:
            val = props.get(prop, None)
            # We'll only add numeric, non-None values
            if val is not None and isinstance(val, (int, float, np.number)):
                all_vals.append(val)
        
        if len(all_vals) == 0:
            # No valid numeric data for this property
            scaling_dict[prop] = {"min": None, "max": None}
        else:
            scaling_dict[prop] = {
                "min": float(min(all_vals)),
                "max": float(max(all_vals))
            }
    return scaling_dict

def compute_property_loss_minmax(original_props: dict, 
                                 new_props: dict, 
                                 weights: dict, 
                                 scaling_dict: dict):
    loss = 0.0
    epsilon = 1e-12

    for prop_name, w in weights.items():
        orig_val = original_props.get(prop_name, None)
        new_val  = new_props.get(prop_name, None)

        # If either is None, skip
        if orig_val is None or new_val is None:
            continue

        # If either is NaN or Inf, skip or clamp
        if not np.isfinite(orig_val) or not np.isfinite(new_val):
            # Option 1: skip
            continue

        smin = scaling_dict[prop_name]["min"]
        smax = scaling_dict[prop_name]["max"]

        if (smin is not None) and (smax is not None) and (smax != smin):
            scaled_orig = (orig_val - smin) / (smax - smin + epsilon)
            scaled_new  = (new_val  - smin) / (smax - smin + epsilon)
            # If these are nan for any reason, skip
            if not np.isfinite(scaled_orig) or not np.isfinite(scaled_new):
                continue
            
            diff = abs(scaled_orig - scaled_new)
            loss += w * diff
        else:
            # If there's no variation or bounds are None, diff is 0
            loss += w * 0.0
    
    return loss

def get_desired_informations():
    """Get desired information from the user."""
    try:
        rows = int(input("Enter desired number of rows: "))
        cols = int(input("Enter desired number of columns: "))
        additional_density = int(input("Enter desired additional density: "))
        num_matrices = int(input("Enter the number of matrices you want to generate: "))
        return rows, cols, additional_density, num_matrices
    except ValueError:
        print("Invalid input. Please enter integers.")
        return None, None, None, None

def generate_multiple_matrices(original_matrix, desired_rows, desired_cols, desired_density, desired_num):
    original_props = compute_matrix_properties(original_matrix)

    generated_matrices = []
    all_props = [original_props]  # We'll store original + each generated matrix's props
    
    # 1) Generate matrices and collect property dictionaries
    for i in range(desired_num):
        print(f"Generating matrix {i+1}/{desired_num}...")

        expanded_matrix = expand_matrix(original_matrix, desired_rows, desired_cols, desired_density)
        generated_matrices.append(expanded_matrix)

        # Compute new matrix props
        new_props = compute_matrix_properties(expanded_matrix)
        all_props.append(new_props)
    
    # 2) Compute scaling dict for all relevant properties
    #    (We assume we want to scale everything we have a weight for)
    prop_names = list(weights.keys())
    scaling_dict = compute_scaling_params(all_props, prop_names)

    # 3) Now compute scaled losses
    loss_values = []
    for i, expanded_matrix in enumerate(generated_matrices, start=1):
        new_props = compute_matrix_properties(expanded_matrix)
        
        # Use the scaled loss
        loss_val = compute_property_loss_minmax(original_props, new_props, weights, scaling_dict)
        loss_values.append(loss_val)

        print(f"Loss for matrix {i} = {loss_val:.3f}")

        # Save the generated matrix
        save_path = f"{output_directory}/expanded_matrix_{i}.mtx"
        scipy.io.mmwrite(save_path, csr_matrix(expanded_matrix))
        print(f"Matrix {i} saved to {save_path}")

    # Summarize best matrix
    best_idx = np.argmin(loss_values)
    print(f"\nBest matrix is matrix {best_idx+1} with loss = {loss_values[best_idx]:.3f}")
    print(f"All {desired_num} matrices have been generated.")

    return generated_matrices, loss_values

"""
def optimize_multiple_matrices(original_matrix,
                               init_matrices,  # list of generated matrices
                               weights,
                               max_iters=50):
    
    import copy

    # Copy the initial solutions
    matrices = [copy.deepcopy(mat) for mat in init_matrices]

    # Precompute original properties once
    orig_props = compute_matrix_properties(original_matrix)
    
    # We'll gather all props for scaling
    all_props = [orig_props]
    for mat in matrices:
        all_props.append(compute_matrix_properties(mat))

    # Build scaling dict for all properties
    prop_names = list(weights.keys())
    scaling_dict = compute_scaling_params(all_props, prop_names)

    # Helper to compute total scaled loss
    def total_loss(matrices_list):
        loss_sum = 0.0
        for mat in matrices_list:
            mat_props = compute_matrix_properties(mat)
            loss_sum += compute_property_loss_minmax(orig_props, mat_props, weights, scaling_dict)
        return loss_sum
    
    current_loss = total_loss(matrices)

    for iteration in range(max_iters):
        k = np.random.randint(len(matrices))
        
        # Keep a copy of the old matrix
        old_matrix = copy.deepcopy(matrices[k])
        old_loss   = current_loss
        
        # Perturb the chosen matrix
        csr_mat = csr_matrix(matrices[k])
        perturbed_matrix = perturb_matrix(csr_mat)
        perturbed_matrix = csr_mat.toarray()
        matrices[k] = perturbed_matrix
        
        # Recompute total loss
        new_loss = total_loss(matrices)
        
        if new_loss > old_loss:
            # revert if no improvement
            matrices[k] = old_matrix
        else:
            current_loss = new_loss
        
        # Debugging or progress update
        if (iteration+1) % 100 == 0:
            print(f"Iter {iteration+1}, current loss = {current_loss:.3f}")
    
    return matrices
"""

def random_perturb_matrix(csr_mat, step=1.0):
    """
    Randomly changes some of the non-zero entries of a CSR matrix.
    For each selected entry, we add or subtract a small random amount.
    """
    data = csr_mat.data.copy()
    
    # Randomly pick ~20% (for example) of non-zero entries to perturb
    num_nonzeros = len(data)
    if num_nonzeros == 0:
        return csr_mat  # Nothing to perturb
    
    k = max(1, num_nonzeros // 5)  # how many entries to perturb
    indices_to_perturb = np.random.choice(num_nonzeros, size=k, replace=False)
    
    for idx in indices_to_perturb:
        sign = np.random.choice([-1, 1])
        change = sign * np.random.uniform(0, step)
        data[idx] += change
    
    new_csr = csr_matrix((data, csr_mat.indices, csr_mat.indptr), shape=csr_mat.shape)
    return new_csr


def optimize_multiple_matrices(original_matrix,
                               init_matrices,   # list of generated matrices (NumPy arrays)
                               weights,
                               scaling_dict,    # min-max scaling info for compute_property_loss_minmax
                               max_iters=1000,
                               step=1.0):
    """
    Minimizes the sum of property-based losses for all matrices 
    using a local search approach with random perturbation.
    
    :param original_matrix: The reference matrix (NumPy array)
    :param init_matrices: List of NumPy arrays (expanded / generated matrices)
    :param weights: Dictionary of property weights
    :param scaling_dict: {prop_name: {'min': val, 'max': val}}, used in loss scaling
    :param max_iters: Number of iterations for local search
    :param step: Magnitude of perturbation
    :return: (optimized_matrices, best_loss)
    """

    # 1) Precompute original properties just once
    orig_props = compute_matrix_properties(original_matrix)

    # 2) Define a helper to compute cost of a single matrix
    def single_matrix_loss(mat):
        new_props = compute_matrix_properties(mat)
        return compute_property_loss_minmax(orig_props, new_props, weights, scaling_dict)

    # 3) Define a helper to compute total cost for the entire list
    def total_loss(mats):
        return sum(single_matrix_loss(m) for m in mats)

    # 4) Copy the initial solutions so we don't overwrite them
    best_mats = [m.copy() for m in init_matrices]
    best_score = total_loss(best_mats)

    # 5) Local search loop
    for iteration in range(max_iters):
        # Pick one matrix at random to try to improve
        k = np.random.randint(len(best_mats))
        
        # Store old state
        old_mat = best_mats[k].copy()
        old_score = best_score

        # Convert to CSR and perturb
        csr_mat = csr_matrix(old_mat)
        perturbed_csr = random_perturb_matrix(csr_mat, step=step)
        perturbed_array = perturbed_csr.toarray()
        if len(perturbed_array.shape) != 2:
            print(f"Error: Perturbed matrix has shape {perturbed_array.shape}")

        # Replace the chosen matrix with the perturbed version
        best_mats[k] = perturbed_array

        # Recompute total cost
        new_score = total_loss(best_mats)
        
        if new_score < old_score:
            # Accept improvement
            best_score = new_score
        else:
            # Revert if worse
            best_mats[k] = old_mat

        # Optionally, reduce step size over time (helps fine-tune)
        step *= 0.999  # e.g., slow exponential decay

        # Print progress occasionally
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{max_iters}, current best loss = {best_score:.4f}")

    return best_mats, best_score



# File paths
file_path = "original-matrices/685_bus.mtx"
output_directory = "generated-matrices"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Load the original matrix
from dynamic_matrix_expansion import load_matrix
original_matrix = load_matrix(file_path)

if original_matrix is not None:
    print("Original matrix loaded successfully.")
    print("Shape of original matrix:", original_matrix.shape)
    
    # Get user inputs
    desired_rows, desired_cols, desired_density, num_matrices = get_desired_informations()
    
    if all(v is not None for v in [desired_rows, desired_cols, desired_density, num_matrices]):
        print(f"Desired dimensions: {desired_rows}x{desired_cols}, Density: {desired_density}, Matrices: {num_matrices}")
        
        # Generate the matrices
        generated_matrices, loss_values = generate_multiple_matrices(
            original_matrix, 
            desired_rows, 
            desired_cols, 
            desired_density, 
            num_matrices
        )

        # Precompute original properties once
        orig_props = compute_matrix_properties(original_matrix)
        
        # We'll gather all props for scaling
        all_props = [orig_props]
        for mat in generated_matrices:
            all_props.append(compute_matrix_properties(mat))

        # Build scaling dict for all properties
        prop_names = list(weights.keys())
        scaling_dict = compute_scaling_params(all_props, prop_names)

        sum_loss = 0.0
        for loss in loss_values:
            sum_loss += loss

        print(f"Total loss for all matrices = {sum_loss:.3f}")

        print("Optimizing generated matrices properties ...")
        optimized_matrices, loss_opt = optimize_multiple_matrices(
            original_matrix, 
            generated_matrices, 
            weights,
            scaling_dict=scaling_dict,
            max_iters=3000,
            step=2.0
        )

        # Specify the directory to save matrices
        output_dir = "optimized-matrices"
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist


        # Save each matrix to .mtx format
        for i, matrix in enumerate(optimized_matrices):
            file_path = os.path.join(output_dir, f"matrix_{i+1}.mtx")
            if isinstance(matrix, list):
                matrix = np.array(matrix)

            if len(matrix.shape) != 2:
                print(f"Error: Matrix {i+1} has shape {matrix.shape}")
            mmwrite(file_path, matrix)
            print(f"Saved matrix {i+1} to {file_path}")
       

    else:
        print("Failed to get valid dimensions or inputs.")
else:
    print("Failed to load the matrix.")
