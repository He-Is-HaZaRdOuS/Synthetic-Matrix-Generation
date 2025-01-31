import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_matrix(file_path):
    """Load a matrix from a .mtx file."""
    try:
        matrix = scipy.io.mmread(file_path)
        return matrix.toarray() if hasattr(matrix, "toarray") else matrix
    except Exception as e:
        print("Error loading the matrix:", e)
        return None

def scale_original_matrix(original_matrix, desired_rows, desired_cols):
    """
    Scale the original matrix to the desired dimensions similarly to 'expand_matrix'
    but without the jitter or extra density. This produces a 'reference scaled matrix'.
    """
    scaled_matrix = np.zeros((desired_rows, desired_cols))
    non_zero_positions = np.argwhere(original_matrix != 0)
    non_zero_values = original_matrix[original_matrix != 0]

    row_scale = desired_rows / original_matrix.shape[0]
    col_scale = desired_cols / original_matrix.shape[1]

    for i, (row, col) in enumerate(non_zero_positions):
        new_row = int(row * row_scale)
        new_col = int(col * col_scale)
        
        # Clamp to bounds
        new_row = min(max(new_row, 0), desired_rows - 1)
        new_col = min(max(new_col, 0), desired_cols - 1)
        
        scaled_matrix[new_row, new_col] = non_zero_values[i]
    
    return scaled_matrix

def get_desired_dimensions():
    """Get desired dimensions from the user."""
    try:
        rows = int(input("Enter desired number of rows: "))
        cols = int(input("Enter desired number of columns: "))
        additional_density = int(input("Enter desired additional density: "))
        return rows, cols, additional_density
    except ValueError:
        print("Invalid input. Please enter integers.")
        return None, None, None

def expand_matrix(original_matrix, desired_rows, desired_cols, additional_density):
    """
    Expand the matrix by scaling original non-zero positions proportionally,
    preserving the pattern, and increasing non-zero count with controlled density.
    
    Parameters:
    - original_matrix: The input sparse matrix to expand.
    - desired_rows: Number of rows in the expanded matrix.
    - desired_cols: Number of columns in the expanded matrix.
    - additional_density: Number of new non-zeros to add around each scaled position.
    
    Returns:
    - expanded_matrix: The expanded matrix with increased non-zero count and preserved pattern.
    """
    # Create an empty matrix with desired dimensions
    expanded_matrix = np.zeros((desired_rows, desired_cols))
    
    # Identify non-zero positions and values in the original matrix
    non_zero_positions = np.argwhere(original_matrix != 0)
    non_zero_values = original_matrix[original_matrix != 0]
    
    # Min and max values of original non-zeros
    min_value = non_zero_values.min()
    max_value = non_zero_values.max()
    
    # Scaling factors for rows and columns
    row_scale = desired_rows / original_matrix.shape[0]
    col_scale = desired_cols / original_matrix.shape[1]
    
    for i, (row, col) in enumerate(non_zero_positions):
        # Scale positions to the new dimensions
        new_row = int(row * row_scale)
        new_col = int(col * col_scale)
        
        # Ensure positions stay within bounds and place the original non-zero
        new_row = min(max(new_row, 0), desired_rows - 1)
        new_col = min(max(new_col, 0), desired_cols - 1)
        expanded_matrix[new_row, new_col] = non_zero_values[i]
        
        # Add additional non-zeros around the scaled position
        for _ in range(additional_density):
            jitter_row = np.random.randint(-3, 4)
            jitter_col = np.random.randint(-3, 4)
            jittered_row = min(max(new_row + jitter_row, 0), desired_rows - 1)
            jittered_col = min(max(new_col + jitter_col, 0), desired_cols - 1)
            
            # Assign a random value between min and max to the new position
            if expanded_matrix[jittered_row, jittered_col] == 0:  # Avoid overwriting
                expanded_matrix[jittered_row, jittered_col] = np.random.uniform(min_value, max_value)
    
    return expanded_matrix

def display_matrices(original_matrix, expanded_matrix):
    """Display the original and expanded matrices' sparsity patterns side by side and show non-zero counts."""
    original_nonzeros = np.count_nonzero(original_matrix)
    expanded_nonzeros = np.count_nonzero(expanded_matrix)
    
    print(f"Number of non-zero elements in the original matrix: {original_nonzeros}")
    print(f"Number of non-zero elements in the expanded matrix: {expanded_nonzeros}")
    
    plt.figure(figsize=(12, 6))
    
    # Display the sparsity pattern of the original matrix
    plt.subplot(1, 2, 1)
    plt.title(f"Original Matrix\nNon-Zeros: {original_nonzeros}")
    plt.spy(original_matrix, markersize=1)
    
    # Display the sparsity pattern of the expanded matrix
    plt.subplot(1, 2, 2)
    plt.title(f"Expanded Matrix\nNon-Zeros: {expanded_nonzeros}")
    plt.spy(expanded_matrix, markersize=1)
    
    plt.tight_layout()
    plt.show()