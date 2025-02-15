import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

"""
def load_matrix(file_path):
    #Load a matrix from a .mtx file.
    try:
        matrix = scipy.io.mmread(file_path)
        return matrix.toarray() if hasattr(matrix, "toarray") else matrix
    except Exception as e:
        print("Error loading the matrix:", e)
        return None
"""

def load_matrix(file_path):
    """Load a matrix from a .mtx file as a sparse matrix."""
    try:
        matrix = scipy.io.mmread(file_path)
        return matrix  # return the sparse matrix as is
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
    Returns a CSR matrix (sparse) instead of a dense numpy array.
    """
    # We'll gather row/col/val in lists
    row_list = []
    col_list = []
    val_list = []

    # Identify non-zero positions and values in the original matrix
    # If original_matrix is dense, we can do:
    #   non_zero_positions = np.argwhere(original_matrix != 0)
    #   non_zero_values = original_matrix[original_matrix != 0]
    #
    # But if original_matrix is already sparse, do something else.
    # Let's handle the possibility that it's dense or sparse:
    if isinstance(original_matrix, csr_matrix):
        # Just gather from .data, .indices, .indptr
        data = original_matrix.data
        indices = original_matrix.indices
        indptr = original_matrix.indptr
        num_rows, num_cols = original_matrix.shape
        row_coords = []
        for r in range(num_rows):
            start = indptr[r]
            end = indptr[r+1]
            row_coords.extend([r]*(end-start))
        non_zero_positions = list(zip(row_coords, indices))
        non_zero_values = data
        orig_rows, orig_cols = num_rows, num_cols
    else:
        # Dense case
        nz = np.argwhere(original_matrix != 0)
        non_zero_positions = [tuple(pos) for pos in nz]
        non_zero_values = original_matrix[nz[:,0], nz[:,1]]
        orig_rows, orig_cols = original_matrix.shape

    # We might compute min/max from non_zero_values for random additions
    if len(non_zero_values) > 0:
        min_value = non_zero_values.min()
        max_value = non_zero_values.max()
    else:
        min_value = 0
        max_value = 1

    # Scaling factors for rows/cols
    row_scale = desired_rows / orig_rows
    col_scale = desired_cols / orig_cols

    # Place original nonzero positions
    for i, (row, col) in enumerate(non_zero_positions):
        new_row = int(row * row_scale)
        new_col = int(col * col_scale)
        new_row = min(max(new_row, 0), desired_rows - 1)
        new_col = min(max(new_col, 0), desired_cols - 1)

        row_list.append(new_row)
        col_list.append(new_col)
        val_list.append(non_zero_values[i])

        # Add additional non-zeros around the scaled position
        for _ in range(additional_density):
            jitter_row = np.random.randint(-3, 4)
            jitter_col = np.random.randint(-3, 4)
            jr = min(max(new_row + jitter_row, 0), desired_rows - 1)
            jc = min(max(new_col + jitter_col, 0), desired_cols - 1)
            # Only add if we haven't already placed a value here?
            # For simplicity, let's just append:
            val = np.random.uniform(min_value, max_value)
            row_list.append(jr)
            col_list.append(jc)
            val_list.append(val)

    # Build a COO, then convert to CSR
    expanded_coo = coo_matrix((val_list, (row_list, col_list)),
                              shape=(desired_rows, desired_cols))
    expanded_csr = expanded_coo.tocsr()
    return expanded_csr

def display_matrices(original_matrix, expanded_matrix):
    """
    Display the original and expanded matrices' sparsity patterns
    side by side and show non-zero counts.
    """

    original_nonzeros = 0
    expanded_nonzeros = 0

    # If they're CSR, we can do .count_nonzero()
    if isinstance(original_matrix, csr_matrix):
        original_nonzeros = original_matrix.count_nonzero()
    else:
        original_nonzeros = np.count_nonzero(original_matrix)

    if isinstance(expanded_matrix, csr_matrix):
        expanded_nonzeros = expanded_matrix.count_nonzero()
    else:
        expanded_nonzeros = np.count_nonzero(expanded_matrix)

    print(f"Number of non-zero elements in the original matrix: {original_nonzeros}")
    print(f"Number of non-zero elements in the expanded matrix: {expanded_nonzeros}")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"Original Matrix\nNon-Zeros: {original_nonzeros}")
    # spy can accept a sparse matrix directly
    plt.spy(original_matrix, markersize=1)

    plt.subplot(1, 2, 2)
    plt.title(f"Expanded Matrix\nNon-Zeros: {expanded_nonzeros}")
    plt.spy(expanded_matrix, markersize=1)

    plt.tight_layout()
    plt.show()
