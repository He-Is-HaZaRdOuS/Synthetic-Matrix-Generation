import numpy as np
import os
import multiprocessing
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from dynamic_matrix_expansion import load_matrix, expand_matrix
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys

#sys.stdout = open("output.log", "w", buffering=1)  # Line-buffered output

# Use "spawn" to prevent memory duplication issues
multiprocessing.set_start_method("spawn", force=True)

def expand_matrix_iteration(matrix, iteration, rows, cols, nnz, output_dir, original_name):
    """Expands the matrix for a single iteration and saves the result."""
    try:
        inc_row, inc_col = iteration, iteration  # Number of rows and cols being added
        new_rows = rows + inc_row
        new_cols = cols + inc_col

        # Compute additional nnz
#        nnz_per_row = nnz / rows if rows > 0 else 0
#        nnz_per_col = nnz / cols if cols > 0 else 0
#        additional_nnz = int(inc_row * nnz_per_row + inc_col * nnz_per_col)

        print(f"Expanding matrix {original_name}, iteration {iteration} to size ({new_rows}, {new_cols}) ")
        expanded_matrix = expand_matrix(matrix, new_rows, new_cols, additional_density=0)
        save_path = os.path.join(output_dir, f"{original_name}_expansion_{iteration}.mtx")
        mmwrite(save_path, csr_matrix(expanded_matrix))
        print(f"Saved expanded matrix {original_name}_expansion_{iteration} to {save_path}")

    except Exception as e:
        print(f"Error during iteration {iteration} for matrix {original_name}: {e}")
        raise  # Re-raise the exception to propagate it

def generate_expanded_matrix(matrix_path, num_generated=100, output_dir="generated-matrices", num_cores=4):
    """Loads a matrix and expands it in parallel over iterations."""
    try:
        print(f"Processing matrix: {matrix_path}")

        # Load matrix inside function (avoids holding all matrices in memory)
        matrix = load_matrix(matrix_path)
        if matrix is None:
            print(f"Failed to load matrix: {matrix_path}")
            return

        matrix = csr_matrix(matrix)
        rows, cols = matrix.shape
        nnz = matrix.nnz  # Number of nonzero elements
        original_name = os.path.basename(matrix_path).replace('.mtx', '')

        os.makedirs(output_dir, exist_ok=True)

        # Use ProcessPoolExecutor to parallelize over iterations
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(1, num_generated + 1):  # Start from 1
                # Submit each iteration as a separate task
                futures.append(executor.submit(
                    expand_matrix_iteration,
                    matrix, i, rows, cols, nnz, output_dir, original_name
                ))

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()  # This will raise any exceptions caught by the worker
                except Exception as e:
                    print(f"Error while processing an iteration: {e}")

    except MemoryError:
        print(f"MemoryError: Skipping matrix {matrix_path} due to insufficient memory.")
    except OSError as e:
        if "Cannot allocate memory" in str(e):
            print(f"Memory allocation failed for {matrix_path}, skipping it.")
        else:
            print(f"OS error for {matrix_path}: {e}")
    except Exception as e:
        print(f"Unexpected error for {matrix_path}: {e}")

def main():
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="Expand matrices in parallel using processes")
    parser.add_argument('input_file', type=str, help="Text file containing the list of matrix paths")
    parser.add_argument('--num-cores', type=int, default=4, help="Number of processes to use for parallel processing (default: 4)")

    args = parser.parse_args()

    # Read matrix paths (lazy loading: don't load them here)
    with open(args.input_file, "r") as f:
        matrix_paths = ["/home/yousif/matrices/" + line.strip() + ".mtx" for line in f if line.strip()]

    if matrix_paths:
        for path in matrix_paths:
            generate_expanded_matrix(path, num_cores=args.num_cores)

if __name__ == "__main__":
    main()
