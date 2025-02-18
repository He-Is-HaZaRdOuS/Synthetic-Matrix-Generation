# Synthetic-Matrix-Generation

Next-gen matrix generation with optimization.

## Disclaimer  
This repository is a **fork** of the original project: [Matrix-Generation](https://github.com/farukaplan/Matrix-Generation) by [farukaplan](https://github.com/farukaplan). All credits for the original implementation go to the original author. This fork may contain modifications, improvements, or additional functionality.  

## Overview
This project focuses on generating and optimizing matrices for various applications. It includes scripts for matrix generation and loss minimization to enhance matrix properties.

## Repository Structure
`generate_matrices.py`: Script to generate multiple matrices and optimize them by minimising the loss.

`dynamic_matrix_expansion.py`: Module for generating a matrix initially.

`compute_loss.py`: Functionality to compute loss metrics for matrix optimization.

`matrices_list.txt`: Contains a list of desired matrices to expand.

`matrix_properties_list.txt`: Contains a list of desired matrix properties for reference.

`original-matrices/`: Directory containing the original, unmodified matrices.

`generated-matrices/`: Directory where newly generated matrices are stored.

`optimized-matrices/`: Directory for matrices that have undergone optimization processes.

## Getting Started
1) Clone the Repository:
```
git clone https://github.com/He-Is-HaZaRdOuS/Synthetic-Matrix-Generation
cd Matrix-Generation
```
2) Install Dependencies: Ensure you have Python installed. Install any necessary packages using:
```
pip install -r requirements.txt
```
3) Generate Matrices: Run the matrix generation script:
```
python3 generate_matrices.py
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.









