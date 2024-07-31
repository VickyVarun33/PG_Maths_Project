import numpy as np

def svd(A, tol=1e-10):
    # Step 1
    cov_matrix = np.dot(A.T, A)
    print("\nStep 1: Compute the covariance matrix:\n", cov_matrix)

    # Step 2
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    print("\nStep 2: Compute eigenvalues and eigenvectors:\nEigenvalues:\n", eigvals, "\nEigenvectors:\n", eigvecs)

    # Step 3
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    print("\nStep 3: Sort eigenvalues and eigenvectors:\nSorted Eigenvalues:\n", eigvals, "\nSorted Eigenvectors:\n", eigvecs)

    # Step 4
    singular_values = np.sqrt(np.abs(eigvals))
    print("\nStep 4: Compute singular values:\n", singular_values)

    # Step 5
    U = eigvecs
    print("\nStep 5: Compute the matrix U (left singular vectors):\n", U)

    # Step 6
    Vt = np.dot(np.linalg.inv(np.diag(singular_values)), np.dot(U.T, A.T))
    print("\nStep 6: Compute the matrix Vt (transpose of right singular vectors):\n", Vt)

    # Step 7
    mask = singular_values > tol
    singular_values = singular_values[mask]
    U = U[:, mask]
    Vt = Vt[mask, :]
    print("\nStep 7: Truncate singular values and vectors based on tolerance:\n", singular_values, U, Vt)

    # Step 8
    S = np.diag(singular_values)
    print("\nStep 8: Construct the diagonal matrix S:\n", S)

    return U, S, Vt

def input_matrix():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    A = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            A[i, j] = float(input(f"Enter the element at position ({i+1}, {j+1}): "))

    return A

# Getting user input for the matrix
A = input_matrix()

# Perform SVD
U, S, Vt = svd(A)

# Reconstruct the matrix using the truncated SVD
A_reconstructed = np.dot(U, np.dot(S, Vt))

print("\nOriginal matrix:")
print(A)
print("\nReconstructed matrix using SVD:")
print(A_reconstructed)
