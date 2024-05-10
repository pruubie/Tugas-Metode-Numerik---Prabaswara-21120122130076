import numpy as np
import scipy.linalg

def solve_using_lu_gauss(A, b):
    P, L, U = scipy.linalg.lu(A)
    y = np.linalg.solve(L, P @ b)
    x = np.linalg.solve(U, y)
    return x

# Testing
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

print("\nMetode Dekomposisi LU Gauss:")
print(solve_using_lu_gauss(A, b))