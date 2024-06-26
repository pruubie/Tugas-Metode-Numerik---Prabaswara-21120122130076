import numpy as np

def solve_using_inverse(A, b):
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Testing
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

print("Metode Matriks Balikan:")
print(solve_using_inverse(A, b))