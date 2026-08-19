# DAY 18: Math for AI - Linear Algebra with NumPy

import numpy as np

# ===== SCALARS =====
scalar = 5
print(f"Scalar: {scalar}")

# ===== VECTORS =====
vector = np.array([1, 2, 3])
print(f"Vector: {vector}")

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Addition: {v1 + v2}")
print(f"Subtraction: {v1 - v2}")
print(f"Multiplication: {v1 * v2}")
print(f"Dot Product: {np.dot(v1, v2)}")

# ===== MATRICES =====
matrix1 = np.array([[1, 2],
                    [3, 4]])

matrix2 = np.array([[5, 6],
                    [7, 8]])

# Matrix operations
print(f"Matrix Addition:\n {matrix1 + matrix2}")
print(f"Matrix Multiplication:\n {np.dot(matrix1, matrix2)}")
print(f"Transpose:\n {matrix1.T}")

# ===== STATISTICS =====
data = np.array([10, 20, 30, 40, 50])
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Std Dev: {np.std(data)}")
print(f"Variance: {np.var(data)}")

# ===== WHY THIS MATTERS IN AI =====
# In ML, data is represented as matrices
# Neural networks use matrix multiplication
# Weights and inputs are vectors
# This is the foundation of everything!

# Example: simple neural network calculation
inputs = np.array([1, 2, 3])      # input layer
weights = np.array([0.5, 0.3, 0.2])  # weights
bias = 1

output = np.dot(inputs, weights) + bias
print(f"\nNeural Network Output: {output}")