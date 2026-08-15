# DAY 13: NumPy - Arrays and Operations

import numpy as np

# Creating arrays
arr1 = np.array([1,2,3,4,5])
arr2 = np.zeros((3,3))
arr3 = np.ones((2,4))
arr4 = np.arange(0,10,2)

# 2D Matrix
matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(matrix.shape)    # (3,3)
print(matrix[0])       # [1 2 3]
print(matrix[1][2])    # 6

# Array operations
arr1 = np.arange(99)
print(arr1.nbytes)     # 792
print(arr1.argmax())   # 98
print(arr1.argmin())   # 0
print(arr1.ravel())    # flatten

# Square root
arr1 = np.sqrt(arr1)

# Where condition
arr1 = np.where(arr1 > 5)

# Count nonzero
arr1 = np.count_nonzero(arr1)
print(arr1)            # 73