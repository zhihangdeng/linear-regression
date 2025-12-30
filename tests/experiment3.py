import numpy as np
import pandas as pd
from src.code.shift_and_add import ShiftAndAdd
n=9
k=6
A = np.random.rand(19,4)
rows, cols = A.shape
if rows % k != 0:
    rows_to_add = k - (rows % k)
    A = np.vstack([A, np.zeros((rows_to_add, cols))])
rows, cols = A.shape
if cols % k != 0:
    cols_to_add = k - (cols % k)
    A = np.hstack([A, np.zeros((rows, cols_to_add))])
rows, cols = A.shape

code = ShiftAndAdd(n=9, k=6)

partitions = np.array_split(A, 6, axis=0)
print("Original Partitions:")
for i, partition in enumerate(partitions):
    print(f"Partition {i}:")
    print(partition)

encoded_partitions = code.encode(partitions)
print("Encoded Partitions:")
for i, partition in enumerate(encoded_partitions):
    print(f"Partition {i}:")
    print(partition)

# matrix multiplication test
x = np.array([[1],[2],[3],[4],[5],[6]])
print("Vector x:")
print(x)

results = {}
for i in range(n):
    Cx_i = encoded_partitions[i] @ x
    print(f"C_{i} * x:")
    print(Cx_i)

results = {i: Cx_i for i, Cx_i in enumerate([encoded_partitions[i] @ x for i in range(n)]) if i != 0 and i != 2}
print("Received Results (excluding partition 1):")
for i in results:
    print(f"Result from Partition {i}:")
    print(results[i])

decoded_partitions = code.decode(results)
decoded_Ax = np.vstack(decoded_partitions)
print("Decoded A * x:")
print(decoded_Ax)

Ax = A @ x
print("Original A * x:")
print(Ax)