import numpy as np
from src.code.rs import RS

A=np.random.rand(12, 4)

code = RS(n=9, k=6)

partitions = np.array_split(A, 6, axis=0)

print("Original Partitions:")
for i, part in enumerate(partitions):
    print(f"Partition {i}:")
    print(part)
    
encoded_partitions = code.encode(partitions)

print("\nEncoded Partitions:")
for i, part in enumerate(encoded_partitions):
    print(f"Partition {i}:")
    print(part)

# Simulate results from slaves (here we just use the encoded partitions as results)
results = {i: encoded_partitions[i] for i in range(9) if i != 6 and i != 1 and i != 2}  # Simulate that slave 6 and 1 failed

decoded_partitions = code.decode(results)

print("\nDecoded Partitions:")
for i, part in enumerate(decoded_partitions):
    print(f"Partition {i}:")
    print(part)