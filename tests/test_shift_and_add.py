import numpy as np
from src.code.shift_and_add import ShiftAndAdd

A=np.random.randint(0, 10, (18, 5))

code = ShiftAndAdd(n=9, k=6)

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
results = {i: encoded_partitions[i] for i in range(9) if i != 0 and i != 2}

print("\nResults received from slaves:")
for i in results:
    print(f"Slave {i}:")
    print(results[i])

decoded_partitions = code.decode(results)

print("\nDecoded Partitions:")
for i, part in enumerate(decoded_partitions):
    print(f"Partition {i}:")
    print(part)