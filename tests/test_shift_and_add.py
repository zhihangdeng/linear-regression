import numpy as np
from src.code.shift_and_add import ShiftAndAdd



# 生成随机数据
A = np.random.rand(12, 4)

code = ShiftAndAdd(n=9, k=6)
partitions = np.array_split(A, 6, axis=0)

print("Original Data Partitions:")
for i, partition in enumerate(partitions):
    print(f"Partition {i}:")
    print(partition)

# 编码
encoded_partitions = code.encode(partitions)

print("\nEncoded Data Partitions:")
for i, partition in enumerate(encoded_partitions):
    print(f"Encoded Partition {i}:")
    print(partition)

# 模拟从 slaves 接收到的结果（假设丢失了分区 0 和 1）
results = {i: encoded_partitions[i] for i in range(9) if i != 0 and i != 2}

decoded_partitions = code.decode(results)

print("\nDecoded Data Partitions:")
for i, partition in enumerate(decoded_partitions):
    print(f"Decoded Partition {i}:")
    print(partition)
