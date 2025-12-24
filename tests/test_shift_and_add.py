import numpy as np
from src.code.shift_and_add import ShiftAndAdd

np.set_printoptions(precision=16, suppress=False)

# 参数设置
num_iterations = 500  # 迭代次数
mse_list = []  # 存储每次迭代的 MSE

for iteration in range(num_iterations):
    print(f"\nIteration {iteration + 1}/{num_iterations}")

    # 生成随机数据
    A = np.random.rand(12, 4)
    A = np.hstack([A, -np.sum(A, axis=1, keepdims=True)])

    code = ShiftAndAdd(n=9, k=6)
    partitions = np.array_split(A, 6, axis=0)

    # 编码
    encoded_partitions = code.encode(partitions)

    # 模拟从 slaves 接收到的结果
    results = {i: encoded_partitions[i] for i in range(9) if i != 0 and i != 2 and i != 7}

    # 解码
    decoded_partitions = code.decode(results)

    # 计算当前迭代的 MSE
    mse = 0
    for original, decoded in zip(partitions, decoded_partitions):
        mse += np.mean((original - decoded) ** 2)
    mse /= len(partitions)

    print(f"Mean Squared Error (MSE) for iteration {iteration + 1}: {mse}")
    mse_list.append(mse)

# 计算多次迭代的平均 MSE
average_mse = np.mean(mse_list)
print(f"\nAverage Mean Squared Error (MSE) over {num_iterations} iterations: {average_mse}")