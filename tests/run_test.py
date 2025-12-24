import numpy as np
from src.code.shift_and_add import ShiftAndAdd
from src.code.rs import RS

np.set_printoptions(precision=16, suppress=False)

# 参数设置
num_iterations = 500  # 迭代次数
n, k = 9, 6  # 编码参数

def run_test(algorithm_name, code_class, generate_data_fn):
    """
    运行测试并计算平均 MSE。
    
    :param algorithm_name: 算法名称（如 "ShiftAndAdd" 或 "RS"）
    :param code_class: 编码类（ShiftAndAdd 或 RS）
    :param generate_data_fn: 数据生成函数
    """
    mse_list = []  # 存储每次迭代的 MSE

    for iteration in range(num_iterations):
        print(f"\n[{algorithm_name}] Iteration {iteration + 1}/{num_iterations}")

        # 生成随机数据
        A = generate_data_fn()

        code = code_class(n=n, k=k)
        partitions = np.array_split(A, k, axis=0)

        # 编码
        encoded_partitions = code.encode(partitions)

        # 模拟从 slaves 接收到的结果
        results = {i: encoded_partitions[i] for i in range(n) if i != 0 and i != 2 and i != 7}

        # 解码
        decoded_partitions = code.decode(results)

        # 计算当前迭代的 MSE
        mse = 0
        for original, decoded in zip(partitions, decoded_partitions):
            mse += np.mean((original - decoded) ** 2)
        mse /= len(partitions)

        print(f"[{algorithm_name}] Mean Squared Error (MSE) for iteration {iteration + 1}: {mse}")
        mse_list.append(mse)

    # 计算多次迭代的平均 MSE
    average_mse = np.mean(mse_list)
    print(f"\n[{algorithm_name}] Average Mean Squared Error (MSE) over {num_iterations} iterations: {average_mse}")
    return average_mse

# 数据生成函数
def generate_data_shift_and_add():
    A = np.random.rand(12, 4)
    A = np.hstack([A, -np.sum(A, axis=1, keepdims=True)])  # 添加校验列
    return A

def generate_data_rs():
    return np.random.rand(12, 4)

# 运行测试
average_mse_shift_and_add = run_test("ShiftAndAdd", ShiftAndAdd, generate_data_shift_and_add)
average_mse_rs = run_test("RS", RS, generate_data_rs)

print("\nFinal Results:")
print(f"ShiftAndAdd Average MSE: {average_mse_shift_and_add}")
print(f"RS Average MSE: {average_mse_rs}")