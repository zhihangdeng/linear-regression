import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.code.shift_and_add import ShiftAndAdd
from src.code.rs import RS

np.set_printoptions(precision=16, suppress=False)

# 参数设置
num_iterations = 100  # 每个方差值的迭代次数
n, k = 9, 6  # 编码参数
variances = [10**i for i in range(0, 21)]  # 方差从 1 到 10^20

def run_test(algorithm_name, code_class, generate_data_fn):
    relative_error_list = []  # 存储每次迭代的相对误差

    for iteration in range(num_iterations):
        # 生成随机数据
        A, x = generate_data_fn()

        code = code_class(n=n, k=k)
        partitions = np.array_split(A, k, axis=0)

        # 编码
        encoded_partitions = code.encode(partitions)

        # 模拟从 slaves 接收到的结果
        results = {i: Cx_i for i, Cx_i in enumerate([encoded_partitions[i] @ x for i in range(n)]) if i != 0 and i != 2 and i != 7}

        # 解码
        decoded_partitions = code.decode(results)
        decoded_Ax = np.vstack(decoded_partitions)

        original_Ax = A @ x

        # 计算当前实验的相对误差
        relative_error = np.linalg.norm(original_Ax - decoded_Ax) / np.linalg.norm(original_Ax)
        relative_error_list.append(relative_error)

    # 计算 50 次独立实验的平均相对误差
    average_relative_error = np.mean(relative_error_list)
    return average_relative_error

    # 计算多次迭代的平均 MSE
    average_mse = np.mean(mse_list)
    return average_relative_error

# 数据生成函数
def generate_data_shift_and_add(mean, std_dev):
    A = np.random.normal(loc=mean, scale=std_dev, size=(960, 100))  # 使用正态分布生成数据
    x = np.random.normal(loc=mean, scale=std_dev, size=(100, 1))
    return A, x

def generate_data_rs(mean, std_dev):
    A = np.random.normal(loc=mean, scale=std_dev, size=(960, 100))  # 使用正态分布生成数据
    x = np.random.normal(loc=mean, scale=std_dev, size=(100, 1))
    return A, x

# 记录结果
shift_and_add_mse = []
rs_mse = []

for variance in variances:
    std_dev = np.sqrt(float(variance))  # 将 variance 转换为 float
    print(f"Running tests for variance: {variance}")

    # ShiftAndAdd
    average_mse_shift_and_add = run_test(
        "ShiftAndAdd",
        ShiftAndAdd,
        lambda: generate_data_shift_and_add(0, std_dev)
    )
    shift_and_add_mse.append(average_mse_shift_and_add)

    # RS
    average_mse_rs = run_test(
        "RS",
        RS,
        lambda: generate_data_rs(0, std_dev)
    )
    rs_mse.append(average_mse_rs)

# 保存数据到 Excel
data = {
    "Noise variance": variances,
    "Shift and Add MSE": shift_and_add_mse,
    "Reed Solomon MSE": rs_mse
}
df = pd.DataFrame(data)
df.to_excel("experiment_results.xlsx", index=False)
print("Results saved to 'experiment_results.xlsx'.")

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(variances, shift_and_add_mse, label="Shift and add", marker='o')
plt.plot(variances, rs_mse, label="Reed solomon", marker='x')
plt.xscale('log')  # 横坐标使用对数刻度
plt.yscale('log')  # 纵坐标使用对数刻度
plt.xlabel("Noise variance")
plt.ylabel("Mean squared error")

# 获取当前图例的 handles 和 labels
handles, labels = plt.gca().get_legend_handles_labels()

# 交换顺序
plt.legend(handles[::-1], labels[::-1])  # 反转 handles 和 labels 的顺序

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()