import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.code.shift_and_add import ShiftAndAdd
from src.code.rs import RS

np.set_printoptions(precision=16, suppress=False)

# 参数设置
num_iterations = 100  # 每个数据点的独立实验次数
n, k = 9, 6  # 编码参数
row_counts = [12 * i for i in range(1, int((10000 - 12) / 12) + 1)]  # A 的行数从 12 开始，每次增加 12
variance = 1e12

def run_test(algorithm_name, code_class, generate_data_fn, rows):
    relative_error_list = []  # 存储每次独立实验的相对误差

    for iteration in range(num_iterations):
        # 生成随机数据
        A, x = generate_data_fn(rows)

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

# 数据生成函数
def generate_data_shift_and_add(rows):
    std_dev = np.sqrt(float(variance))
    A = np.random.normal(loc=0, scale=std_dev, size=(rows, 100))
    x = np.random.normal(loc=0, scale=std_dev, size=(100, 1))
    return A, x

def generate_data_rs(rows):
    std_dev = np.sqrt(float(variance))
    A = np.random.normal(loc=0, scale=std_dev, size=(rows, 100))
    x = np.random.normal(loc=0, scale=std_dev, size=(100, 1))
    return A, x

# 记录结果
shift_and_add_relative_error = []
rs_relative_error = []

for rows in row_counts:
    print(f"Running tests for row count: {rows}")

    # ShiftAndAdd
    average_relative_error_shift_and_add = run_test(
        "ShiftAndAdd",
        ShiftAndAdd,
        generate_data_shift_and_add,
        rows
    )
    shift_and_add_relative_error.append(average_relative_error_shift_and_add)

    # RS
    average_relative_error_rs = run_test(
        "RS",
        RS,
        generate_data_rs,
        rows
    )
    rs_relative_error.append(average_relative_error_rs)

# 保存数据到 Excel
data = {
    "Row Count": row_counts,
    "Shift and Add Relative Error": shift_and_add_relative_error,
    "Reed Solomon Relative Error": rs_relative_error
}
df = pd.DataFrame(data)
df.to_excel("experiment_results.xlsx", index=False)
print("Results saved to 'experiment_results.xlsx'.")

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(row_counts, shift_and_add_relative_error, label="Shift and add", marker='o')
plt.plot(row_counts, rs_relative_error, label="Reed solomon", marker='x')
plt.xscale('log')  # 横坐标使用对数刻度
plt.yscale('log')  # 纵坐标使用对数刻度
plt.xlabel("Number of rows")
plt.ylabel("Relative error")

# 获取当前图例的 handles 和 labels
handles, labels = plt.gca().get_legend_handles_labels()

# 交换顺序
plt.legend(handles[::-1], labels[::-1])  # 反转 handles 和 labels 的顺序
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()