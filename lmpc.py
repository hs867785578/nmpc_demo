from casadi import *
import matplotlib.pyplot as plt
import numpy as np

##LMPC实现了ACC的参考速度跟踪
# 决策变量为速度v和加速度a


# 定义参数
N = 200  # 控制的时间范围（预测范围）
dt = 0.05  # 时间步长
v_target = 30.0  # 目标速度（单位：m/s）

# 参考速度权重
Q_v = 1.0
# 加速度权重
Q_a = 0.1

# 初始化CasADi变量
v = SX.sym("v")  # 车辆速度
a = SX.sym("a")  # 车辆加速度

# 定义状态和控制输入的向量
v_vec = SX.sym("v_vec", N + 1)  # 速度向量
a_vec = SX.sym("a_vec", N)  # 加速度控制向量

# 定义目标函数和约束
obj = 0  # 目标函数
g = []  # 约束条件列表
lbx = []  # 优化变量下界
ubx = []  # 优化变量上界

# 初始速度约束
g.append(v_vec[0] - 10)  # 初始速度为10m/s

# 构建目标函数和动态约束
for k in range(N):
    # 目标函数（最小化速度误差和加速度使用）
    obj += Q_v * (v_vec[k] - v_target) ** 2 + Q_a * a_vec[k] ** 2

    # 动态约束
    g.append(v_vec[k + 1] - (v_vec[k] + a_vec[k] * dt))  # 动态模型v(k+1)=v(k)+a(k)*dt

# 优化变量的上下界
for i in range(2 * N + 1):
    if i < N + 1:
        lbx.append(0)
        ubx.append(100)
    else:
        lbx.append(-5)
        ubx.append(5)

# 定义优化问题
opt_variables = vertcat(v_vec, a_vec)
nlp_problem = {"f": obj, "x": opt_variables, "g": vertcat(*g)}
opts = {"ipopt.print_level": 0, "print_time": 0}
solver = nlpsol("solver", "ipopt", nlp_problem, opts)

# 求解
sol = solver(lbg=0, ubg=0, lbx=lbx, ubx=ubx)
v_solution = sol["x"][0 : N + 1]
a_solution = sol["x"][N + 1 : 2 * N + 1]

# 绘制结果
time = np.linspace(0, N * dt, N + 1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time, v_solution, label="Velocity")
plt.plot(time, np.ones_like(time) * v_target, "--", label="Target Velocity")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Tracking")
plt.legend()

plt.subplot(1, 2, 2)
plt.step(time[:-1], a_solution, where="post", label="Acceleration")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Control Input (Acceleration)")
plt.legend()

plt.tight_layout()
plt.show()
