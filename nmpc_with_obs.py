import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import math
from matplotlib.patches import Circle

##NMPC实现了自行车模型下的轨迹跟踪
##控制量为线速度v和角速度omega

# 定义时间步长和预测时间窗口
dt = 0.1  # 时间步长
N = 100  # 预测时间窗口长度

# 障碍物中心位置和半径
x_obs = [9, 2.5]
y_obs = [8, 2.1]
r_obs = [1, 1.5]  # 增加障碍物半径以明显看到避障效果

# 定义系统的初始状态
x0 = 0
y0 = 0
theta0 = math.pi / 2

# 定义参考轨迹（直线）
# x_ref = np.linspace(0, 2, N)
# y_ref = np.linspace(0, 2, N)

# 参考轨迹为正弦曲线
x_ref = np.linspace(0, 20, N)
y_ref = 10 * np.sin(0.1 * x_ref)


# 定义优化变量
opt_vars = []
constraints = []
cost = 0

# 定义状态变量和控制输入的符号变量
x = SX.sym("x")
y = SX.sym("y")
theta = SX.sym("theta")
v = SX.sym("v")
omega = SX.sym("omega")

# 定义系统动态
x_next = x + v * cos(theta) * dt
y_next = y + v * sin(theta) * dt
theta_next = theta + omega * dt

# 跟踪权重
Q_x = 1.0
Q_y = 1.0
# 控制权重
Q_v = 0.1
Q_omega = 0.1

lbx = []
ubx = []

lbg = []
ubg = []

# 创建优化问题
for k in range(N):
    # 控制变量
    v_k = MX.sym(f"v_{k}")
    omega_k = MX.sym(f"omega_{k}")
    opt_vars += [v_k, omega_k]

    lbx += [0, -2]
    ubx += [10, 2]

    # 状态更新
    if k == 0:
        x_k, y_k, theta_k = x0, y0, theta0
    else:
        x_k, y_k, theta_k = x_next, y_next, theta_next

    x_next = x_k + v_k * cos(theta_k) * dt
    y_next = y_k + v_k * sin(theta_k) * dt
    theta_next = theta_k + omega_k * dt

    # 成本函数
    cost += Q_x * (x_next - x_ref[k]) ** 2 + Q_y * (y_next - y_ref[k]) ** 2 + Q_v * v_k**2 + Q_omega * omega_k**2

    # 更新状态
    constraints += [x_next, y_next, theta_next]
    lbg += [-inf, -inf, -inf]
    ubg += [inf, inf, inf]

    for j in range(len(x_obs)):
        # 避障约束
        dist_to_obs = (x_next - x_obs[j]) ** 2 + (y_next - y_obs[j]) ** 2
        cost += 1 / (dist_to_obs - r_obs[j] ** 2 + 1e-6)
        constraints.append(dist_to_obs - r_obs[j] ** 2)
        lbg.append(0.1)
        ubg.append(inf)


# 设置优化问题
opt_problem = {"f": cost, "x": vertcat(*opt_vars), "g": vertcat(*constraints)}

# 使用`ipopt`求解器
solver = nlpsol("solver", "ipopt", opt_problem)

# 求解优化问题
sol = solver(lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)

# 提取解
v_opt = sol["x"][0::2].full().flatten()
omega_opt = sol["x"][1::2].full().flatten()

# 计算最优轨迹
x_traj, y_traj, theta_traj = [x0], [y0], [theta0]
for k in range(N):
    x_next, y_next, theta_next = (
        x_traj[-1] + v_opt[k] * cos(theta_traj[-1]) * dt,
        y_traj[-1] + v_opt[k] * sin(theta_traj[-1]) * dt,
        theta_traj[-1] + omega_opt[k] * dt,
    )
    x_traj.append(x_next)
    y_traj.append(y_next)
    theta_traj.append(theta_next)

# 绘制结果
plt.figure(figsize=(8, 6))
plt.plot(x_ref, y_ref, "r--", label="Reference Trajectory")
plt.plot(x_traj, y_traj, "b", label="NMPC Trajectory")

# 添加障碍物
for m in range(len(x_obs)):
    obstacle = Circle((x_obs[m], y_obs[m]), r_obs[m], color="g", label="Obstacle", fill=True, alpha=0.5)
    plt.gca().add_patch(obstacle)

plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("NMPC Trajectory Tracking")
plt.grid(True)
plt.show()
