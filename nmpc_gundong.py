from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

# 定义时间步长和预测时间窗口
dt = 0.1  # 时间步长
N_total = 100  # 总的时间步长
N_pred = 20  # 滚动预测的时间窗口长度

# 定义系统的初始状态
x0 = 0
y0 = 0
theta0 = math.pi / 2

# 参考轨迹为正弦曲线
x_ref_full = np.linspace(0, 20, N_total)
y_ref_full = 10 * np.sin(0.1 * x_ref_full)

# 跟踪权重
Q_x = 1.0
Q_y = 1.0
# 控制权重
Q_v = 0
Q_omega = 0

# 初始化状态列表
x_traj = [x0]
y_traj = [y0]
theta_traj = [theta0]

for i in range(N_total - N_pred):
    # 每次优化只考虑接下来的N_pred个点
    x_ref = x_ref_full[i:i+N_pred]
    y_ref = y_ref_full[i:i+N_pred]
    
    # 定义优化变量
    opt_vars = []
    constraints = []
    cost = 0

    lbx = []
    ubx = []
    
    # 定义状态变量和控制输入的符号变量
    x = SX.sym("x")
    y = SX.sym("y")
    theta = SX.sym("theta")
    
    # 创建优化问题
    for k in range(N_pred):
        # 控制变量
        v_k = MX.sym(f"v_{k}")
        omega_k = MX.sym(f"omega_{k}")
        opt_vars += [v_k, omega_k]
        lbx += [0, -2]
        ubx += [10, 2]
        
        # 状态更新
        if k == 0:
            x_k, y_k, theta_k = x_traj[-1], y_traj[-1], theta_traj[-1]
        else:
            x_k, y_k, theta_k = x_next, y_next, theta_next
        
        x_next = x_k + v_k * cos(theta_k) * dt
        y_next = y_k + v_k * sin(theta_k) * dt
        theta_next = theta_k + omega_k * dt
        
        # 成本函数
        cost += Q_x * (x_next - x_ref[k]) ** 2 + Q_y * (y_next - y_ref[k]) ** 2 + Q_v * v_k**2 + Q_omega * omega_k**2
    
    # 设置优化问题
    opt_problem = {"f": cost, "x": vertcat(*opt_vars)}
    
    # 使用`ipopt`求解器
    solver = nlpsol("solver", "ipopt", opt_problem)
    
    # 求解优化问题
    sol = solver(lbg=-inf, ubg=inf, lbx=lbx, ubx=ubx)
    
    # 提取解
    v_opt = sol["x"][0::2].full().flatten()[0]
    omega_opt = sol["x"][1::2].full().flatten()[0]

    print(f"Optimal v: {v_opt}, Optimal omega: {omega_opt}")
    
    # 更新状态
    x_next, y_next, theta_next = (
        x_traj[-1] + v_opt * cos(theta_traj[-1]) * dt,
        y_traj[-1] + v_opt * sin(theta_traj[-1]) * dt,
        theta_traj[-1] + omega_opt * dt,
    )
    x_traj.append(x_next)
    y_traj.append(y_next)
    theta_traj.append(theta_next)


# # 绘制结果
# plt.figure(figsize=(8, 6))
# plt.plot(x_ref_full, y_ref_full, "r--", label="Reference Trajectory")
# plt.plot(x_traj, y_traj, "b", label="NMPC Trajectory")
# plt.legend()
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("NMPC Trajectory Tracking with Rolling Horizon")
# plt.grid(True)
# plt.show()


# 动画

fig, ax = plt.subplots()
ax.set_xlim((0, 20))
ax.set_ylim((-10, 10))
line, = ax.plot([], [], 'r--', label="Reference Trajectory")
point, = ax.plot([], [], 'bo', label="NMPC Trajectory")

def init():
    line.set_data(x_ref_full, y_ref_full)
    point.set_data([], [])
    return line, point,

def update(frame):
    point.set_data(x_traj[:frame], y_traj[:frame])
    return line, point,

ani = FuncAnimation(fig, update, frames=range(len(x_traj)), init_func=init, blit=True)

plt.legend()
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.title("NMPC Trajectory Tracking")

plt.grid(True)
plt.show()