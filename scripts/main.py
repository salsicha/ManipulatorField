
import numpy as np
import matplotlib.pyplot as plt

# Arm parameters
L = np.array([2.0, 1.5])  # lengths of links
n = len(L)
step_size = 0.01

# Goal configuration
theta_goal = np.array([np.pi / 2, -np.pi / 2])
goal_pos = np.array([0, 3.5])  # end-effector position at goal

# Initial joint angles
theta = np.array([0.0, 0.0])

# Obstacle parameters
obstacles = [
    {'center': np.array([1.5, 1.5]), 'radius': 0.5},
    {'center': np.array([0.5, 2.5]), 'radius': 0.3}
]
repulsion_gain = 0.01
repulsion_range = 0.5
attraction_gain = 1.0

def forward_kinematics(theta):
    points = [np.array([0.0, 0.0])]
    for i in range(n):
        angle = np.sum(theta[:i+1])
        next_point = points[-1] + L[i] * np.array([np.cos(angle), np.sin(angle)])
        points.append(next_point)
    return points

def compute_attractive_force(end_effector, goal):
    return -attraction_gain * (end_effector - goal)

def compute_repulsive_force(point):
    force = np.zeros(2)
    for obs in obstacles:
        obs_vec = point - obs['center']
        dist = np.linalg.norm(obs_vec)
        if dist < repulsion_range:
            force += repulsion_gain * (1.0 / dist - 1.0 / repulsion_range) / (dist ** 3) * obs_vec
    return force

def jacobian(theta):
    J = np.zeros((2, n))
    for i in range(n):
        s = np.sum(theta[:i+1])
        J[:, i] = [-L[i]*np.sin(s), L[i]*np.cos(s)]
        for j in range(i):
            s -= theta[j]
            J[:, i] += [-L[j]*np.sin(s), L[j]*np.cos(s)]
    return J

# Simulation loop
trajectory = []
for iteration in range(200):
    joint_positions = forward_kinematics(theta)
    end_effector = joint_positions[-1]

    # Attractive force
    f_att = compute_attractive_force(end_effector, goal_pos)

    # Repulsive forces
    f_rep = np.zeros(2)
    for pos in joint_positions[1:]:
        f_rep += compute_repulsive_force(pos)

    f_total = f_att + f_rep

    # Compute Jacobian and update theta
    J = jacobian(theta)
    dtheta = np.linalg.pinv(J) @ f_total
    theta += step_size * dtheta
    trajectory.append(end_effector.copy())

# Plotting
fig, ax = plt.subplots()
for obs in obstacles:
    circle = plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.5)
    ax.add_patch(circle)

traj = np.array(trajectory)
ax.plot(traj[:, 0], traj[:, 1], 'b--', label='End-Effector Path')
final_positions = forward_kinematics(theta)
for i in range(len(final_positions) - 1):
    ax.plot([final_positions[i][0], final_positions[i+1][0]],
            [final_positions[i][1], final_positions[i+1][1]], 'ko-')
ax.plot(goal_pos[0], goal_pos[1], 'go', label='Goal')
ax.set_aspect('equal')
ax.legend()
plt.title('Potential Field Motion Planning')
plt.grid(True)
plt.show()
