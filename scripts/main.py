
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import jax.numpy as jnp
from jax import jit
import jax

# Arm parameters
L = jnp.array([2.0, 1.5])  # lengths of links
n = len(L)
step_size = 0.01

# Goal configuration
theta_goal = jnp.array([np.pi / 2, -np.pi / 2])
goal_pos = jnp.array([0, 3.5])  # end-effector position at goal

# Initial joint angles
theta = jnp.array([0.0, 0.0])

end_effector = jnp.array([0.0, 0.0])

# Obstacle parameters
obstacles = [
    {'center': jnp.array([1.5, 1.5]), 'radius': 0.5},
    {'center': jnp.array([0.5, 2.5]), 'radius': 0.3}
]
repulsion_gain = 0.01
repulsion_range = 0.5
attraction_gain = 1.0

@jax.jit
def forward_kinematics(theta):
    points = [jnp.array([0.0, 0.0])]
    for i in range(n):
        angle = jnp.sum(theta[:i+1])
        next_point = points[-1] + L[i] * jnp.array([jnp.cos(angle), jnp.sin(angle)])
        points.append(next_point)
    return points

@jax.jit
def compute_attractive_force(end_effector, goal):
    return -attraction_gain * (end_effector - goal)

@jax.jit
def compute_repulsive_force(point):
    force = jnp.zeros(2)
    for obs in obstacles:
        obs_vec = point - obs['center']
        dist = jnp.linalg.norm(obs_vec)
        addition = jnp.where(dist < repulsion_range, repulsion_gain * (1.0 / dist - 1.0 / repulsion_range) / (dist ** 3) * obs_vec, 0)
        force += addition
    return force

@jax.jit
def jacobian(theta):
    J = jnp.zeros((2, n))
    for i in range(n):
        s = jnp.sum(theta[:i+1])
        # J[:, i] = [-L[i]*np.sin(s), L[i]*np.cos(s)]
        J = J.at[:, i].set([-L[i]*jnp.sin(s), L[i]*jnp.cos(s)])
        for j in range(i):
            s -= theta[j]
            # J[:, i] += [-L[j]*np.sin(s), L[j]*np.cos(s)]
            J += J.at[:, i].set([-L[j]*jnp.sin(s), L[j]*jnp.cos(s)])
    return J

# Plots
plots = []

# Simulation loop
trajectory = []

# for iteration in range(200):
def update(iteration):
    global theta
    global trajectory
    global goal_pos
    global plots
    global end_effector
    global end_eff

    joint_positions = forward_kinematics(theta)
    
    end_effector = joint_positions[-1]
    print(f"end_effector: {end_effector}")

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

    traj = np.array(trajectory)
    end_eff[0].set_xdata(traj[:, 0])
    end_eff[0].set_ydata(traj[:, 1])
    for i in range(len(joint_positions) - 1):
        plots[i][0].set_xdata([joint_positions[i][0], joint_positions[i+1][0]])
        plots[i][0].set_ydata([joint_positions[i][1], joint_positions[i+1][1]])

fig, ax = plt.subplots()
for obs in obstacles:
    circle = plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.5)
    ax.add_patch(circle)
ax.plot(goal_pos[0], goal_pos[1], 'go', label='Goal')

# Initialize plots
joint_positions = forward_kinematics(theta)

end_eff = ax.plot(end_effector[0], end_effector[0], 'b--', label='End-Effector Path')
for i in range(len(joint_positions) - 1):
    j_plt = ax.plot([joint_positions[i][0], joint_positions[i+1][0]],
                    [joint_positions[i][1], joint_positions[i+1][1]], 'ko-')
    plots.append(j_plt)    
    print(f"j_plt: {j_plt}")

ax.set_aspect('equal')
ax.legend()
plt.title('Potential Field Motion Planning')
plt.grid(True)

ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=30)
plt.show()



"""

def forward_kinematics(theta1, theta2, l1, l2):
    # Calculates the end-effector position given joint angles.
    x = l1 * jnp.cos(theta1) + l2 * jnp.cos(theta1 + theta2)
    y = l1 * jnp.sin(theta1) + l2 * jnp.sin(theta1 + theta2)
    return jnp.array([x, y])

def inverse_kinematics(target_x, target_y, l1, l2, initial_guess):
    # Calculates the joint angles given the end-effector target position.
    def loss_fn(thetas):
        x, y = forward_kinematics(thetas[0], thetas[1], l1, l2)
        return (x - target_x)**2 + (y - target_y)**2
    
    thetas = initial_guess
    learning_rate = 0.01
    for _ in range(1000):
      loss, grads = jax.value_and_grad(loss_fn)(thetas)
      thetas = thetas - learning_rate * grads
    return thetas

# Define arm lengths and target position
l1, l2 = 1.0, 1.0
target_x, target_y = 1.5, 0.5

# Initial guess for joint angles
initial_guess = jnp.array([0.0, 0.0])

# Calculate inverse kinematics
thetas = inverse_kinematics(target_x, target_y, l1, l2, initial_guess)

# Print results
print("Joint angles (radians):", thetas)

# Verify the solution
x, y = forward_kinematics(thetas[0], thetas[1], l1, l2)
print("End-effector position:", x, y)

"""