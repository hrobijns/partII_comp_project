import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from barneshut import Simulation as BarnesHutSimulation, Body as BarnesHutBody
from naivegravity import Simulation as NaiveSimulation, Body as NaiveBody

# Constants
NUM_BODIES = 100
SIMULATION_STEPS = 100
INTERVAL = 50  # Milliseconds

# Generate random bodies
np.random.seed(42)
bodies_barneshut = [
    BarnesHutBody(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=np.random.uniform(5e26, 5e27)
    ) for _ in range(NUM_BODIES)
]

bodies_naive = [
    NaiveBody(
        position=body.position.copy(),
        velocity=body.velocity.copy(),
        mass=body.mass
    ) for body in bodies_barneshut  # Ensure both simulations start identically
]

# Initialize simulations
simulation_barneshut = BarnesHutSimulation(bodies_barneshut, -1e11, 1e11, -1e11, 1e11)
simulation_naive = NaiveSimulation(bodies_naive)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('black')

# Set up subplots
for ax in axes:
    ax.set_xlim(-1e11, 1e11)
    ax.set_ylim(-1e11, 1e11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('black')

axes[0].set_title("Barnes-Hut Simulation", color="white")
axes[1].set_title("Naïve N-Body Simulation", color="white")

# Initialize scatter plots
scatters_barneshut = axes[0].scatter(
    [body.position[0] for body in bodies_barneshut], 
    [body.position[1] for body in bodies_barneshut], 
    color='white', 
    s=[max(body.mass / 1e27, 5) for body in bodies_barneshut]
)

scatters_naive = axes[1].scatter(
    [body.position[0] for body in bodies_naive], 
    [body.position[1] for body in bodies_naive], 
    color='white', 
    s=[max(body.mass / 1e27, 5) for body in bodies_naive]
)

# Update function for animation
def update(frame):
    simulation_barneshut.move()
    simulation_naive.move()

    # Update positions
    positions_barneshut = np.array([body.position for body in bodies_barneshut])
    positions_naive = np.array([body.position for body in bodies_naive])

    scatters_barneshut.set_offsets(positions_barneshut)
    scatters_naive.set_offsets(positions_naive)
    
    return scatters_barneshut, scatters_naive

# Run animation
ani = FuncAnimation(fig, update, frames=SIMULATION_STEPS, interval=INTERVAL, repeat=True)

plt.show()