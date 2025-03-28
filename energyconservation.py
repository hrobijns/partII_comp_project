import numpy as np
import matplotlib.pyplot as plt
from naivegravity import Body, Simulation

G = 6.67430e-11  # Gravitational constant
dt = 3600  # Time step (seconds)
STEPS = 500  # Number of simulation steps

# Function to compute total energy
def compute_total_energy(bodies):
    kinetic_energy = sum(0.5 * body.mass * np.linalg.norm(body.velocity) ** 2 for body in bodies)
    potential_energy = 0.0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                distance = np.linalg.norm(body1.position - body2.position)
                if distance > 0:
                    potential_energy -= G * body1.mass * body2.mass / distance
    return kinetic_energy + potential_energy

# Initialize simulation
np.random.seed(42)
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=np.random.uniform(5e26, 5e27)
    ) for _ in range(100)
]

simulation = Simulation(bodies)

time_steps = np.arange(STEPS)
total_energies = []

# Run simulation and record energy
total_energies.append(compute_total_energy(simulation.bodies))
for _ in range(STEPS):
    simulation.move()
    total_energies.append(compute_total_energy(simulation.bodies))

# Plot energy over time
plt.figure(figsize=(8, 6))
plt.plot(time_steps, total_energies[:-1], label="Total Energy", color='blue')
plt.xlabel("Time Step")
plt.ylabel("Total Energy (J)")
plt.title("Total Energy Conservation in N-Body Simulation")
plt.legend()
plt.show()