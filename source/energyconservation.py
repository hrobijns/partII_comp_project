import numpy as np
import matplotlib.pyplot as plt
from naivegravity import Body, Simulation

# note that units are in terms of AU (distance), day (time) and solar mass (mass)
G = 2.959122082855911e-4  # gravitational constant in AU^3 M_sun^-1 day^-2
dt = 1  # Time step in days (for simplicity, 1 day per update)
steps = 1000

# Function to compute total energy
def compute_total_energy(bodies):
    e = 1e-2 # a small parameter to avoid singularities (here, 0.01 AU)
    kinetic_energy = sum(0.5 * body.mass * np.linalg.norm(body.velocity) ** 2 for body in bodies)
    potential_energy = 0.0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                distance = np.linalg.norm(body1.position - body2.position)
                if distance > 6e6:
                    potential_energy -= G * body1.mass * body2.mass / (distance + e)
    return kinetic_energy + potential_energy

# Initialize simulation
np.random.seed()
bodies = [
    Body(
        position=np.random.uniform(-2, 2, 2),  # in AU
        velocity=np.random.uniform(-0.05, 0.05, 2),  # in AU/day
        mass=np.random.uniform(0.1, 1),  # in M_sun
    )
    for _ in range(100)
    ]

simulation = Simulation(bodies)

time_steps = np.arange(steps)
total_energies = []

# Run simulation and record energy
total_energies.append(compute_total_energy(simulation.bodies))
for i in range(steps):
    print('step: '+str(i))
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