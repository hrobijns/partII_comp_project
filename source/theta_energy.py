import numpy as np
import matplotlib.pyplot as plt
from barneshut import Body, Simulation as BHTreeSim
from naivegravity import Simulation as NaiveSim

# Constants
G = 2.959122082855911e-4  # AU^3 M_sun^-1 day^-2
dt = 1/24  # days
steps = 300
theta_values = [0,0.4,0.8,1.2,1.6]

# Function to compute total energy
def compute_total_energy(bodies):
    kinetic_energy = sum(0.5 * body.mass * np.linalg.norm(body.velocity) ** 2 for body in bodies)
    potential_energy = 0.0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                distance = np.linalg.norm(body1.position - body2.position)
                if distance > 1e-5:
                    potential_energy -= G * body1.mass * body2.mass / ((distance**2 + 1e-2)**(1/2))
    return kinetic_energy + potential_energy

# Prepare figure
plt.figure(figsize=(10, 6))
time_steps = np.arange(steps + 1)

# Fix seed and initial bodies
np.random.seed(26)
initial_bodies = [
    Body(
        position=np.random.uniform(-2, 2, 2),
        velocity=np.random.uniform(-0.05, 0.05, 2),
        mass=np.random.uniform(0.1, 1),
    )
    for _ in range(100)
]

# Run naive gravity first to get baseline energy
naive_bodies = [Body(b.position.copy(), b.velocity.copy(), b.mass) for b in initial_bodies]
naive_sim = NaiveSim(naive_bodies)
naive_energies = [compute_total_energy(naive_sim.bodies)]
for _ in range(steps):
    naive_sim.move()
    naive_energies.append(compute_total_energy(naive_sim.bodies))

# Run Barnes-Hut simulations and compute energy differences
for theta in theta_values:
    bodies = [Body(b.position.copy(), b.velocity.copy(), b.mass) for b in initial_bodies]
    simulation = BHTreeSim(bodies, space_size=2, theta=theta)
    bh_energies = [compute_total_energy(simulation.bodies)]
    for _ in range(steps):
        simulation.move()
        bh_energies.append(compute_total_energy(simulation.bodies))

    # Compute absolute difference from naive energy
    energy_diffs = [abs(e_bh - e_naive) for e_bh, e_naive in zip(bh_energies, naive_energies)]
    plt.plot(time_steps, energy_diffs, label=f'|ΔE| θ = {theta}')

# Finalize plot
plt.xlabel("Time Step")
plt.ylabel("Absolute Energy Difference (AU²/day²)")
plt.title("Absolute Energy Difference from Naive Gravity vs Barnes-Hut")
plt.legend()
plt.tight_layout()
plt.show()