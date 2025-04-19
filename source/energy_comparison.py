import numpy as np
import matplotlib.pyplot as plt
from naivegravity import Body, Simulation

# Constants
G = 2.959122082855911e-4  # gravitational constant in AU^3 M_sun^-1 day^-2
dt = 1/24  # Time step in days
steps = 100
softening_values = [0.1]  # Different softening parameters

# Function to compute total energy
def compute_total_energy(bodies, e):
    kinetic_energy = sum(0.5 * body.mass * np.linalg.norm(body.velocity) ** 2 for body in bodies)
    potential_energy = 0.0
    for i, body1 in enumerate(bodies):
        for j, body2 in enumerate(bodies):
            if i < j:
                distance = np.linalg.norm(body1.position - body2.position)
                if distance > 6e-6:  # Ensure nonzero division
                    potential_energy -= G * body1.mass * body2.mass / (distance + e)
    return kinetic_energy + potential_energy

# Initialize plot
plt.figure(figsize=(10, 6))
time_steps = np.arange(steps + 1)

# Run simulations for different softening values
for e in softening_values:
    np.random.seed(42)  # Fix seed for fair comparison
    bodies = [
        Body(
            position=np.random.uniform(-100, 100, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(100)
    ]
    
    simulation = Simulation(bodies)
    total_energies = [compute_total_energy(simulation.bodies, e)]
    print(e)
    for _ in range(steps):
        simulation.move()
        total_energies.append(compute_total_energy(simulation.bodies, e))
        print(_)
        
    # Plot energy trend
    plt.plot(time_steps, total_energies, label=f'e = {e}')

# Format and display plot
plt.xlabel("Time Step")
plt.ylabel("Total Energy (AU^2/day^2)")
plt.title("Energy Conservation in N-Body Simulation for Different Softening Parameters")
plt.legend()
plt.show()
