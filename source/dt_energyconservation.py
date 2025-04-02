import numpy as np
import matplotlib.pyplot as plt
from naivegravity import Body, Simulation

# Constants
G = 2.959122082855911e-4  # gravitational constant in AU^3 M_sun^-1 day^-2
year_days = 365.25  # Number of days in a year
time_steps = [24,12,6,3,2,1]  # Different time step values in days
softening_value = 0.01  # Fixed softening parameter

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

# Run simulations for different time steps
for dt in time_steps:
    np.random.seed(1)  # Fix seed for fair comparison
    bodies = [
        Body(
            position=np.random.uniform(-2, 2, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(100)
    ]
    
    simulation = Simulation(bodies)
    total_energies = [compute_total_energy(simulation.bodies, softening_value)]
    
    steps = int(year_days / dt)  # Simulate for one year
    
    for _ in range(steps):
        simulation.move()
        total_energies.append(compute_total_energy(simulation.bodies, softening_value))
        print(_)
    
    # Plot energy trend
    plt.plot(np.arange(steps + 1) * dt, total_energies, label=f'dt = {dt} days')

# Mark initial energy level with a horizontal dotted line
initial_energy = total_energies[0]
plt.axhline(initial_energy, color='black', linestyle='dotted', label='Initial Energy')

# Format and display plot
plt.xlabel("Time (Days)")
plt.ylabel("Total Energy (AU^2/day^2)")
plt.title("Energy Conservation for Different Time Steps over One Year")
plt.legend()
plt.show()
