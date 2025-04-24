import matplotlib.pyplot as plt
import numpy as np
from FMM import Body, Simulation, G, dt

def kinetic_energy(bodies):
    return 0.5 * sum(b.mass * np.dot(b.velocity, b.velocity) for b in bodies)

def potential_energy(bodies, epsilon):
    U = 0
    for i, b1 in enumerate(bodies):
        for j in range(i + 1, len(bodies)):
            b2 = bodies[j]
            r2 = np.sum((b1.position - b2.position) ** 2) + epsilon ** 2
            U -= G * b1.mass * b2.mass * np.log(np.sqrt(r2))
    return U

def total_energy(bodies, epsilon):
    return kinetic_energy(bodies) + potential_energy(bodies, epsilon)

def create_bodies(n):
    np.random.seed(42)
    return [
        Body(
            position=np.random.uniform(-100, 100, 2),
            velocity=np.random.uniform(-0.01, 0.01, 2),
            mass=np.random.uniform(0.1, 0.5),
        )
        for _ in range(n)
    ]

orders = [2, 4, 8]
colors = ['r', 'g', 'b']
labels = ['p = 2', 'p = 4', 'p = 8']
timesteps = 100
n_bodies = 30

plt.figure()
for order, color, label in zip(orders, colors, labels):
    bodies = create_bodies(n_bodies)
    sim = Simulation(bodies, expansion_order=order, epsilon=0.1)
    energies = []

    for _ in range(timesteps):
        sim.move()
        energies.append(total_energy(bodies, sim.epsilon))

    initial_energy = energies[0]
    times = np.arange(timesteps) * dt
    plt.plot(times, energies, label=label, color=color)

# Dotted black line for initial energy
plt.axhline(y=initial_energy, color='k', linestyle='--', label='Initial Energy')

plt.xlabel("Time (days)")
plt.ylabel("Total Energy")
plt.title("Energy Conservation vs Expansion Order")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("energy_over_time.png")
plt.show()