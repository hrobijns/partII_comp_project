import numpy as np
import matplotlib.pyplot as plt
import copy

# Gravitational constant in AU^3 M_sun^-1 day^-2
G = 2.959122082855911e-4
# timestep (1 hour)
dt = 1/24

def compute_forces(bodies, e=1e-1):
    for b in bodies:
        b.force[:] = 0.0
    N = len(bodies)
    for i in range(N):
        for j in range(i+1, N):
            b1, b2 = bodies[i], bodies[j]
            diff = b2.position - b1.position
            r2 = np.dot(diff, diff) + e**2
            r = np.sqrt(r2)
            F = G * b1.mass * b2.mass / r2
            fvec = F * diff / r
            b1.force += fvec
            b2.force -= fvec

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)

class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies
        compute_forces(self.bodies)

    def move_leapfrog(self):
        # kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
        # drift
        for b in self.bodies:
            b.position += b.velocity * dt
        # kick & force update
        compute_forces(self.bodies)
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

    def move_euler(self):
        for b in self.bodies:
            b.position += b.velocity * dt
        for b in self.bodies:
            b.velocity += (b.force / b.mass) * dt
        compute_forces(self.bodies)

    def total_energy(self, e=1e-1):
        # kinetic
        K = sum(0.5 * b.mass * np.dot(b.velocity, b.velocity) for b in self.bodies)
        # potential
        U = 0.0
        N = len(self.bodies)
        for i in range(N):
            for j in range(i+1, N):
                b1, b2 = self.bodies[i], self.bodies[j]
                r2 = np.sum((b1.position - b2.position)**2) + e**2
                U -= G * b1.mass * b2.mass / np.sqrt(r2)
        return K + U

if __name__ == "__main__":
    np.random.seed(41)
    bodies_initial = [
        Body(
            position=np.random.uniform(-5, 5, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        ) for _ in range(50)
    ]

    # Create independent simulations
    sim_lf = Simulation(copy.deepcopy(bodies_initial))
    sim_eu = Simulation(copy.deepcopy(bodies_initial))

    steps = 1000
    energy_lf = []
    energy_eu = []

    # initial energies
    E0_lf = sim_lf.total_energy()
    E0_eu = sim_eu.total_energy()

    for step in range(steps):
        sim_lf.move_leapfrog()
        sim_eu.move_euler()
        # compute and record percentage change
        Elf = sim_lf.total_energy()
        Eeu = sim_eu.total_energy()
        energy_lf.append(100 * (Elf - E0_lf) / abs(E0_lf))
        energy_eu.append(100 * (Eeu - E0_eu) / abs(E0_eu))

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(range(steps), energy_lf, 'r', label='kick-drift-kick')
    plt.plot(range(steps), energy_eu, 'g', label='Euler')
    plt.axhline(0, color='k', linestyle=':')
    #plt.title('Energy (% change from initial)')
    plt.xlabel('Step')
    plt.xlim(0,1000)
    plt.ylabel('% energy change from initial')
    plt.legend()
    plt.tight_layout()
    plt.show()