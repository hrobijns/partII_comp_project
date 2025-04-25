import time
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
k    = 1.0    # Coulomb constant
soft = 1e-1   # softening length
dt   = 1/24   # time step (in days)

class Body:
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge   = charge
        self.mass     = mass
        self.force    = np.zeros(2)

class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self):
        # reset forces
        for b in self.bodies:
            b.force = np.zeros(2)
        # pairwise Coulomb
        for i, b1 in enumerate(self.bodies):
            for j in range(i+1, len(self.bodies)):
                b2 = self.bodies[j]
                diff = b1.position - b2.position
                dist = np.linalg.norm(diff)
                f_mag = k * b1.charge * b2.charge / (dist**2 + soft**2)
                f_vec = (f_mag / (dist + 1e-16)) * diff
                b1.force +=  f_vec
                b2.force += -f_vec

    def move(self):
        # velocity Verlet: kick–drift–kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
        for b in self.bodies:
            b.position += b.velocity * dt
        self.compute_forces()
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

def time_one_step(N, seed):
    """Initialize N bodies with a given seed, compute initial forces,
    perform one move(), and return elapsed time."""
    np.random.seed(seed)
    bodies = [
        Body(
            position=np.random.uniform(-1, 1, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            charge=1.0,
            mass=1.0
        )
        for _ in range(N)
    ]
    sim = Simulation(bodies)
    sim.compute_forces()          # warm up initial forces
    start = time.perf_counter()
    sim.move()                    # single integration step
    return time.perf_counter() - start

if __name__ == '__main__':
    # Range of N values and seeds
    Ns = np.logspace(1.5, 2.5, num=10, dtype=int)
    seeds = range(10)

    # Collect timings: shape (len(Ns), len(seeds))
    times = np.zeros((len(Ns), len(seeds)))
    for i, N in enumerate(Ns):
        print(N)
        for j, seed in enumerate(seeds):
            times[i, j] = time_one_step(N, seed)

    # Compute mean and standard deviation across seeds
    means = times.mean(axis=1)
    stds  = times.std(axis=1)

    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        Ns, means, yerr=stds,
        fmt='o-', markersize=8, capsize=5, linewidth=2
    )
    plt.title("Computation Time for 10 Steps (s)", fontsize=16)
    plt.xlabel("Number of Bodies (N)", fontsize=14)
    plt.ylabel("Time (s) for 1 Step", fontsize=14)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()