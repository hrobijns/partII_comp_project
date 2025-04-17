import time
import numpy as np
import matplotlib.pyplot as plt
from barneshut3D import Body, Simulation
from scipy.stats import linregress

def generate_bodies(n, seed=42):
    np.random.seed(seed)
    return [
        Body(
            position=np.random.uniform(-0.5, 0.5, 3),  # 3D position
            velocity=np.random.uniform(-0.05, 0.05, 3),  # 3D velocity
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(n)
    ]

def count_forces(theta, steps=5, n_bodies=500):
    bodies = generate_bodies(n_bodies)
    sim = Simulation(bodies, space_size=2, theta=theta)

    total_force_evals = 0
    for _ in range(steps):
        sim.move()
        total_force_evals += sim.force_count  # we assume we add force counting below

    return total_force_evals / steps  # average per step

def benchmark_thetas_force(theta_values, steps=10, n_bodies=100):
    forces = []
    for theta in theta_values:
        print(f"Running force benchmark for theta={theta:.2f}...")
        f = count_forces(theta, steps, n_bodies)
        forces.append(f)
    return forces

if __name__ == "__main__":
    theta_values = np.linspace(0.1, 1, 10)
    force_counts = benchmark_thetas_force(theta_values, steps=20, n_bodies=200)

    log_force = np.log(force_counts)
    log_theta_values = np.log(theta_values)

    slope, intercept, *_ = linregress(log_theta_values, log_force)
    print(f"Estimated alpha (force ∝ theta^alpha): {slope:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(theta_values, force_counts, marker='o')
    plt.xlabel('Theta (Barnes-Hut opening angle)')
    plt.ylabel('Avg. Force Computations per Step')
    plt.title('Force Count vs. Theta')
    plt.grid(True)
    plt.tight_layout()
    plt.show()