import time
import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation, Body

def benchmark_theta(theta_values, N, seeds, space_size):
    """
    Run a single step of the simulation for each theta in theta_values,
    averaged over the provided random seeds, and compute std.dev.

    Returns:
        avg_times (list of float): Mean time per step for each theta.
        std_times (list of float): Std. dev. of time per step for each theta.
    """
    avg_times = []
    std_times = []
    for theta_value in theta_values:
        times = []
        for seed in seeds:
            np.random.seed(seed)
            positions = np.random.uniform(-space_size, space_size, size=(N, 2))
            bodies = [Body(position=pos, velocity=[0, 0], charge=1.0) for pos in positions]

            sim = Simulation(bodies, space_size, theta_value)

            start = time.time()
            sim.step()
            end = time.time()
            times.append(end - start)

        mean_t = np.mean(times)
        std_t  = np.std(times, ddof=1)  # sample std. dev.

        avg_times.append(mean_t)
        std_times.append(std_t)
        print(f"theta={theta_value:.2f}: avg = {mean_t:.6f}s, std = {std_t:.6f}s")
    return avg_times, std_times

def main():
    theta_values = np.linspace(0.1, 1.0, 10)
    N = 200
    seeds = list(range(10))
    space_size = 20.0

    # Run benchmark
    avg_times, std_times = benchmark_theta(theta_values, N, seeds, space_size)

    # Plot results with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(theta_values, avg_times, yerr=std_times, fmt='o-', capsize=5)
    plt.xlabel(r'$\theta$', fontsize = 16)  
    plt.ylabel('Computation time per step (s)', fontsize = 16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/theta.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
