import time
import numpy as np
import matplotlib.pyplot as plt

from simulation import Simulation, Body
from quadtree import theta

def benchmark_step(N_values, seeds, space_size, theta_value):
    """
    Run a single step of the simulation for each N in N_values,
    averaged over the provided random seeds.

    Parameters:
        N_values (array-like): List of body counts to test.
        seeds (list of int): Random seeds for reproducibility.
        space_size (float): Half-width of the square simulation domain.
        theta_value (float): Barnes-Hut opening angle.

    Returns:
        list of float: Average time per step for each N.
    """
    avg_times = []
    for N in N_values:
        times = []
        for seed in seeds:
            # Set seed for reproducibility
            np.random.seed(seed)
            # Initialize bodies with random positions, zero velocity, unit charge
            positions = np.random.uniform(-space_size, space_size, size=(N, 2))
            bodies = [Body(position=pos, velocity=[0, 0], charge=1.0) for pos in positions]

            # Initialize simulation (computes initial forces)
            sim = Simulation(bodies, space_size, theta_value)

            # Time a single step
            start = time.time()
            sim.step()
            end = time.time()
            times.append(end - start)

        avg = np.mean(times)
        avg_times.append(avg)
        print(f"N={N}: avg step time = {avg:.6f} s")
    return avg_times

def main():
    # Parameters
    N_values = np.linspace(10, 500, 10, dtype=int)
    seeds = list(range(3))
    space_size = 1.0
    theta_value = theta

    # Run benchmark
    avg_times = benchmark_step(N_values, seeds, space_size, theta_value)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(N_values, avg_times, marker='o')
    plt.xlabel('Number of bodies N')
    plt.ylabel('Time per step (s)')
    plt.title('Barnes-Hut Simulation Step Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
