"""
Benchmark script for the Fast Multipole Method implementation in simulation.py.

Runs the FMM-based `potential` call for various numbers of particles N,
averaged over multiple random seeds, and plots timing results.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from simulation import Particle, potential


def generate_particles(N, seed, bounds=(-10.0, 10.0)):
    """
    Generate a list of N Particle instances with random positions and charges.

    Parameters:
    - N: number of particles
    - seed: integer seed for reproducibility
    - bounds: tuple (min, max) for uniform sampling in x and y

    Returns:
    - List[Particle]
    """
    np.random.seed(seed)
    xs = np.random.uniform(bounds[0], bounds[1], size=N)
    ys = np.random.uniform(bounds[0], bounds[1], size=N)
    qs = np.random.uniform(-1.0, 1.0, size=N)
    return [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]


def benchmark_fmm(N_values, n_seeds=5, tree_thresh=None, bbox=None, p_order=5):
    """
    Benchmark the FMM `potential` implementation over a range of N values.

    Parameters:
    - N_values: iterable of particle counts (integers)
    - n_seeds: number of random seeds to average over
    - tree_thresh: threshold for quadtree subdivision (pass-through to `potential`)
    - bbox: bounding box for domain (pass-through to `potential`)
    - p_order: order of multipole expansion

    Returns:
    - dict mapping N -> average elapsed time (seconds)
    """
    results = {}
    for N in N_values:
        print(N)
        timings = []
        for seed in range(n_seeds):
            particles = generate_particles(N, seed)
            start = time.perf_counter()
            potential(particles, tree_thresh=tree_thresh, bbox=bbox, p_order=p_order)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        avg_time = sum(timings) / len(timings)
        print(f"N={N:4d} | avg time over {n_seeds} runs: {avg_time:.6f} s")
        results[N] = avg_time

    return results


if __name__ == "__main__":
    # Define 10 values of N evenly spaced between 10 and 1000
    N_values = np.linspace(10, 1000, 10, dtype=int)

    # Run benchmark and collect results
    results = benchmark_fmm(N_values)

    # Plot results
    Ns = list(results.keys())
    times = [results[N] for N in Ns]

    plt.figure(figsize=(8, 5))
    plt.plot(Ns, times, marker='o')
    plt.xlabel('Number of particles (N)')
    plt.ylabel('Average time (seconds)')
    plt.title('FMM Benchmark: Average runtime vs. N')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
