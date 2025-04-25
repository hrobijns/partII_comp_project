import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from simulation import SimulationVectorised, Simulation, Body

def benchmark_vectorized(N, seeds, n_steps=10):
    times = []
    for seed in seeds:
        np.random.seed(seed)
        pos    = np.random.uniform(-1, 1, size=(N, 2))
        vel    = np.random.uniform(-0.05, 0.05, size=(N, 2))
        charge = np.ones(N)
        sim = SimulationVectorised(pos, vel, charge)
        sim.compute_forces()  # warm-up

        t0 = time.perf_counter()
        for _ in range(n_steps):
            sim.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)

def benchmark_naive(N, seeds, n_steps=10):
    times = []
    for seed in seeds:
        np.random.seed(seed)
        bodies = [
            Body(
                position=np.random.uniform(-1, 1, 2),
                velocity=np.random.uniform(-0.05, 0.05, 2),
                charge=1.0
            ) for _ in range(N)
        ]
        sim = Simulation(bodies)
        sim.compute_forces()  # warm-up

        t0 = time.perf_counter()
        for _ in range(n_steps):
            sim.step()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)

def main():
    seeds = [0, 1, 2, 3, 4]
    n_steps = 10
    N_values = np.linspace(10, 1000, 12, dtype=int)

    vec_means, vec_stds = [], []
    naive_means, naive_stds = [], []

    for N in N_values:
        vec_times   = benchmark_vectorized(N, seeds, n_steps)
        naive_times = benchmark_naive(N, seeds, n_steps)

        vec_means.append(vec_times.mean())
        vec_stds.append(vec_times.std(ddof=1))

        naive_means.append(naive_times.mean())
        naive_stds.append(naive_times.std(ddof=1))

        print(f"N={N:4d}  vec: {vec_means[-1]:.4f}s ±{vec_stds[-1]:.4f}s  "
              f"naive: {naive_means[-1]:.4f}s ±{naive_stds[-1]:.4f}s")

    # assemble results into a DataFrame
    df = pd.DataFrame({
        'N':            N_values,
        'vector_mean':  vec_means,
        'vector_std':   vec_stds,
        'naive_mean':   naive_means,
        'naive_std':    naive_stds,
    })

    # print the full table
    print("\nFinal benchmark results:")
    print(df.to_string(index=False))

    # save to CSV
    out_csv = '\data\naivetiming.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    # Plot both on the same axes
    plt.figure(figsize=(8,5))
    plt.errorbar(
        N_values, vec_means, yerr=vec_stds,
        fmt='o-', capsize=4, label='Vectorized'
    )
    plt.errorbar(
        N_values, naive_means, yerr=naive_stds,
        fmt='s--', capsize=4, label='Naive'
    )
    plt.title(f"Benchmark: {n_steps} steps, {len(seeds)} seeds")
    plt.xlabel("Number of particles (N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()