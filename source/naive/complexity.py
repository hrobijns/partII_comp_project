import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tracemalloc

from simulation import SimulationVectorised, Simulation, Body

def benchmark_vectorised(N, seeds, n_steps=10):
    times = []
    memories = []
    for seed in seeds:
        np.random.seed(seed)
        pos    = np.random.uniform(-1, 1, size=(N, 2))
        vel    = np.random.uniform(-0.05, 0.05, size=(N, 2))
        charge = np.ones(N)
        sim = SimulationVectorised(pos, vel, charge)

        # warm-up outside measurement
        sim.compute_forces()

        # start measuring memory allocations
        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            sim.step()
        t1 = time.perf_counter()
        # get peak memory and stop tracing
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        # convert bytes to megabytes
        memories.append(peak / (1024 ** 2))

    return np.array(times), np.array(memories)

def benchmark_naive(N, seeds, n_steps=10):
    times = []
    memories = []
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

        # warm-up
        sim.compute_forces()

        tracemalloc.start()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            sim.step()
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        memories.append(peak / (1024 ** 2))

    return np.array(times), np.array(memories)

def main():
    seeds    = [3, 45, 1]
    n_steps  = 1
    N_values = np.linspace(10, 500, 12, dtype=int)
    N_values = [10,100,1000,10000,100000]

    # storage for timing & memory stats
    vec_time_means, vec_time_stds = [], []
    vec_mem_means,  vec_mem_stds  = [], []
    naive_time_means, naive_time_stds = [], []
    naive_mem_means, naive_mem_stds    = [], []

    for N in N_values:
        vec_times,   vec_mems   = benchmark_vectorised(N, seeds, n_steps)
        #naive_times, naive_mems = benchmark_naive(N, seeds, n_steps)

        vec_time_means.append(vec_times.mean())
        vec_time_stds.append(vec_times.std(ddof=1))
        vec_mem_means.append(vec_mems.mean())
        vec_mem_stds.append(vec_mems.std(ddof=1))

        #naive_time_means.append(naive_times.mean())
        #naive_time_stds.append(naive_times.std(ddof=1))
        #naive_mem_means.append(naive_mems.mean())
        #naive_mem_stds.append(naive_mems.std(ddof=1))

        print(
            f"N={N:4d}  vec: {vec_times.mean():.4f}s ±{vec_times.std(ddof=1):.4f}s, "
            f"{vec_mems.mean():.2f} MB ±{vec_mems.std(ddof=1):.2f} MB  |  "
        #    f"naive: {naive_times.mean():.4f}s ±{naive_times.std(ddof=1):.4f}s, "
        #    f"{naive_mems.mean():.2f} MB ±{naive_mems.std(ddof=1):.2f} MB"
        )

    # assemble results into a DataFrame
    df = pd.DataFrame({
        'N':                N_values,
        'vector_time_mean': vec_time_means,
        'vector_time_std':  vec_time_stds,
        'vector_mem_mean':  vec_mem_means,
        'vector_mem_std':   vec_mem_stds,
    #    'naive_time_mean':  naive_time_means,
    #    'naive_time_std':   naive_time_stds,
    #    'naive_mem_mean':   naive_mem_means,
    #    'naive_mem_std':    naive_mem_stds,
    })

    print("\nFinal benchmark results:")
    print(df.to_string(index=False))

    # save to CSV
    #out_csv = 'data/naive.csv'
    #df.to_csv(out_csv, index=False)
    #print(f"\nSaved results to {out_csv}")

    # Plot only the non-vectorised timing
    plt.figure(figsize=(8,5))
    #plt.errorbar(
    #    N_values, naive_time_means, yerr=naive_time_stds,
    #    fmt='o-', capsize=4
    #)
    plt.title(f"Average time for {n_steps} steps")
    plt.xlabel("Number of particles (N)")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()