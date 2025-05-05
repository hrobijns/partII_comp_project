import time
import tracemalloc

import numpy as np
import pandas as pd

from simulation import Simulation, Body
from quadtree import theta

def benchmark_one_step(N_values, seeds, space_size, theta_value):
    """
    For each N in N_values, runs a single symplectic step, measures:
      - time taken (seconds)
      - peak memory allocations (MB)
    averaged over the provided seeds, with sample std dev.
    Returns a pandas.DataFrame with columns:
      ['N','mean_time_s','std_time_s','mean_mem_MB','std_mem_MB']
    """
    records = []

    for N in N_values:
        times = []
        mems  = []

        for seed in seeds:
            np.random.seed(seed)
            positions = np.random.uniform(-space_size, space_size, size=(N, 2))
            bodies = [Body(position=pos, velocity=[0,0], charge=1.0) for pos in positions]
            sim = Simulation(bodies, space_size, theta_value)

            tracemalloc.start()
            t0 = time.perf_counter()
            # single step
            sim.step()
            t1 = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(t1 - t0)
            mems.append(peak / (1024**2))

        mean_t = np.mean(times)
        std_t  = np.std(times, ddof=1)
        mean_m = np.mean(mems)
        std_m  = np.std(mems, ddof=1)

        records.append({
            'N':           N,
            'mean_time_s': mean_t,
            'std_time_s':  std_t,
            'mean_mem_MB': mean_m,
            'std_mem_MB':  std_m,
        })

        print(
            f"N={N:5d} → time={mean_t:.4f}±{std_t:.4f}s, "
            f"mem={mean_m:.2f}±{std_m:.2f}MB"
        )

    return pd.DataFrame.from_records(records)

def main():
    # build sample N values
    lin = np.linspace(10, 1000, 10, dtype=int)[1:-1]
    explicit = np.array([10, 100, 1000], dtype=int)
    N_values = np.unique(np.concatenate([lin, explicit]))

    seeds      = [10, 41, 42, 55, 44]
    space_size = 5
    theta_val  = 0.3

    df = benchmark_one_step(N_values, seeds, space_size, theta_val)

    # save to CSV
    df.to_csv('data/BH1.csv', index=False)
    # print("\n► Saved results to benchmark_one_step.csv")

if __name__ == '__main__':
    main()

