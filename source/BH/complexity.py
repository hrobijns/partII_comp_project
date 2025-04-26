import time
import tracemalloc

import numpy as np
import pandas as pd

from simulation import Simulation, Body
from quadtree import theta

def benchmark_10_steps(N_values, seeds, space_size, theta_value):
    """
    For each N in N_values, runs 10 symplectic steps, measures:
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
            # reproducibility
            np.random.seed(seed)
            positions = np.random.uniform(-space_size, space_size, size=(N, 2))
            bodies = [Body(position=pos, velocity=[0,0], charge=1.0) for pos in positions]
            sim = Simulation(bodies, space_size, theta_value)

            tracemalloc.start()
            t0 = time.perf_counter()
            for _ in range(1):
                sim.step()
            t1 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(t1 - t0)
            mems.append(peak / (1024**2))

        records.append({
            'N':           N,
            'mean_time_s': np.mean(times),
            'std_time_s':  np.std(times, ddof=1),
            'mean_mem_MB': np.mean(mems),
            'std_mem_MB':  np.std(mems,  ddof=1),
        })

        print(f"N={N:5d} → time={records[-1]['mean_time_s']:.4f}±{records[-1]['std_time_s']:.4f}s, "
              f"mem={records[-1]['mean_mem_MB']:.2f}±{records[-1]['std_mem_MB']:.2f}MB")

    return pd.DataFrame.from_records(records)


def main():
    # 1) build 16 in-between, plus 4 explicit → 20 total
    lin = np.linspace(10, 1000, 18, dtype=int)[1:-1]
    explicit = np.array([10, 100, 1000], dtype=int)
    N_values = np.unique(np.concatenate([lin, explicit]))

    seeds      = [24,32,643,64,25,51,523,63,85,245]
    space_size = 30
    theta_val  = theta

    df = benchmark_10_steps(N_values, seeds, space_size, theta_val)

    # save to CSV
    df.to_csv('BH.csv', index=False)
    print("\n► Saved results to benchmark_results.csv")


if __name__ == '__main__':
    main()
