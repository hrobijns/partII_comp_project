import time
import tracemalloc

import numpy as np
import pandas as pd

from fmm import Particle, potential

def random_particles(N, seed=None):
    """Generate N random particles in the unit square with random charges."""
    rng = np.random.RandomState(seed)
    xs = rng.rand(N)
    ys = rng.rand(N)
    qs = rng.randn(N)
    return [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]

def benchmark_fmm(N_values, seeds, tree_thresh, nterms):
    """
    For each N in N_values, runs a single FMM evaluation, measures:
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
            particles = random_particles(N, seed=seed)

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = potential(particles, tree_thresh=tree_thresh, nterms=nterms)
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
    # 1) generate 10 linearly spaced values between 10 and 1000
    #base = np.linspace(10, 1000, 10, dtype=int)
    # 2) explicitly include 10, 100, 1000
    #explicit = np.array([10, 100, 1000], dtype=int)
    # 3) merge and dedupe, then sort
    #N_values = np.unique(np.concatenate([base, explicit]))
    N_values = [10,100,1000,10000,100000]

    seeds      = [10, 41, 42]
    tree_thresh = 15
    nterms      = 3

    df = benchmark_fmm(N_values, seeds, tree_thresh, nterms)

    out_csv = 'data/FMM.csv'
    #df.to_csv(out_csv, index=False)
    print(f"\n► Saved results to {out_csv}")

    # optional plotting
    try:
        import matplotlib.pyplot as plt
        plt.errorbar(df['N'], df['mean_time_s'], yerr=df['std_time_s'],
                     fmt='o-', capsize=3)
        plt.xlabel('Number of particles $N$')
        plt.ylabel('Average time (s)')
        plt.title('FMM time scaling')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()

