import time
import numpy as np
import matplotlib.pyplot as plt

from FMM import potential, Particle

def run_benchmark(ns, nterms=5, tree_thresh=None, bbox=None, boundary='wall'):
    """
    For each N in ns, generate N random particles and measure
    the time to compute all-to-all potential via FMM.
    Returns a list of elapsed times.
    """
    times = []
    for N in ns:
        # generate random particles in [0,1]^2 with random charges ~N(0,1)
        xs = np.random.rand(N)
        ys = np.random.rand(N)
        qs = np.random.randn(N)
        particles = [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]

        start = time.perf_counter()
        potential(particles,
                  bbox=bbox,
                  tree_thresh=tree_thresh,
                  nterms=nterms,
                  boundary=boundary)
        elapsed = time.perf_counter() - start

        print(f"N={N:6d} → {elapsed:.4f} s")
        times.append(elapsed)
    return times

if __name__ == "__main__":
    # list of particle counts to test
    ns = [100, 200, 500, 1000, 2000, 5000, 10000]

    print("Benchmarking FMM:")
    times = run_benchmark(ns, nterms=5, tree_thresh=50)

    # Plotting
    plt.figure(figsize=(6,4))
    plt.plot(ns, times, marker='o')
    plt.xlabel("Number of particles $N$")
    plt.ylabel("Elapsed time (s)")
    plt.title("FMM runtime scaling")
    plt.grid(True)
    plt.tight_layout()
    plt.show()