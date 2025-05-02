import time
import numpy as np
import matplotlib.pyplot as plt

from fmm import Particle, potential  

def random_particles(N, seed=None):
    """Generate N random particles in the unit square with random charges."""
    rng = np.random.RandomState(seed)
    xs = rng.rand(N)
    ys = rng.rand(N)
    qs = rng.randn(N)
    return [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]

def time_fmm(N, tree_thresh=2, nterms=5, seed=0):
    """Run a single FMM timing for N particles."""
    particles = random_particles(N, seed=seed)
    t0 = time.perf_counter()
    _ = potential(particles, tree_thresh=tree_thresh, nterms=nterms)
    return time.perf_counter() - t0

if __name__ == "__main__":
    Ns = np.linspace(10,1000,10).astype(int).tolist()
    seeds = [20, 32, 43, 54, 44]
    
    avg_times = []
    print(f"{'N':>8s}  {'avg time (s)':>12s}")
    for N in Ns:
        times = []
        for seed in seeds:
            t = time_fmm(N, tree_thresh=5, nterms=6, seed=seed)
            times.append(t)
        mean_t = np.mean(times)
        avg_times.append(mean_t)
        print(f"{N:8d}  {mean_t:12.4f}")
    
    # Plot on log–log axes of average times
    plt.figure()
    plt.plot(Ns, avg_times, marker='o')
    plt.xlabel('Number of particles $N$')
    plt.ylabel('Average FMM time (s) over 5 seeds')
    plt.title('Scaling of 2D FMM (averaged over random seeds)')
    plt.tight_layout()
    plt.show()

