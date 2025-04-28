import time
import numpy as np
import matplotlib.pyplot as plt
from fmm import potential, Particle  # make sure this imports your improved FMM

def make_random_particles(N, seed=None):
    rng = np.random.default_rng(seed)
    xs = rng.random(N)
    ys = rng.random(N)
    return [Particle(x, y, charge=1.0) for x, y in zip(xs, ys)]

def time_fmm(N, ntrials=3, tree_thresh=2, nterms=3):
    """Time FMM for N particles, averaged over ntrials runs."""
    particles = make_random_particles(N, seed=42)
    # warm-up
    potential(particles, tree_thresh=tree_thresh, nterms=nterms)
    times = []
    for _ in range(ntrials):
        for p in particles:
            p.phi = 0.0
        t0 = time.perf_counter()
        potential(particles, tree_thresh=tree_thresh, nterms=nterms)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

if __name__ == "__main__":
    Ns = [100, 200, 400, 800, 1600, 3200, 6400]
    timings = []
    for N in Ns:
        dt = time_fmm(N)
        print(f"N={N:6d} → {dt:.6f} s")
        timings.append(dt)

    plt.figure()
    plt.plot(Ns, timings, marker='o')
    plt.xlabel("Number of particles $N$")
    plt.ylabel("FMM time (s)")
    plt.title("FMM2D Scaling")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()