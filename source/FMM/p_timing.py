
import time
import numpy as np
import matplotlib.pyplot as plt

from fmm import Particle, potential

def make_random_particles(n):
    """
    Generate n particles uniformly in [0,1]^2 with random charges ±1.
    """
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    qs = np.random.choice([-1.0, +1.0], size=n)
    return [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]

def time_fmm_step(particles, tree_thresh, nterms):
    """
    Build & run FMM once and return elapsed time (in seconds).
    """
    t0 = time.perf_counter()
    _ = potential(particles, tree_thresh=tree_thresh, nterms=nterms)
    t1 = time.perf_counter()
    return t1 - t0

def main():
    # fixed seed for reproducibility
    np.random.seed(1234)

    # parameters
    N_bodies   = 100
    tree_thresh = 5            # max pts per leaf
    orders     = [2,4,6,8,10,12]

    # generate particles
    particles = make_random_particles(N_bodies)

    # time each expansion order
    times = []
    print(f"{'nterms':>6s}   {'time (s)':>10s}")
    print("-" * 20)
    for p in orders:
        dt = time_fmm_step(particles, tree_thresh, p)
        times.append(dt)
        print(f"{p:6d}   {dt:10.6f}")

    # now plot
    plt.figure()
    plt.plot(orders, times, marker='o')
    plt.xlabel('Expansion Order')
    plt.ylabel('Computation Time (s)')
    #plt.title('FMM Computation Time vs Expansion Order')
    plt.grid(True)
    plt.tight_layout()

    # save and show
    #plt.savefig("figures/FMMtiming_p", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
