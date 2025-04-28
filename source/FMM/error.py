import numpy as np
import matplotlib.pyplot as plt
from quadtree1 import QuadTree

# Adjust these imports to point to your FMM implementation
from fmm1 import potentialFMM  # function that runs FMM on a QuadTree and populates p.phi


def direct_potential(particles):
    """
    Compute direct all-to-all Coulomb potentials using brute force.
    Returns an array of potentials (phi) for each particle.
    """
    N = len(particles)
    phi = np.zeros(N)
    for i, pi in enumerate(particles):
        for j, pj in enumerate(particles):
            if i == j:
                continue
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            r = np.hypot(dx, dy)
            # avoid singularity
            if r == 0:
                continue
            phi[i] -= pi.q * np.log(r)
    return phi


def make_random_particles(N, domain=(0, 1, 0, 1), seed=None):
    """
    Generate N particles with random positions in the given rectangular domain
    and random charges in [-1,1].
    Each particle gets attributes: x, y, q, phi, pos (for FMM evaluation), and position (for quadtree insertion).
    """
    if seed is not None:
        np.random.seed(seed)
    x_min, x_max, y_min, y_max = domain
    xs = np.random.uniform(x_min, x_max, N)
    ys = np.random.uniform(y_min, y_max, N)
    qs = np.random.uniform(-1, 1, N)
    particles = []
    for x, y, q in zip(xs, ys, qs):
        # Simple object to hold attributes
        p = type('P', (), {})()
        p.x, p.y = x, y
        p.q = q
        p.phi = 0.0
        p.position = (x, y)
        particles.append(p)
    return particles


def compute_error(N=100, tree_thresh=2, nterms_list=None, domain=(0,1,0,1), seed=42):
    """
    Runs FMM and direct for different expansion orders and computes RMS error.
    Returns two arrays: nterms_list and rms_errors.
    """
    if nterms_list is None:
        nterms_list = list(range(1, 11))

    # generate test set
    particles = make_random_particles(N, domain=domain, seed=seed)
    # compute reference solution
    phi_direct = direct_potential(particles)

    rms_errors = []
    for nterms in nterms_list:
        # reset particle potentials
        for p in particles:
            p.phi = 0

        # build tree and run FMM
        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        boundary = (min(xs), min(ys), max(xs), max(ys))
        tree = QuadTree(particles, boundary=boundary, max_points=tree_thresh)
        potentialFMM(tree, nterms=nterms)

        # collect FMM result
        phi_fmm = np.array([p.phi for p in particles])

        # compute RMS error
        err = np.sqrt(np.mean((phi_fmm - phi_direct)**2))
        rms_errors.append(err)
        print(f"nterms={nterms:2d}: RMS error = {err:.3e}")

    return np.array(nterms_list), np.array(rms_errors)


if __name__ == '__main__':
    # parameters
    N = 200            # number of particles
    tree_thresh = 2    # max bodies per leaf
    nterms_list = list(range(1, 11))

    # run error analysis
    nterms_arr, errors = compute_error(N=N,
                                      tree_thresh=tree_thresh,
                                      nterms_list=nterms_list)

    # plot
    plt.figure()
    plt.semilogy(nterms_arr, errors, marker='o')
    plt.xlabel('Expansion order (nterms)')
    plt.ylabel('RMS error')
    plt.title('FMM error vs. expansion order')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
