import numpy as np
import matplotlib.pyplot as plt

from FMM import potential, potentialDS, Particle

def compute_error(xs, ys, qs, nterms_list, tree_thresh=50, bbox=None, boundary='wall'):
    # Ground-truth via direct sum
    particles_ds = [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]
    phi_ds = potentialDS(particles_ds)

    errors = []
    for nterms in nterms_list:
        # Fresh particles for each FMM run
        particles_fmm = [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]
        potential(particles_fmm,
                  nterms=nterms,
                  tree_thresh=tree_thresh,
                  bbox=bbox,
                  boundary=boundary)
        phi_fmm = np.array([p.phi for p in particles_fmm])

        # relative L2 error
        err = np.linalg.norm(phi_fmm - phi_ds) / np.linalg.norm(phi_ds)
        print(f"nterms={nterms:2d} → rel L2 error = {err:.3e}")
        errors.append(err)
    return errors

if __name__ == "__main__":
    # reproducible random set
    np.random.seed(42)
    N = 100
    xs = np.random.rand(N)
    ys = np.random.rand(N)
    qs = np.random.randn(N)

    nterms_list = list(range(1, 11))  # try orders 1 through 10
    print("Computing relative errors:")
    errors = compute_error(xs, ys, qs, nterms_list, tree_thresh=2)

    # plot on a semilog-y scale
    plt.figure(figsize=(6,4))
    plt.plot(nterms_list, errors, marker='o')
    plt.xlabel("Multipole expansion order")
    plt.ylabel("Relative $L^2$ error")
    plt.title("FMM accuracy vs expansion order")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()