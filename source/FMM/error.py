import numpy as np
import matplotlib.pyplot as plt
from fmm import Particle, potential, potentialDS

# 1) build a random test problem
np.random.seed(42)
N = 10
points  = np.random.rand(N, 2)       # random (x,y) in [0,1]^2
charges = np.random.randn(N)         # random charges ~ N(0,1)

orders = list(range(1, 11))
rel_errors = []

for n in orders:
    # --- FMM evaluation with expansion order = nterms = n ---
    particles_fmm = [Particle(x, y, q) for (x, y), q in zip(points, charges)]
    potential(particles_fmm, nterms=n)
    phi_fmm = np.array([p.phi for p in particles_fmm])

    # --- direct‐sum for “ground truth” ---
    particles_ds = [Particle(x, y, q) for (x, y), q in zip(points, charges)]
    phi_ds = potentialDS(particles_ds)

    # compute max relative error
    rel_err = np.max(np.abs(phi_fmm - phi_ds) / (np.abs(phi_ds)))
    rel_errors.append(rel_err)

# Plot
plt.plot(orders, rel_errors, marker='o')
plt.xlabel('Expansion Order (nterms)')
plt.ylabel('Max Relative Error')
plt.title('FMM Relative Error vs Expansion Order')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()