import numpy as np
from simulation import Particle, potential, potential_naive

# Define more particles, scattered
np.random.seed(42)
particles = [
    Particle(x, y, 1.0) for x, y in np.random.uniform(-5, 5, (30, 2))
]

# Compute exact potentials (naive method)
particles_exact = [Particle(p.x, p.y, p.q) for p in particles]
for p in particles_exact:
    p.phi = 0.0
potential_naive(particles_exact)
phi_exact = np.array([p.phi for p in particles_exact])

# Test FMM convergence
print("Expansion order | Relative Error")
print("----------------|---------------")
for p_order in range(1, 10):
    particles_fmm = [Particle(p.x, p.y, p.q) for p in particles]
    for p in particles_fmm:
        p.phi = 0.0

    potential(particles_fmm, tree_thresh=1, p_order=p_order)
    phi_fmm = np.array([p.phi for p in particles_fmm])

    error = np.linalg.norm(phi_fmm - phi_exact) / np.linalg.norm(phi_exact)
    print(f"{p_order:^15} | {error:.6e}")