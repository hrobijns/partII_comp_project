import numpy as np
from simulation import Particle, potential, potential_naive

# Define 5 particles explicitly
particles = [
    Particle(0.0, 0.0, 1.0),
    Particle(2.0, 0.0, -1.0),
    Particle(1.0, 1.0, 0.5),
    Particle(-1.0, -1.0, -0.5),
    Particle(-2.0, 2.0, 1.5)
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