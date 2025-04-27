import numpy as np
import matplotlib.pyplot as plt
from simulation import Particle, potential, potential_naive

# Generate random particles
np.random.seed(0)
n_particles = 100
positions = np.random.uniform(-10, 10, size=(n_particles, 2))
charges = np.random.uniform(1, 1, size=n_particles)

# Compute exact potentials
particles_naive = [Particle(x, y, q) for (x, y), q in zip(positions, charges)]
for p in particles_naive:
    p.phi = 0.0
potential_naive(particles_naive)
phi_exact = np.array([p.phi for p in particles_naive])

# Test range of p values
p_values = range(1, 16)
errors = []

for p_order in p_values:
    # Reset particles for FMM each time
    particles_fmm = [Particle(x, y, q) for (x, y), q in zip(positions, charges)]
    for p_fmm in particles_fmm:
        p_fmm.phi = 0.0

    # Compute potentials using FMM
    potential(particles_fmm, tree_thresh=1, p_order=p_order)

    phi_fmm = np.array([particle.phi for particle in particles_fmm])

    # Calculate relative L2 error
    err = np.linalg.norm(phi_fmm - phi_exact) / np.linalg.norm(phi_exact)
    errors.append(err)

    print(f"p={p_order}, error={err:.6f}")

# Plot the errors clearly
plt.figure(figsize=(8,5))
plt.semilogy(p_values, errors, marker='o', linestyle='-', color='blue')
plt.xlabel('Expansion Order (p)')
plt.ylabel('Relative L2 Error')
plt.title('FMM Relative Error vs Expansion Order p')
plt.grid(True)
plt.tight_layout()
plt.show()