import numpy as np
from kernels import multipole, M2L
from simulation import Particle, potential_naive

# One isolated source particle
source_particles = [Particle(0.0, 0.0, 1.0)]

# Multipole expansion around source (0,0)
source_center = (0.0, 0.0)
p_order = 15
M = multipole(source_particles, center=source_center, p=p_order)

# Shift multipole expansion to local expansion around target at (5,0) (well separated)
target_center = (5.0, 0.0)
z0 = complex(source_center[0], source_center[1]) - complex(target_center[0], target_center[1])
L = M2L(M, z0)

# Evaluate local expansion at particle at (5.1, 0.1)
eval_point = complex(5.1, 0.1) - complex(target_center[0], target_center[1])
phi_local = np.real(sum(L[j] * eval_point**j for j in range(len(L))))

# Exact calculation for comparison
eval_particle = Particle(5.1, 0.1, 1.0)
potential_naive([eval_particle], source_particles)
phi_exact = eval_particle.phi

print(f"Local Expansion potential: {phi_local:.6f}")
print(f"Exact potential:           {phi_exact:.6f}")
print(f"Relative Error:            {abs(phi_local - phi_exact)/abs(phi_exact):.6e}")