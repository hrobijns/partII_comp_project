import numpy as np
import matplotlib.pyplot as plt

from FMM import multipole, _convert_oi, Particle

# 1) Build a compact cluster of N=100 unit‐charges inside a radius 0.5 disk
np.random.seed(0)
N = 100
angles = 2*np.pi*np.random.rand(N)
radii  = np.sqrt(np.random.rand(N))*0.5
xs = radii * np.cos(angles)
ys = radii * np.sin(angles)
qs = np.ones(N)

particles = [Particle(x, y, q) for x, y, q in zip(xs, ys, qs)]
center = (0.0, 0.0)       # expansion center at cluster centroid

# 2) Choose a far-field evaluation point
zeval = 5.0 + 0.0j       # 5 units away along the x-axis

# 3) Compute the “ground-truth” potential at zeval by direct sum
phi_exact = sum(
    -p.q * np.log(abs(zeval - complex(p.x, p.y)))
    for p in particles
)

# 4) Build one big multipole expansion up to order K and then slice it
K = 12
all_coeffs = multipole(particles, center=center, nterms=K)

orders = np.arange(1, K+1)
errors = []

for k in orders:
    # take only the first k terms of the outer expansion
    outer_k = all_coeffs[: k+1]

    # shift outer → inner about zeval
    z0 = zeval - complex(*center)
    inner_k = _convert_oi(outer_k, z0)

    # evaluate the inner expansion at z = zeval:
    #   φ ≈ −Re[ ∑_{l=0}^k inner_k[l] · (z−zeval)^l ]
    # but at z=zeval the higher l>0 terms vanish, so φ≈−Re(inner_k[0])
    phi_fmm = -np.real(inner_k[0])

    rel_err = abs((phi_fmm - phi_exact) / phi_exact)
    errors.append(rel_err)
    print(f"order={k:2d}  rel-error = {rel_err:.2e}")

# 5) Plot
plt.figure(figsize=(6,4))
plt.semilogy(orders, errors, 'o-')
plt.xlabel("Multipole expansion order $k$")
plt.ylabel("Relative error")
plt.title("Pure multipole ∴ far-field error ∼ $O(1/R^{k+1})$")
plt.grid(True, which='both')
plt.tight_layout()
plt.show()