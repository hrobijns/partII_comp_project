import numpy as np
from kernels import multipole, M2M, M2L, L2L

class Source:
    def __init__(self, x, y, q):
        self.x = x
        self.y = y
        self.q = q


def direct_potential(sources, z):
    """Compute the exact potential at point z due to all sources."""
    return -sum(s.q * np.log(np.abs(z - (s.x + 1j*s.y))) for s in sources)


def eval_multipole(coeffs, z, z0):
    """Evaluate multipole expansion at z for center z0, returning a real potential."""
    w = z - z0
    # use log(|w|) so there's no imaginary branch‐cut residue
    phi = -coeffs[0] * np.log(np.abs(w))
    for k in range(1, len(coeffs)):
        # only the real part of each term contributes to the physical potential
        phi -= np.real(coeffs[k] * w**(-k))
    return phi


def eval_local(inner, z, z0):
    """Evaluate local expansion at z for center z0."""
    w = z - z0
    phi = inner[0]
    for l in range(1, len(inner)):
        phi += inner[l] * w**l
    return phi


def test_multipole_convergence():
    np.random.seed(1)
    # random sources inside radius 0.5
    N = 8
    rs = 0.5
    angles = np.random.rand(N) * 2*np.pi
    radii = np.sqrt(np.random.rand(N)) * rs
    sources = [Source(radii[i]*np.cos(angles[i]), radii[i]*np.sin(angles[i]), q=np.random.randn()) for i in range(N)]
    center0 = 0+0j
    # target points on circle of radius 2
    R = 2.0
    thetas = np.linspace(0, 2*np.pi, 16, endpoint=False)
    targets = [R * np.exp(1j*theta) for theta in thetas]

    print("Multipole expansion convergence (max error) vs p:")
    print("p, error")
    for p in range(1, 11):
        coeffs = multipole(sources, center=(0, 0), nterms=p)
        errs = []
        for z in targets:
            errs.append(abs(direct_potential(sources, z) - eval_multipole(coeffs, z, center0)))
        print(f"{p}, {max(errs):.3e}")


def test_M2M_translation():
    np.random.seed(2)
    N = 8
    rs = 0.5
    angles = np.random.rand(N) * 2*np.pi
    radii = np.sqrt(np.random.rand(N)) * rs
    sources = [Source(radii[i]*np.cos(angles[i]), radii[i]*np.sin(angles[i]), q=np.random.randn()) for i in range(N)]
    center0 = (0, 0)
    center1 = (1.0, -0.3)
    z0 = complex(center1[0] - center0[0], center1[1] - center0[1])

    print("\nM2M translation error vs p:")
    print("p, rel_error")
    for p in range(1, 11):
        c0 = multipole(sources, center=center0, nterms=p)
        c1 = multipole(sources, center=center1, nterms=p)
        c1_trans = M2M(c0, z0)
        rel_err = np.linalg.norm(c1_trans - c1) / np.linalg.norm(c1)
        print(f"{p}, {rel_err:.3e}")


def test_M2L_and_L2L():
    np.random.seed(3)
    N = 8
    rs = 0.5
    angles = np.random.rand(N) * 2*np.pi
    radii = np.sqrt(np.random.rand(N)) * rs
    sources = [Source(radii[i]*np.cos(angles[i]), radii[i]*np.sin(angles[i]), q=np.random.randn()) for i in range(N)]
    # source cluster center
    cs = (0, 0)
    # two target centers
    ct0 = (2.0, 0.0)
    ct1 = (2.5, 0.5)
    z0 = complex(ct0[0] - cs[0], ct0[1] - cs[1])
    dz = complex(ct1[0] - ct0[0], ct1[1] - ct0[1])

    print("\nM2L and L2L translation consistency vs p:")
    print("p, rel_error")
    for p in range(1, 11):
        c = multipole(sources, center=cs, nterms=p)
        inner0 = M2L(c, z0)
        inner_direct = M2L(c, z0 + dz)
        inner_shifted = L2L(inner0, dz)
        rel_err = np.linalg.norm(inner_shifted - inner_direct) / np.linalg.norm(inner_direct)
        print(f"{p}, {rel_err:.3e}")

if __name__ == "__main__":
    test_multipole_convergence()
    test_M2M_translation()
    test_M2L_and_L2L()