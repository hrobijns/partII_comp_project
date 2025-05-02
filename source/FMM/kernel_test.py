import numpy as np
from kernels import multipole, M2M, M2L, L2L

class BareParticle:
    """Minimal particle-like class for kernel tests."""
    def __init__(self, x, y, q):
        # Ensure compatibility with kernels expecting .x and .y
        self.x = x
        self.y = y
        self.position = (x, y)
        self.q = q


def random_cluster(n, center=(0.0, 0.0), scale=0.1, seed=None):
    """
    Generate n random BareParticle objects around a given center.
    """
    rng = np.random.RandomState(seed)
    pts = np.array(center) + scale * (rng.rand(n, 2) - 0.5)
    qs = rng.randn(n)
    return [BareParticle(x, y, q) for (x, y), q in zip(pts, qs)]


def test_m2m():
    """Test translation invariance of multipole expansion via M2M."""
    for nterms in [3, 5, 8]:
        particles = random_cluster(20, center=(0.2, 0.3), scale=0.2, seed=1234)
        c1 = (0.2, 0.3)
        c2 = (0.5, 0.7)
        M1 = multipole(particles, center=c1, nterms=nterms)
        M2 = multipole(particles, center=c2, nterms=nterms)
        z0 = complex(*c1) - complex(*c2)
        M1_shifted = M2M(M1, z0)
        assert np.allclose(M1_shifted, M2, atol=1e-8), f"M2M failed for nterms={nterms}"  
    print("✅ M2M translation invariance: PASSED")


def test_m2l():
    """Test local expansion from multipole via M2L against direct sum."""
    for nterms in [4, 6, 8]:
        src_center = np.array([0.2, 0.2])
        tgt_center = np.array([0.8, 0.8])
        sources = random_cluster(30, center=src_center, scale=0.1, seed=5678)
        M_src = multipole(sources, center=tuple(src_center), nterms=nterms)
        z1 = complex(*tgt_center) - complex(*src_center)
        L_tgt = M2L(M_src, z1)
        rng = np.random.RandomState(0)
        for _ in range(5):
            dz = 0.05 * (rng.rand(2) - 0.5)
            z_test = complex(*(tgt_center + dz))
            phi_local = sum(L_tgt[k] * (z_test - complex(*tgt_center))**k for k in range(len(L_tgt)))
            phi_direct = sum(s.q * np.log(abs(z_test - complex(*s.position))) for s in sources)
            assert np.isclose(phi_local.real, phi_direct.real, atol=1e-6), "M2L local expansion mismatch"
    print("✅ M2L local expansion vs direct sum: PASSED")


def test_l2l():
    """Test translation invariance of local-to-local (L2L)."""
    nterms = 6
    rng = np.random.RandomState(42)
    L_parent = rng.randn(nterms+1) + 1j*rng.randn(nterms+1)
    c_parent = complex(0.5, 0.5)
    z0 = complex(0.1, -0.05)
    c_child = c_parent + z0
    L_child = L2L(L_parent, z0)
    for _ in range(5):
        dz = 0.02 * (rng.rand(2) - 0.5)
        z_test = c_child + (dz[0] + 1j*dz[1])
        phi_parent = sum(L_parent[k] * (z_test - c_parent)**k for k in range(len(L_parent)))
        phi_child = sum(L_child[k] * (z_test - c_child)**k for k in range(len(L_child)))
        assert np.allclose(phi_parent, phi_child, atol=1e-8), "L2L translation failed"
    print("✅ L2L translation invariance: PASSED")

if __name__ == '__main__':
    try:
        test_m2m()
        test_m2l()
        test_l2l()
    except AssertionError as e:
        print(f"❌ Test failed: {e}")