import math
from kernels import multipole, M2L

# 1) Single source at the origin
class FakeP: 
    def __init__(self,x,y,q): 
        self.x,self.y,self.q = x,y,q

src = FakeP(0.0, 0.0, q=1.0)

# 2) Build its multipole around (0,0)
p = 10
M = multipole([src], center=(0.0, 0.0), p=p)

# 3) Translate to local about target-center at (1,0)
z0 = complex(0,0) - complex(1,0)   # = -1+0j
L  = M2L(M, z0)

# 4) Evaluate local expansion at z = 0.8+0.2j  (within |z0|=1)
z = complex(0.8, 0.2) - complex(1, 0)   # shift so expansion is in (z−1,0)
phi_loc = sum(L[j] * z**j for j in range(p+1)).real

# 5) Compare to exact log-potential
z_exact = complex(0.8,0.2) - complex(src.x, src.y)
phi_exact = src.q * math.log(abs(z_exact))

print(f"φ_loc   = {phi_loc:.6f}")
print(f"φ_exact = {phi_exact:.6f}")
print("error   =", abs(phi_loc - phi_exact))