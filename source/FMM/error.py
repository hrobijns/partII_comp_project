import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy


from fmm import Particle, potential

def naive_potential(particles):
    """
    Compute Coulomb potential by direct all-to-all summation.
    """
    N = len(particles)
    phi = np.zeros(N)
    for i, p in enumerate(particles):
        for j, s in enumerate(particles):
            if i == j:
                continue
            # softening
            r = np.hypot(p.x - s.x, p.y - s.y) 
            phi[i] += s.q * np.log(r)
    return phi

def main():
    np.random.seed(42)
    N = 100

    # generate random particles in [0,1]^2 with random charges
    particles = [
        Particle(x=np.random.rand(),
                 y=np.random.rand(),
                 charge=np.random.randn())
        for _ in range(N)
    ]
    particlesDS = deepcopy(particles)

    # compute "exact" reference
    phi_ref = naive_potential(particlesDS)

    orders = list(range(1, 11))
    errors = []

    for n in orders:
        # reset potentials
        for p in particles:
            p.phi = 0.0

        # compute via your FMM
        phi_fmm = potential(particles, tree_thresh=2, nterms=n)

        # relative L2 error (%) 
        rel_err = np.linalg.norm(phi_fmm - phi_ref) / np.linalg.norm(phi_ref) * 100
        errors.append(rel_err)
        print (f'p = {n}, relative error = {rel_err} %')

    # print results
    df = pd.DataFrame({
        'Expansion Order': orders,
        'Relative Error (%)': errors
    })
    print(df.to_string(index=False))

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(orders, errors, 'o-')
    plt.xlabel('Expansion Order')
    plt.ylabel('Relative Error (%)')
    #plt.title('FMM vs Naive Direct: Relative Error vs Expansion Order')
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    #plt.savefig('figures/FMMerror.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
