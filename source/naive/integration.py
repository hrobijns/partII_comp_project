import numpy as np
import matplotlib.pyplot as plt
import copy

from simulation import SimulationVectorised, k, soft, dt

def compute_energy(pos, vel, charge, mass):
    """Compute total energy (kinetic + potential) for the 2D Coulomb (logarithmic) system."""
    # Kinetic energy
    KE = 0.5 * np.sum(mass[:, None] * vel**2)

    # Potential energy: U - k * q_i * q_j * ln(r)
    N = pos.shape[0]
    PE = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = pos[i] - pos[j]
            r = np.sqrt(np.dot(diff, diff) + soft**2)
            PE -= k * charge[i] * charge[j] * np.log(r)
    return KE + PE


def euler_step(pos, vel, charge, mass):
    """
    Perform one explicit Euler integration step under 2D Coulomb forces.
    """
    # Vectorised force computation: F = k * q_i * q_j * diff / (r^2 + soft^2)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(diff * diff, axis=2) + soft**2
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    qq = np.outer(charge, charge)
    Fmat = k * qq[:, :, None] * diff * inv_r2[:, :, None]
    force = np.sum(Fmat, axis=1)

    # Euler update
    vel_new = vel + (force / mass[:, None]) * dt
    pos_new = pos + vel * dt
    return pos_new, vel_new


def main():
    # Number of bodies
    N = 10
    np.random.seed(42)

    # Random initial conditions
    pos0 = np.random.uniform(-1.0, 1.0, (N, 2))
    vel0 = np.random.normal(0.0, 0.1, (N, 2))
    charge = np.random.uniform(-1.0, 1.0, N)
    mass   = np.random.uniform(0.5, 1.5, N)

    # Prepare Verlet simulation
    sim_v = SimulationVectorised(pos0.copy(), vel0.copy(), charge, mass)
    sim_v.compute_forces()

    # Prepare Euler state
    pos_e = pos0.copy()
    vel_e = vel0.copy()

    # Number of steps
    steps = 1000
    idx = np.arange(steps)

    # Storage for energy error
    err_v = np.zeros(steps)
    err_e = np.zeros(steps)

    # Initial energies
    E0_v = compute_energy(sim_v.pos, sim_v.vel, charge, mass)
    E0_e = compute_energy(pos_e, vel_e, charge, mass)

    # Time evolution
    for n in range(steps):
        # Verlet step
        sim_v.step()
        Ev = compute_energy(sim_v.pos, sim_v.vel, charge, mass)
        err_v[n] = (Ev - E0_v) / abs(E0_v) * 100

        # Euler step
        pos_e, vel_e = euler_step(pos_e, vel_e, charge, mass)
        Ee = compute_energy(pos_e, vel_e, charge, mass)
        err_e[n] = (Ee - E0_e) / abs(E0_e) * 100
        print(n)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(idx, err_v, label='kick-drift-kick')
    plt.plot(idx, err_e, label='Euler')
    plt.axhline(0, color='black', linestyle=':')
    plt.xlabel('Step')
    plt.ylabel('Percentage Total Energy Change of System (%)')
    #plt.title(f'Energy Conservation: {N}-Body 2D Coulomb System')
    plt.legend()
    plt.xlim(0,1000)
    plt.tight_layout()
    plt.savefig('figures/integration.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
