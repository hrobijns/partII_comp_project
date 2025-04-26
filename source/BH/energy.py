import numpy as np
import matplotlib.pyplot as plt

from quadtree import k, soft
from simulation import Simulation, Body


def compute_total_energy(bodies):
    """
    Compute total energy (kinetic + potential) of the system.

    Parameters:
        bodies (list of Body): List of Body instances.

    Returns:
        float: Total energy.
    """
    # Kinetic energy: 1/2 m v^2 for each body
    KE = 0.5 * sum(b.mass * np.dot(b.velocity, b.velocity) for b in bodies)

    # Potential energy: sum over unique pairs i<j of k q_i q_j / r_ij
    U = 0.0
    N = len(bodies)
    for i in range(N):
        for j in range(i + 1, N):
            diff = bodies[i].position - bodies[j].position
            r = np.sqrt(np.dot(diff, diff) + soft**2)
            U -= k * bodies[i].charge * bodies[j].charge * np.log(r)

    return KE + U


def run_energy_conservation(theta_values, N, num_steps, space_size, seed=0):
    """
    Simulate dynamics and record percentage energy change over time for different theta.

    Parameters:
        theta_values (list of float): Opening angles to test.
        N (int): Number of bodies.
        num_steps (int): Number of simulation steps.
        space_size (float): Half-width of simulation domain.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Mapping theta -> array of percentage energy changes.
    """
    results = {}
    for theta in theta_values:
        # Initialize random bodies
        np.random.seed(seed)
        positions = np.random.uniform(-space_size, space_size, size=(N, 2))
        bodies = [Body(position=pos, velocity=[0, 0], charge=1.0) for pos in positions]

        # Initialize BH simulation
        sim = Simulation(bodies, space_size, theta)

        # Record initial energy
        E0 = compute_total_energy(sim.bodies)
        energies = [E0]

        # Run simulation steps
        for _ in range(num_steps):
            sim.step()
            energies.append(compute_total_energy(sim.bodies))

        energies = np.array(energies)
        pct_change = (energies - E0) / abs(E0) * 100.0
        results[theta] = pct_change
        print(theta)
    return results


def main():
    # Benchmark parameters
    theta_values = [0.1, 0.3, 0.5, 0.7]
    N = 30
    num_steps = 1000
    space_size = 10.0

    # Run energy conservation tests
    results = run_energy_conservation(theta_values, N, num_steps, space_size)

        # Step array for plotting
    steps = np.arange(num_steps + 1)

    # Plot percentage energy change
    plt.figure(figsize=(8, 6))
    for theta, pct in results.items():
        plt.plot(steps, pct, linestyle='-', label=fr'$\theta={theta:.2f}$')

    # Reference line at zero change
    plt.axhline(0.0, color='black', linestyle='--')

    plt.xlabel('Steps')
    plt.ylabel('Percentage Total Energy Change of System (%)')
    #plt.title('Energy Conservation vs. Time for Different Theta')
    plt.legend()
    plt.xlim(0,1000)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/theta_energy.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()