import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from simulation import Body, Simulation, potential_naive, force_naive

# Coulomb constant (as in simulation.py)
k = 1.0

def compute_total_energy(bodies):
    """½∑m v² + ½∑φ  (φ must already be set on each body)."""
    KE = 0.5 * sum(b.mass * np.dot(b.v, b.v) for b in bodies)
    U  = 0.5 * sum(b.phi for b in bodies)
    return KE + U

def compute_naive_forces_and_potential(bodies):
    """
    Reset φ, fx, fy; then call the two direct‐sum routines
    to set bodies[].phi and bodies[].fx/fy.
    Returns list of force‐vectors like Simulation.compute_forces().
    """
    for b in bodies:
        b.phi = b.fx = b.fy = 0.0
    # set φ via direct‐sum
    potential_naive(bodies)
    # set fx, fy via direct‐sum
    force_naive(bodies)
    return [np.array((b.fx, b.fy), dtype=float) for b in bodies]

def step_naive(bodies, dt):
    """
    One leapfrog step using direct‐sum forces.
    """
    # half-kick
    F = compute_naive_forces_and_potential(bodies)
    for b, f in zip(bodies, F):
        b.v += 0.5 * dt * f / b.mass

    # drift
    for b in bodies:
        b.x += dt * b.v[0]
        b.y += dt * b.v[1]
        b.pos = (b.x, b.y)

    # half-kick
    F = compute_naive_forces_and_potential(bodies)
    for b, f in zip(bodies, F):
        b.v += 0.5 * dt * f / b.mass

def run_comparison(n_bodies, dt, steps, p_orders):
    """
    • Build one shared initial snapshot.
    • Run a brute‐force “naive” simulation once → P_naive(t).
    • For each p in p_orders, run an FMM sim → P_fmm(t).
    • Compute ΔP(t) = P_fmm(t) – P_naive(t) to isolate FMM truncation error.
    Returns a dict p→ΔP_list.
    """
    print(f"Initializing comparison: {n_bodies} bodies, dt={dt}, {steps} steps, p_orders={p_orders}")
    # 1) make the master initial condition
    np.random.seed(42)
    master = [
        Body(
            position=np.random.uniform(-10, 10, 2),
            velocity=np.random.uniform(-2, 2, 2),
            mass    =np.random.uniform(0.1, 1.0)
        )
        for _ in range(n_bodies)
    ]

    # 2) brute-force run
    print("Starting brute-force (naive) run...")
    bodies_naive = copy.deepcopy(master)
    compute_naive_forces_and_potential(bodies_naive)
    E0 = compute_total_energy(bodies_naive)
    P_naive = []
    for i in range(steps):
        step_naive(bodies_naive, dt)
        compute_naive_forces_and_potential(bodies_naive)  # update φ
        En = compute_total_energy(bodies_naive)
        err = (En - E0) / abs(E0) * 100.0
        P_naive.append(err)
        print(f"  Naive step {i+1}/{steps}: energy error = {err:.6f}%")

    # 3) FMM runs & ΔP
    deltaP = {}
    for p in p_orders:
        print(f"\nStarting FMM run with expansion order p = {p}...")
        bodies_fmm = copy.deepcopy(master)
        sim = Simulation(bodies_fmm, dt=dt, nterms=p)
        sim.compute_forces()  # initialize φ
        Pf = []
        for i in range(steps):
            sim.step()
            sim.compute_forces()  # update φ
            Ef = compute_total_energy(bodies_fmm)
            err_fmm = (Ef - E0) / abs(E0) * 100.0
            Pf.append(err_fmm)
            delta = err_fmm - P_naive[i]
            print(f"  FMM p={p} step {i+1}/{steps}: energy error = {err_fmm:.6f}%, ΔP = {delta:.6f}%")
        Δ = [f - n for f, n in zip(Pf, P_naive)]
        deltaP[p] = Δ

    print("\nAll simulations complete.")
    return deltaP

def plot_truncation_error(deltaP):
    plt.figure(figsize=(10,6))
    for p, Δ in deltaP.items():
        plt.plot(Δ, label=f'p = {p}')
    plt.axhline(0, color='k', linestyle=':', linewidth=1.5,
                label='No FMM error')
    plt.xlabel('Timestep')
    plt.ylabel('Δ(% Energy error) = FMM − brute-force')
    plt.title('Isolated FMM Truncation Error vs. Expansion Order')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    N        = 100
    DT       = 0.01
    STEPS    = 1
    P_ORDERS = [3, 5, 7, 9, 12]

    deltaP = run_comparison(N, DT, STEPS, P_ORDERS)
    plot_truncation_error(deltaP)
