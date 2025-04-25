import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Barnes‐Hut & Naive imports stay the same
from BH.barneshut import Simulation as BarnesHutSimulation, Body as BarnesHutBody
from source.naive.naivegravity import Simulation as NaiveSimulation, Body as NaiveBody

# **FMM** — import from *working.py* instead of FMM.py
from source.FMM.working import Simulation as FastMultipoleSimulation, Body as FastMultipoleBody

# Constants
SIMULATION_STEPS = 1
N_MIN = 10   # Minimum number of bodies
N_MAX = 1000  # Maximum number of bodies
NUM_TRIALS = 5

# Shared RNG seed strategy
def run_barnes_hut_simulation(N, seed):
    np.random.seed(seed)
    bodies = [
        BarnesHutBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    sim = BarnesHutSimulation(bodies, space_size=2e11, theta=0.5)
    t0 = time.time()
    for _ in range(SIMULATION_STEPS):
        sim.move()
    return time.time() - t0

def run_naive_simulation(N, seed):
    np.random.seed(seed)
    bodies = [
        NaiveBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    sim = NaiveSimulation(bodies)
    t0 = time.time()
    for _ in range(SIMULATION_STEPS):
        sim.move()
    return time.time() - t0

def run_fast_multipole_simulation(N, seed, dt=0.01, nterms=4):
    np.random.seed(seed)
    bodies = [
        FastMultipoleBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    # note: working.Simulation.step() is your FMM integrator
    sim = FastMultipoleSimulation(bodies, dt=dt, nterms=nterms)
    t0 = time.time()
    for _ in range(SIMULATION_STEPS):
        sim.step()
    return time.time() - t0

def compare_computation_times():
    Ns = np.logspace(np.log10(N_MIN), np.log10(N_MAX), num=50, dtype=int)
    bh_times = []; nv_times = []; fmm_times = []
    bh_err = []; nv_err = []; fmm_err = []
    total = len(Ns)*NUM_TRIALS
    prog_step = max(1, total//20)
    count = 0

    for N in Ns:
        bh_trial = []; nv_trial = []; fmm_trial = []
        for _ in range(NUM_TRIALS):
            seed = np.random.randint(0,10_000)
            bh_trial.append(run_barnes_hut_simulation(N, seed))
            #nv_trial.append(run_naive_simulation(N, seed))
            fmm_trial.append(run_fast_multipole_simulation(N, seed))
            count += 1
            if count % prog_step == 0:
                print(f"Progress: {100*count//total}%")
        bh_times.append(np.mean(bh_trial)); bh_err.append(np.std(bh_trial))
        #nv_times.append(np.mean(nv_trial)); nv_err.append(np.std(nv_trial))
        fmm_times.append(np.mean(fmm_trial)); fmm_err.append(np.std(fmm_trial))

    '''
    os.makedirs("data", exist_ok=True)
    np.save("data/computation_times.npy", {
        "N": Ns,
        "BH_times": bh_times,   "BH_err": bh_err,
        "NV_times": nv_times,   "NV_err": nv_err,
        "FMM_times": fmm_times, "FMM_err": fmm_err
    })
    print("Saved to data/computation_times.npy")
    '''

    # --- plotting ---
    plt.figure(figsize=(8,6))
    plt.errorbar(Ns, bh_times, yerr=bh_err,  label='Barnes-Hut', marker='o', capsize=3)
    #plt.errorbar(Ns, nv_times, yerr=nv_err,  label='Naïve N²', marker='s', capsize=3)
    plt.errorbar(Ns, fmm_times, yerr=fmm_err, label='Fast Multipole', marker='^', capsize=3)

    plt.xlabel('Number of Bodies (N)')
    plt.ylabel(f'Time for {SIMULATION_STEPS} step(s) (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    #plt.xscale('log')  # optional: makes power‐law scaling clearer
    #plt.yscale('log')  # optional
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_computation_times()