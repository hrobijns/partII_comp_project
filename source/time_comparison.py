import numpy as np
import time
import matplotlib.pyplot as plt
import os
from barneshut import Simulation as BarnesHutSimulation, Body as BarnesHutBody
from naivegravity import Simulation as NaiveSimulation, Body as NaiveBody
from FMM import Simulation as FastMultipoleSimulation, Body as FastMultipoleBody  # Importing FMM components from FMM.py

# Constants
SIMULATION_STEPS = 1
INTERVAL = 50  # Milliseconds (interval for the animation)
N_MIN = 10  # Minimum number of bodies
N_MAX = 500  # Maximum number of bodies
NUM_TRIALS = 1  # Number of trials per N for averaging

def run_barnes_hut_simulation(N, seed):
    np.random.seed(seed)
    bodies_barneshut = [
        BarnesHutBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    simulation_barneshut = BarnesHutSimulation(bodies_barneshut, space_size=2e11, theta=0.5)
    
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_barneshut.move()
    end_time = time.time()
    
    return end_time - start_time

def run_naive_simulation(N, seed):
    np.random.seed(seed)
    bodies_naive = [
        NaiveBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    simulation_naive = NaiveSimulation(bodies_naive)
    
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_naive.move()
    end_time = time.time()
    
    return end_time - start_time

def run_fast_multipole_simulation(N, seed):
    np.random.seed(seed)
    
    # Create bodies for the FMM simulation
    bodies_fastmultipole = [
        FastMultipoleBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    
    # Set up FMM simulation with bodies
    simulation_fastmultipole = FastMultipoleSimulation(bodies_fastmultipole)
    
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_fastmultipole.move()  # Perform a single step in the FMM simulation
    end_time = time.time()
    
    return end_time - start_time

def compare_computation_times():
    Ns = np.logspace(np.log10(N_MIN), np.log10(N_MAX), num=50, dtype=int)
    barnes_hut_times = []
    naive_times = []
    fast_multipole_times = []
    barnes_hut_errors = []
    naive_errors = []
    fast_multipole_errors = []
    
    total_combinations = len(Ns) * NUM_TRIALS  # Total combinations of N and trials
    progress_step = total_combinations // 20  # Update progress every 5% of the total combinations
    progress_count = 0
    
    for N in Ns:
        print(f"Running simulations for N = {N}...")

        barnes_hut_trial_times = []
        naive_trial_times = []
        fast_multipole_trial_times = []
        
        for i in range(NUM_TRIALS):
            seed = np.random.randint(0, 10000)  # Generate a new seed for each trial
            barnes_hut_trial_times.append(run_barnes_hut_simulation(N, seed))
            naive_trial_times.append(run_naive_simulation(N, seed))
            fast_multipole_trial_times.append(run_fast_multipole_simulation(N, seed))
            
            progress_count += 1
            if progress_count % progress_step == 0:
                print(f"Progress: {100 * progress_count // total_combinations}% completed")
        
        barnes_hut_times.append(np.mean(barnes_hut_trial_times))
        naive_times.append(np.mean(naive_trial_times))
        fast_multipole_times.append(np.mean(fast_multipole_trial_times))
        
        barnes_hut_errors.append(np.std(barnes_hut_trial_times))
        naive_errors.append(np.std(naive_trial_times))
        fast_multipole_errors.append(np.std(fast_multipole_trial_times))
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save results to a .npy file
    np.save("data/computation_times.npy", {
        "N": Ns,
        "BH_times": barnes_hut_times,
        "BH_errors": barnes_hut_errors,
        "naive_times": naive_times,
        "naive_errors" : naive_errors,
        "FMM_times": fast_multipole_times,
        "FMM_errors": fast_multipole_errors
    })
    print("Computation times saved to data/computation_times.npy")
    
    # Plotting the computation times
    plt.figure(figsize=(8, 6))
    plt.errorbar(Ns, barnes_hut_times, yerr=barnes_hut_errors, label='Barnes-Hut Simulation', color='blue', marker='o', markersize=4, capsize=3)
    plt.errorbar(Ns, naive_times, yerr=naive_errors, label='Naïve N-Body Simulation', color='red', marker='s', markersize=4, capsize=3)
    plt.errorbar(Ns, fast_multipole_times, yerr=fast_multipole_errors, label='Fast Multipole Simulation', color='green', marker='^', markersize=4, capsize=3)
    
    plt.xlabel('Number of Bodies (N)', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Computation Time for 5 Steps', fontsize=14)
    
    #plt.xscale('log')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_computation_times()
