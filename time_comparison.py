import numpy as np
import time
import matplotlib.pyplot as plt
from barneshut import Simulation as BarnesHutSimulation, Body as BarnesHutBody
from naivegravity import Simulation as NaiveSimulation, Body as NaiveBody
from FMM import Simulation as FastMultipoleSimulation, Body as FastMultipoleBody

# Constants
SIMULATION_STEPS = 5
INTERVAL = 50  # Milliseconds (interval for the animation)
N_MIN = 10  # Minimum number of bodies
N_MAX = 1000  # Maximum number of bodies
NUM_TRIALS = 3  # Number of trials per N for averaging
seeds = [11,68,83,96]

# Function to run Barnes-Hut Simulation and time it
def run_barnes_hut_simulation(N, seed):
    np.random.seed(seed)
    bodies_barneshut = [
        BarnesHutBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    simulation_barneshut = BarnesHutSimulation(bodies_barneshut, space_size=2e11)
    
    # Time the Barnes-Hut simulation
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_barneshut.move()
    end_time = time.time()
    
    return end_time - start_time

# Function to run Naïve N-body Simulation and time it
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
    
    # Time the Naive simulation
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_naive.move()
    end_time = time.time()
    
    return end_time - start_time

# Function to run Fast Multipole Simulation and time it
def run_fast_multipole_simulation(N, seed):
    np.random.seed(seed)
    bodies_fastmultipole = [
        FastMultipoleBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    simulation_fastmultipole = FastMultipoleSimulation(bodies_fastmultipole, space_size=2e11)
    
    # Time the Fast Multipole simulation
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_fastmultipole.move()
    end_time = time.time()
    
    return end_time - start_time

# Main function to run all simulations for a range of N and plot the times
def compare_computation_times():
    # Generate 100 logarithmically spaced points between N_MIN and N_MAX
    Ns = np.logspace(np.log10(N_MIN), np.log10(N_MAX), num=50, dtype=int)
    barnes_hut_times = []
    naive_times = []
    fast_multipole_times = []
    barnes_hut_errors = []
    naive_errors = []
    fast_multipole_errors = []
    
    # Run simulations for each N and record the times
    for N in Ns:
        print(f"Running simulations for N = {N}")
        
        barnes_hut_trial_times = []
        naive_trial_times = []
        fast_multipole_trial_times = []
        
        for seed in seeds:
            barnes_hut_trial_times.append(run_barnes_hut_simulation(N, seed))
            naive_trial_times.append(run_naive_simulation(N, seed))
            fast_multipole_trial_times.append(run_fast_multipole_simulation(N, seed))
        
        barnes_hut_times.append(np.mean(barnes_hut_trial_times))
        naive_times.append(np.mean(naive_trial_times))
        fast_multipole_times.append(np.mean(fast_multipole_trial_times))
        
        barnes_hut_errors.append(np.std(barnes_hut_trial_times))
        naive_errors.append(np.std(naive_trial_times))
        fast_multipole_errors.append(np.std(fast_multipole_trial_times))
    
    # Plotting the computation times as a function of N with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(Ns, barnes_hut_times, yerr=barnes_hut_errors, label='Barnes-Hut Simulation', color='blue', marker='o', markersize=4, capsize=3)
    plt.errorbar(Ns, naive_times, yerr=naive_errors, label='Naïve N-Body Simulation', color='red', marker='s', markersize=4, capsize=3)
    plt.errorbar(Ns, fast_multipole_times, yerr=fast_multipole_errors, label='Fast Multipole Simulation', color='green', marker='^', markersize=4, capsize=3)
    
    plt.xlabel('Number of Bodies (N)', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Computation Time for 5 Steps', fontsize=14)
    
    # Set the x-axis to a logarithmic scale
    plt.xscale('log')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_computation_times()