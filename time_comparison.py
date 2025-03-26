import numpy as np
import time
import matplotlib.pyplot as plt
from barneshut import Simulation as BarnesHutSimulation, Body as BarnesHutBody
from naivegravity import Simulation as NaiveSimulation, Body as NaiveBody

# Constants
SIMULATION_STEPS = 100
INTERVAL = 50  # Milliseconds (interval for the animation)
N_MIN = 3  # Minimum number of bodies
N_MAX = 40  # Maximum number of bodies

# Function to run Barnes-Hut Simulation and time it
def run_barnes_hut_simulation(N):
    bodies_barneshut = [
        BarnesHutBody(
            position=np.random.uniform(-1e11, 1e11, 2),
            velocity=np.random.uniform(-3e3, 3e3, 2),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(N)
    ]
    simulation_barneshut = BarnesHutSimulation(bodies_barneshut, -1e11, 1e11, -1e11, 1e11)
    
    # Time the Barnes-Hut simulation
    start_time = time.time()
    for _ in range(SIMULATION_STEPS):
        simulation_barneshut.move()
    end_time = time.time()
    
    return end_time - start_time

# Function to run Naïve N-body Simulation and time it
def run_naive_simulation(N):
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

# Main function to run both simulations for a range of N and plot the times
def compare_computation_times():
    Ns = range(N_MIN, N_MAX + 1)  # Range of N from 3 to 40
    barnes_hut_times = []
    naive_times = []
    
    # Run simulations for each N and record the times
    for N in Ns:
        print(f"Running simulations for N = {N}")
        
        # Time Barnes-Hut simulation
        barnes_hut_time = run_barnes_hut_simulation(N)
        barnes_hut_times.append(barnes_hut_time)
        
        # Time Naive simulation
        naive_time = run_naive_simulation(N)
        naive_times.append(naive_time)
    
    # Plotting the computation times as a function of N
    plt.figure(figsize=(8, 6))
    plt.plot(Ns, barnes_hut_times, label='Barnes-Hut Simulation', color='blue', marker='o')
    plt.plot(Ns, naive_times, label='Naïve N-Body Simulation', color='red', marker='s')
    
    plt.xlabel('Number of Bodies (N)', fontsize=12)
    plt.ylabel('Computation Time (seconds)', fontsize=12)
    plt.title('Computation Time for 100 Steps', fontsize=14)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_computation_times()