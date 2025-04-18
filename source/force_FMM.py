import numpy as np
import matplotlib.pyplot as plt
import time
from FMM import Body, Simulation  # Import the Body and Simulation classes from FMM.py

# Function to run the simulation and measure force calculation count
def run_simulation(N):
    # Generate N random bodies
    bodies = [
        Body(
            position=np.random.uniform(-10, 10, 2),  # in AU
            velocity=np.random.uniform(0.0, 0.0, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(N)
    ]
    
    # Create a simulation object
    simulation = Simulation(bodies)
    
    # Compute the forces and return the count of calculations
    simulation.compute_forces()
    return simulation.calculation_count

# Main function to run the scaling test and plot the results
def plot_force_scaling():
    # List of N values to test (number of bodies)
    N_values = [10, 50, 100, 200, 500, 1000, 2000, 5000]
    
    # List to store the number of force calculations for each N
    calculation_counts = []
    
    # Run the simulation for each N and record the calculation count
    for N in N_values:
        print(f"Running simulation for N = {N}...")
        calc_count = run_simulation(N)
        calculation_counts.append(calc_count)
    
    # Plot the results: calculation count vs N (number of bodies)
    plt.figure(figsize=(8, 6))
    plt.plot(N_values, calculation_counts)
    plt.xlabel('Number of Bodies (N)')
    plt.ylabel('Number of Force Calculations')
    plt.title('Force Calculations as a Function of N')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_force_scaling()