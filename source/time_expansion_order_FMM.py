import time
import numpy as np
import matplotlib.pyplot as plt
from FMM import Body, Simulation, Animation  # Import necessary components

# Parameters
n_bodies = 50  # Number of bodies (fixed)
steps = 1  # Number of animation steps
interval = 50  # Interval in ms

# Create random bodies
np.random.seed(42)
bodies = [
    Body(
        position=np.random.uniform(-10, 10, 2),
        velocity=np.random.uniform(-0.002, 0.002, 2),
        mass=np.random.uniform(0.02, 0.1)
    )
    for _ in range(n_bodies)
]

# Test different expansion orders
expansion_orders = [2, 4, 6, 8, 10]
times = []

for p in expansion_orders:
    print('expansion order:')
    print(p)
    sim = Simulation(bodies, expansion_order=p)
    
    # Time the simulation run for a fixed number of steps
    start_time = time.time()
    
    for _ in range(steps):
        sim.move()  # Perform simulation step
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

# Plotting the result
plt.plot(expansion_orders, times, marker='o')
plt.xlabel('Expansion Order')
plt.ylabel('Computation Time (s)')
plt.title(f'Computation Time vs. Expansion Order for {n_bodies} Bodies')
plt.grid(True)
plt.show()