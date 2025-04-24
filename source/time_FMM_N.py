import time
import numpy as np
import matplotlib.pyplot as plt

# import your FMM bodies & sim
from working import Body, Simulation

# Benchmark parameters
expansion_order = 4   # your multipole order
steps = 1             # how many .step() calls to time
dt = 0.01             # time‐step (matches your Simulation default)
init_velocity = (0.0, 0.0)

# List of problem sizes to test
n_bodies_list = np.linspace(10,1000,10, dtype=int)
times = []

for n in n_bodies_list:
    print(f"Running N = {n} bodies…")
    np.random.seed(2)
    bodies = []
    for _ in range(n):
        x, y = np.random.uniform(-1000,1000, 2)
        m = np.random.uniform(0.1, 1.0)   # positive masses
        bodies.append(Body(position=(x, y),
                           velocity=init_velocity,
                           mass=m))
    # (Optional) add a central heavy mass:
    # bodies.append(Body(position=(0,0), velocity=(0,0), mass=1e4))

    sim = Simulation(bodies, dt=dt, nterms=expansion_order)

    t0 = time.time()
    for _ in range(steps):
        sim.step()
    t1 = time.time()

    times.append(t1 - t0)

# Plot results
plt.figure(figsize=(6,4))
plt.plot(n_bodies_list, times, marker='o')
plt.xlabel('Number of Bodies')
plt.ylabel('Computation Time (s)')
plt.title(f'FMM Step Time vs. N (order={expansion_order})')
plt.grid(True)
plt.tight_layout()
plt.show()