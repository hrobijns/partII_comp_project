import time
import numpy as np
import matplotlib.pyplot as plt

# import your FMM bodies & sim
from source.FMM.working import Body, Simulation

# Benchmark parameters
no_bodies     = 50    # number of bodies
steps         = 10     # how many .step() calls to time
dt            = 0.01  # time-step (matches your Simulation default)
init_velocity = (0.0, 0.0)

# List of expansion orders to test
expansion_orders = [2, 3, 4, 5, 6, 7]
times = []

for order in expansion_orders:
    print(f"Running expansion order = {order}")
    np.random.seed(48)

    # build bodies
    bodies = []
    for _ in range(no_bodies):
        x, y = np.random.uniform(-1000, 1000, 2)
        m    = np.random.uniform(0.1, 1.0)
        bodies.append(Body(position=(x, y),
                           velocity=init_velocity,
                           mass=m))

    # pass the integer `order` here:
    sim = Simulation(bodies, dt=dt, nterms=order)

    # time one step
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    t1 = time.time()

    times.append(t1 - t0)

# Plot results
plt.figure(figsize=(6,4))
plt.plot(expansion_orders, times, marker='o')
plt.xlabel('Expansion Order')
plt.ylabel('Computation Time (s)')
plt.title(f'FMM Step Time vs. Expansion Order (N={no_bodies})')
plt.grid(True)
plt.tight_layout()
plt.show()