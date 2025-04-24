import time
import numpy as np
import matplotlib.pyplot as plt

# import your FMM bodies & sim
from source.FMM.working import Body, Simulation

# Benchmark parameters
steps         = 1      # how many .step() calls to time
dt            = 0.01    # time-step (matches your Simulation default)
init_velocity = (0.0, 0.0)

# Range of N to test
n_bodies_list = np.linspace(10, 10000, 10, dtype=int)

times = []

for N in n_bodies_list:
    print(f"Running N = {N}")
    np.random.seed(48)

    # build bodies
    bodies = [
        Body(
            position=tuple(np.random.uniform(-1000, 1000, 2)),
            velocity=init_velocity,
            mass=np.random.uniform(0.1, 1.0)
        )
        for _ in range(N)
    ]

    sim = Simulation(bodies, dt=dt, nterms=4)  # fix expansion order if you like

    # time N-body stepping
    t0 = time.time()
    for _ in range(steps):
        sim.step()
    t1 = time.time()

    times.append(t1 - t0)

# Plot results
plt.figure(figsize=(6,4))
plt.plot(n_bodies_list, times, marker='o')
plt.xlabel('Number of bodies (N)')
plt.ylabel(f'Time for {steps} steps (s)')
plt.title('FMM Step Time vs. N')
plt.grid(True)
plt.tight_layout()
plt.show()