import time
import matplotlib.pyplot as plt
import numpy as np
from FMM import Body, Simulation

def create_bodies(n):
    np.random.seed(42)
    return [
        Body(
            position=np.random.uniform(-1, 1, 2),
            velocity=np.random.uniform(-0.01, 0.01, 2),
            mass=np.random.uniform(0.1, 0.5),
        )
        for _ in range(n)
    ]

orders = range(1, 10)
times = []

for p in orders:
    bodies = create_bodies(50)
    sim = Simulation(bodies, expansion_order=p)

    start = time.time()
    for _ in range(10):  # Run a few steps
        sim.move()
    end = time.time()

    times.append(end - start)
    print(f"Expansion order {p}: {times[-1]:.4f}s")

plt.figure()
plt.plot(orders, times, marker='o')
plt.xlabel("Expansion Order (p)")
plt.ylabel("Time (s) for 10 steps")
plt.title("FMM Timing vs Expansion Order")
plt.tight_layout()
#plt.savefig("timing_vs_order.png")
plt.show()