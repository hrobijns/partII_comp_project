import time
import numpy as np
import matplotlib.pyplot as plt
from barneshut import Body, Simulation

def generate_bodies(n, seed=42):
    np.random.seed(seed)
    return [
        Body(
            position=np.random.uniform(-2, 2, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(n)
    ]

def time_simulation(theta, steps=10, n_bodies=100):
    bodies = generate_bodies(n_bodies)
    sim = Simulation(bodies, space_size=2, theta=theta)
    
    start = time.time()
    for _ in range(steps):
        sim.move()
    end = time.time()
    
    return end - start

def benchmark_thetas(theta_values, steps=10, n_bodies=100):
    times = []
    for theta in theta_values:
        print(f"Running simulation for theta={theta:.2f}...")
        t = time_simulation(theta, steps, n_bodies)
        times.append(t)
    return times

if __name__ == "__main__":
    theta_values = np.linspace(0.1, 1.5, 15)
    times = benchmark_thetas(theta_values, steps=20, n_bodies=200)

    plt.figure(figsize=(8, 5))
    plt.plot(theta_values, times, marker='o')
    plt.xlabel('Theta (Barnes-Hut opening angle)')
    plt.ylabel('Time (seconds for 20 steps)')
    plt.title('Simulation Time vs. Theta')
    plt.grid(True)
    plt.tight_layout()
    plt.show()