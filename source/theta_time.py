import time
import numpy as np
import matplotlib.pyplot as plt
from barneshut3D import Body, Simulation  # <-- use barneshut3D
from scipy.stats import linregress

def generate_bodies(n, seed=42):
    np.random.seed(seed)
    return [
        Body(
            position=np.random.uniform(-0.5, 0.5, 3),  # 3D position
            velocity=np.random.uniform(-0.05, 0.05, 3),  # 3D velocity
            mass=np.random.uniform(0.1, 1),
        )
        for _ in range(n)
    ]

def time_simulation(theta, steps=5, n_bodies=500):
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
    theta_values = np.linspace(0.1, 1.0, 15)
    times = benchmark_thetas(theta_values, steps=20, n_bodies=200)

    log_times = np.log(times)
    log_theta_values = np.log(theta_values)

    # Perform linear regression to find the slope (alpha)
    slope, intercept, r_value, p_value, std_err = linregress(log_theta_values, log_times)
    print(f"Estimated alpha: {slope:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(theta_values, times, marker='o')
    plt.xlabel('Theta (Barnes-Hut opening angle)')
    plt.ylabel('Time (seconds for 20 steps)')
    plt.title('Simulation Time vs. Theta (3D Barnes-Hut)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

