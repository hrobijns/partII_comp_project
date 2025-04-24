import numpy as np
from source.bruteforce.core import Body, Simulation
from source.bruteforce.visualisation import EnergyMomentumAnimation

def main():
    # fixed seed for reproducibility
    np.random.seed(42)

    # create 100 random bodies (position in ±5 AU, velocity in ±0.05 AU/day, mass 0.1–1 M_sun)
    bodies = [
        Body(
            position=np.random.uniform(-5, 5, size=2),
            velocity=np.random.uniform(-0.05, 0.05, size=2),
            mass=np.random.uniform(0.1, 1.0),
        )
        for _ in range(100)
    ]

    # build simulation (default dt=1/24 day, softening=1e-1)
    sim = Simulation(bodies)

    # animate for 1000 steps with a 50 ms frame interval
    anim = EnergyMomentumAnimation(bodies, sim, steps=1000, interval=50)
    anim.show()

if __name__ == "__main__":
    main()