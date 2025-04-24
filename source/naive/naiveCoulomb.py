import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters (dimensionless units)
k    = 1.0    # Coulomb constant
soft = 1e-1   # softening length
dt   = 1/24  # time step (in days)

class Body:
    """Represents a particle with position, velocity, charge, mass, and net force."""
    def __init__(self, position, velocity, charge, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.charge   = charge   # Coulomb charge
        self.mass     = mass     # inertial mass
        self.force    = np.zeros(2)

class Simulation:
    """Naive pairwise Coulomb interaction simulation."""
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self):
        """Compute pairwise repulsive Coulomb forces between all bodies."""
        # reset forces
        for b in self.bodies:
            b.force = np.zeros(2)

        # pairwise loop (i<j to avoid double counting)
        for i, b1 in enumerate(self.bodies):
            for j in range(i+1, len(self.bodies)):
                b2 = self.bodies[j]
                # vector from b2 to b1 (repulsion)
                diff = b1.position - b2.position
                dist = np.linalg.norm(diff)
                # Coulomb force magnitude
                f_mag = k * b1.charge * b2.charge / (dist**2 + soft**2)
                # unit direction vector
                f_vec = (f_mag / (dist + 1e-16)) * diff
                # apply equal and opposite forces
                b1.force +=  f_vec
                b2.force += -f_vec

    def move(self):
        # half-kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt
        # drift
        for b in self.bodies:
            b.position += b.velocity * dt
        # recompute forces
        self.compute_forces()
        # half-kick
        for b in self.bodies:
            b.velocity += 0.5 * (b.force / b.mass) * dt

class Animation:
    """Handles visualization using Matplotlib."""
    def __init__(self, bodies, sim, steps=200, interval=50):
        self.bodies   = bodies
        self.sim      = sim
        self.steps    = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.scat = [self.ax.plot([], [], 'wo', markersize=4)[0]
                     for _ in bodies]
        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self.steps,
                                 interval=self.interval,
                                 blit=True)

    def update(self, frame):
        self.sim.move()
        for sc, b in zip(self.scat, self.bodies):
            sc.set_data(b.position[0], b.position[1])
        return self.scat

    def show(self):
        plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    N = 20
    # initialize bodies with uniform positive charge
    bodies = [
        Body(
            position=np.random.uniform(-1, 1, 2),
            velocity=np.random.uniform(-0.05, 0.05, 2),
            charge=1.0,
            mass=1.0
        ) for _ in range(N)
    ]
    sim  = Simulation(bodies)
    anim = Animation(bodies, sim)
    anim.show()
