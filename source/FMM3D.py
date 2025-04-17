import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

G = 6.67430e-11  # Gravitational constant
dt = 3600  # Time step in seconds
THETA = 0.5  # Multipole acceptance criterion
EPSILON = 1e-3  # Softening parameter to avoid singularities

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(3)

class Octree:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        self.center_of_mass = np.zeros(3)
        self.total_mass = 0.0
        self.bodies = []
        self.children = None

    def contains(self, position):
        x, y, z = position
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        return (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max)

    def insert(self, body):
        if not self.bodies and self.total_mass == 0:
            self.bodies.append(body)
            self.center_of_mass = body.position
            self.total_mass = body.mass
            return

        if self.children is None:
            self.subdivide()

        self.total_mass += body.mass
        self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass) +
                               body.position * body.mass) / self.total_mass

        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return

    def subdivide(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_max) / 2

        self.children = []
        for dx in [(x_min, x_mid), (x_mid, x_max)]:
            for dy in [(y_min, y_mid), (y_mid, y_max)]:
                for dz in [(z_min, z_mid), (z_mid, z_max)]:
                    self.children.append(
                        Octree(dx[0], dx[1], dy[0], dy[1], dz[0], dz[1])
                    )

        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []

    def compute_force(self, body):
        if self.total_mass == 0 or (len(self.bodies) == 1 and self.bodies[0] is body):
            return np.zeros(3)

        dx = self.center_of_mass - body.position
        distance = np.linalg.norm(dx) + EPSILON
        width = self.bounds[1] - self.bounds[0]

        if width / distance < THETA or not self.children:
            force_mag = G * body.mass * self.total_mass / (distance**2)
            return force_mag * dx / distance

        total_force = np.zeros(3)
        for child in self.children:
            total_force += child.compute_force(body)
        return total_force

class Simulation:
    def __init__(self, bodies, space_size):
        self.bodies = bodies
        self.space_size = space_size

    def compute_forces(self):
        root = Octree(-self.space_size, self.space_size,
                      -self.space_size, self.space_size,
                      -self.space_size, self.space_size)
        for body in self.bodies:
            root.insert(body)

        for body in self.bodies:
            body.force = root.compute_force(body)

    def move(self):
        self.compute_forces()
        for body in self.bodies:
            body.velocity += body.force / body.mass * dt
            body.position += body.velocity * dt

class Animation3D:
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        self.ax.set_zlim(-1e11, 1e11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        positions = np.array([body.position for body in bodies])
        self.scat = self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                    c='white', s=[body.mass / 1e26 for body in bodies])
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps,
                                 interval=self.interval, repeat=True)

    def update(self, frame):
        self.simulation.move()
        positions = np.array([body.position for body in self.bodies])
        self.scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        return self.scat,

    def show(self):
        plt.show()

if __name__ == "__main__":
    np.random.seed(28)
    bodies = [
        Body(
            position=np.random.uniform(-1e11, 1e11, 3),
            velocity=np.random.uniform(-3e3, 3e3, 3),
            mass=np.random.uniform(5e26, 5e27)
        ) for _ in range(100)
    ]

    simulation = Simulation(bodies, space_size=2e11)
    anim = Animation3D(bodies, simulation)
    anim.show()