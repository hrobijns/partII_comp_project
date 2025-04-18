import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11
dt = 3600
THETA = 0.5
P = 2  # Only monopole and dipole for now

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.array([0.0, 0.0])

class Cell:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
        self.center_of_mass = np.zeros(2)
        self.total_mass = 0.0
        self.bodies = []
        self.children = None
        self.multipole = {
            "monopole": 0.0,
            "dipole": np.zeros(2)
        }
        self.local_expansion = np.zeros(2)

    def contains(self, pos):
        return self.x_min <= pos[0] < self.x_max and self.y_min <= pos[1] < self.y_max

    def insert(self, body):
        if not self.bodies and self.total_mass == 0:
            self.bodies.append(body)
            self.center_of_mass = body.position
            self.total_mass = body.mass
            return

        if self.children is None:
            self.subdivide()

        self.total_mass += body.mass
        self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass) + body.position * body.mass) / self.total_mass

        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return

    def subdivide(self):
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        self.children = [
            Cell(self.x_min, x_mid, self.y_min, y_mid),
            Cell(x_mid, self.x_max, self.y_min, y_mid),
            Cell(self.x_min, x_mid, y_mid, self.y_max),
            Cell(x_mid, self.x_max, y_mid, self.y_max)
        ]
        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []

    def compute_multipole(self):
        if self.children:
            self.multipole["monopole"] = 0.0
            self.multipole["dipole"] = np.zeros(2)
            for child in self.children:
                child.compute_multipole()
                m = child.multipole["monopole"]
                d = child.multipole["dipole"]
                r = child.center - self.center
                self.multipole["monopole"] += m
                self.multipole["dipole"] += d + m * r
        else:
            m = sum(b.mass for b in self.bodies)
            d = sum(b.mass * (b.position - self.center) for b in self.bodies)
            self.multipole["monopole"] = m
            self.multipole["dipole"] = d

    def apply_fmm_force(self, body, source):
        r = body.position - source.center
        dist2 = np.sum(r**2) + 1e-10
        dist = np.sqrt(dist2)
        width = source.x_max - source.x_min

        if source.children is None or width / dist < THETA:
            m = source.multipole["monopole"]
            d = source.multipole["dipole"]
            F_monopole = -G * body.mass * m * r / dist2**(1.5)
            F_dipole = -G * body.mass * (
                (3 * np.dot(d, r) * r - dist2 * d) / dist2**(2.5)
            )
            return F_monopole + F_dipole
        else:
            F = np.zeros(2)
            for child in source.children:
                F += self.apply_fmm_force(body, child)
            return F

class Simulation:
    def __init__(self, bodies, space_size):
        self.bodies = bodies
        self.space_size = space_size

    def compute_forces(self):
        root = Cell(-self.space_size, self.space_size, -self.space_size, self.space_size)
        for b in self.bodies:
            root.insert(b)
        root.compute_multipole()
        for b in self.bodies:
            b.force = root.apply_fmm_force(b, root)

    def move(self):
        self.compute_forces()
        for b in self.bodies:
            b.velocity += b.force / b.mass * dt
            b.position += b.velocity * dt

class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        self.scatters = [self.ax.plot([], [], 'wo', markersize=(b.mass)/1e27)[0] for b in bodies]
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True)

    def update(self, frame):
        self.simulation.move()
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters

    def show(self):
        plt.show()

# Example usage:
np.random.seed(28)
bodies = [Body(
    position=np.random.uniform(-1e11, 1e11, 2),
    velocity=np.random.uniform(-3e3, 3e3, 2),
    mass=np.random.uniform(5e26, 5e27)
) for _ in range(100)]

simulation = Simulation(bodies, space_size=2e11)
anim = Animation(bodies, simulation)
anim.show()