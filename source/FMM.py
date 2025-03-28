import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # Gravitational constant
dt = 3600  # Time step (seconds)
THETA = 0.5  # Multipole approximation threshold
P = 3  # Order of multipole expansion

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
        self.center_of_mass = np.array([0.0, 0.0])
        self.total_mass = 0.0
        self.bodies = []
        self.children = None
        self.multipole_moments = np.zeros((P, 2))
    
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
    
    def contains(self, position):
        return self.x_min <= position[0] < self.x_max and self.y_min <= position[1] < self.y_max
    
    def compute_multipole_moments(self):
        for child in self.children or []:
            child.compute_multipole_moments()
            self.multipole_moments += child.multipole_moments
    
    def compute_force(self, body):
        dx, dy = self.center_of_mass - body.position
        distance = np.sqrt(dx**2 + dy**2) + 1e-10
        width = self.x_max - self.x_min
        
        if width / distance < THETA or not self.children:
            force_magnitude = G * body.mass * self.total_mass / (distance**2)
            return force_magnitude * np.array([dx, dy]) / distance
        
        total_force = np.array([0.0, 0.0])
        for child in self.children:
            total_force += child.compute_force(body)
        return total_force

class Simulation:
    def __init__(self, bodies, space_size):
        self.bodies = bodies
        self.space_size = space_size
    
    def compute_forces(self):
        root = Cell(-self.space_size, self.space_size, -self.space_size, self.space_size)
        for body in self.bodies:
            root.insert(body)
        root.compute_multipole_moments()
        for body in self.bodies:
            body.force = root.compute_force(body)
    
    def move(self):
        self.compute_forces()
        for body in self.bodies:
            body.velocity += body.force / body.mass * dt
            body.position += body.velocity * dt

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
        self.scatters = [self.ax.plot([], [], 'wo', markersize=(body.mass)/1e27)[0] for body in bodies]
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
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=np.random.uniform(5e26, 5e27)
    ) for _ in range(100)
]

simulation = Simulation(bodies, space_size=2e11)
anim = Animation(bodies, simulation)
anim.show()