import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant
dt = 3600  # Time step (seconds)
theta = 0.5  # Barnes-Hut opening angle criterion

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.array([0.0, 0.0])

class QuadTreeNode:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.center_of_mass = np.array([0.0, 0.0])
        self.mass = 0.0
        self.body = None
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def insert(self, body):
        if self.is_leaf():
            if self.body is None:
                # If no body in node, store this body
                self.body = body
                self.mass = body.mass
                self.center_of_mass = body.position
            else:
                # Subdivide and reinsert both bodies
                self.subdivide()
                self.insert(self.body)
                self.body = None  # Clear body reference
                self.insert(body)
        else:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.update_com()

    def subdivide(self):
        mid_x = (self.x_min + self.x_max) / 2
        mid_y = (self.y_min + self.y_max) / 2
        self.children = [
            QuadTreeNode(self.x_min, mid_x, self.y_min, mid_y),  # Bottom-left
            QuadTreeNode(mid_x, self.x_max, self.y_min, mid_y),  # Bottom-right
            QuadTreeNode(self.x_min, mid_x, mid_y, self.y_max),  # Top-left
            QuadTreeNode(mid_x, self.x_max, mid_y, self.y_max)   # Top-right
        ]
    
    def contains(self, position):
        return (self.x_min <= position[0] < self.x_max and
                self.y_min <= position[1] < self.y_max)

    def update_com(self):
        if self.body is not None:
            self.center_of_mass = self.body.position
            self.mass = self.body.mass
        else:
            total_mass = 0
            weighted_position = np.array([0.0, 0.0])
            for child in self.children:
                if child.mass > 0:
                    total_mass += child.mass
                    weighted_position += child.center_of_mass * child.mass
            if total_mass > 0:
                self.mass = total_mass
                self.center_of_mass = weighted_position / total_mass
            else:
                self.mass = 0
                self.center_of_mass = np.array([0.0, 0.0])

    def calculate_force(self, body, theta):
        if self.is_leaf():
            if self.body is not None and self.body is not body:
                return self.calculate_pairwise_force(body, self.body)
            return np.array([0.0, 0.0])
        else:
            distance = np.linalg.norm(body.position - self.center_of_mass)
            size = max(self.x_max - self.x_min, self.y_max - self.y_min)
            if size / distance < theta:
                # Treat entire node as a single mass at center of mass
                return self.calculate_pairwise_force(body, Body(self.center_of_mass, [0.0, 0.0], self.mass))
            else:
                total_force = np.array([0.0, 0.0])
                for child in self.children:
                    total_force += child.calculate_force(body, theta)
                return total_force

    def calculate_pairwise_force(self, body1, body2):
        if body2 is None:
            return np.array([0.0, 0.0])  # Skip if there's no body
        
        diff_vector = body1.position - body2.position
        distance = np.linalg.norm(diff_vector)
        
        if distance > 6e6:  # Prevent division by zero
            f_mag = G * body1.mass * body2.mass / (distance**2)
            f = f_mag * diff_vector / distance
            return f
        else:
            return np.array([0.0, 0.0])  # No force if too close

class Simulation:
    def __init__(self, bodies, x_min, x_max, y_min, y_max):
        self.bodies = bodies
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def build_tree(self):
        self.root = QuadTreeNode(self.x_min, self.x_max, self.y_min, self.y_max)
        for body in self.bodies:
            self.root.insert(body)

    def move(self):
        self.build_tree()
        for body in self.bodies:
            body.force = self.root.calculate_force(body, theta)
        
        for body in self.bodies:
            body.velocity += (body.force / body.mass) * dt
            body.position += body.velocity * dt

class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Initialize scatter plot with correct sizes
        self.scatters = self.ax.scatter(
            [body.position[0] for body in bodies], 
            [body.position[1] for body in bodies], 
            color='white', 
            s=[body.mass / 1e27 for body in bodies]  # Ensure minimum size
        )

        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True)

    def update(self, frame):
        self.simulation.move()
        positions = np.array([body.position for body in self.bodies])
        self.scatters.set_offsets(positions)
        return self.scatters,

    def show(self):
        plt.show()

# Generate random bodies
np.random.seed(42)
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=np.random.uniform(5e26, 5e27)  # Mass varies by an order of magnitude
    ) for _ in range(100)
]

# Run simulation
simulation = Simulation(bodies, -1e11, 1e11, -1e11, 1e11)
anim = Animation(bodies, simulation)
anim.show()