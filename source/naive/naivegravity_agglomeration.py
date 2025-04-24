import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # gravitational constant
dt = 3600  # time step (seconds)

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.array([0.0, 0.0])
        self.radius = (self.mass / 1e27) * 1e8  # Approximate radius scaling

class Simulation:
    def __init__(self, bodies):
        self.bodies = bodies

    def compute_forces(self):
        for body in self.bodies:
            body.force = np.array([0.0, 0.0])
        
        for i, body1 in enumerate(self.bodies):
            for j, body2 in enumerate(self.bodies):
                if i != j:
                    diff_vector = body2.position - body1.position
                    distance = np.linalg.norm(diff_vector)
                    if distance > 6e6:  # Prevent division by zero
                        f_mag = G * body1.mass * body2.mass / (distance**2)
                        f = f_mag * diff_vector / distance
                        body1.force += f

    def merge_bodies(self):
        merged_bodies = []
        while self.bodies:
            body = self.bodies.pop()
            for other in self.bodies[:]:
                distance = np.linalg.norm(body.position - other.position)
                if distance < (body.radius + other.radius):  # Merge if within radius
                    new_mass = body.mass + other.mass
                    new_position = (body.position * body.mass + other.position * other.mass) / new_mass
                    new_velocity = (body.velocity * body.mass + other.velocity * other.mass) / new_mass
                    merged_bodies.append(Body(new_position, new_velocity, new_mass))
                    self.bodies.remove(other)
                    break
            else:
                merged_bodies.append(body)
        self.bodies = merged_bodies

    def move(self):
        self.compute_forces()
        for body in self.bodies:
            body.velocity += body.force / body.mass * dt
            body.position += body.velocity * dt
        self.merge_bodies()

class Animation:
    def __init__(self, bodies, simulation, steps=100, interval=5):
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
        self.scatters = []
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        for body in self.simulation.bodies:
            scatter, = self.ax.plot(body.position[0], body.position[1], 'wo', markersize=(body.mass)/1e27)
            self.scatters.append(scatter)
        return self.scatters
    
    def show(self):
        plt.show()

# Example usage:
np.random.seed(42)
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=np.random.uniform(5e26, 5e27)  # Mass varies by an order of magnitude
    ) for _ in range(100)
]

simulation = Simulation(bodies)
anim = Animation(bodies, simulation)
anim.show()