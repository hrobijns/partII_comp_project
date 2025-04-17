import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# note that units are in terms of AU (distance), day (time) and solar mass (M_sun, mass)
G = 2.959122082855911e-4  # gravitational constant in AU^3 M_sun^-1 day^-2
dt = 1/24  # time step (in units of days, e.g. 1/24 is equivalent to steps of an hour)
theta = 0.5  # Barnes-Hut parameter
e = 1e-1


class Body:
    """Represents a celestial body with position, velocity, mass and net force."""
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)  # in AU
        self.velocity = np.array(velocity, dtype=float)  # in AU/day
        self.mass = mass  # in solar masses (M_sun)
        self.force = np.zeros(2)  # in AU/day^2, initialised to zero


class QuadTree:
    """Forms the quadtree and calculates forces according to the BH approximation."""
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center_of_mass = np.array([0.0, 0.0])
        self.total_mass = 0.0
        self.bodies = []
        self.children = None
    
    def insert(self, body):
        """Attempts to insert the body into a node, updating mass and c.o.m alongside"""
        if not self.bodies and self.total_mass == 0: # an empty node, simple insertion 
            self.bodies.append(body)
            self.center_of_mass = body.position
            self.total_mass = body.mass
            return
        
        if self.children is None: # an occupied node with no children, further divison
            self.subdivide()
        
        self.total_mass += body.mass 
        self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass) 
                               + body.position * body.mass) / self.total_mass 
        
        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return
    
    def subdivide(self):
        """Further subdivides a node into four quadrants"""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2
        self.children = [
            QuadTree(self.x_min, x_mid, self.y_min, y_mid),
            QuadTree(x_mid, self.x_max, self.y_min, y_mid),
            QuadTree(self.x_min, x_mid, y_mid, self.y_max),
            QuadTree(x_mid, self.x_max, y_mid, self.y_max)
        ]
        for body in self.bodies:
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        self.bodies = []
    
    def contains(self, position):
        """Check if a coordinate falls within the specific quadrant/node"""
        return self.x_min <= position[0] < self.x_max and self.y_min <= position[1] < self.y_max
    
    def compute_force(self, body, theta):
        """Compute total forces, employing BH approximation for far away nodes"""
        if not self.total_mass or (len(self.bodies) == 1 and self.bodies[0] is body):
            return np.array([0.0, 0.0])
        
        dx, dy = self.center_of_mass - body.position
        distance = np.sqrt(dx**2 + dy**2) 
        width = self.x_max - self.x_min
        
        if width / distance < theta or not self.children:
            force_magnitude = G * body.mass * self.total_mass / (distance**2 + e**2)
            return force_magnitude * np.array([dx, dy]) / distance
        
        total_force = np.array([0.0, 0.0])
        for child in self.children:
            total_force += child.compute_force(body, theta)
        return total_force


class Simulation:
    """Simulates the physics using a quadtree and BH approximation"""
    def __init__(self, bodies, space_size, theta):
        self.bodies = bodies
        self.space_size = space_size
        self.theta = theta
        self.compute_forces()
    
    def compute_forces(self):
        root = QuadTree(-self.space_size, self.space_size, -self.space_size, 
                        self.space_size)
        for body in self.bodies:
            root.insert(body)
        for body in self.bodies:
            body.force = root.compute_force(body, self.theta)
    
    def move(self):
        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt
            body.position += body.velocity * dt

        self.compute_forces()

        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt

class Animation:
    """Handles visualization and formatting of the simulation using Matplotlib."""
    def __init__(self, bodies, simulation, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.scatters = [self.ax.plot([], [], 'wo', 
                                      markersize=(body.mass)*2)[0] for body in bodies]
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps, 
                                 interval=self.interval, repeat=True)
    
    def update(self, frame):
        self.simulation.move()
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters
    
    def show(self):
        plt.show()

#####################################################################################################

if __name__ == "__main__":
    np.random.seed(28)
    bodies = [
        Body(
            position=np.random.uniform(-2, 2, 2),  # in AU
            velocity=np.random.uniform(-0.05, 0.05, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(20)
    ]

    simulation = Simulation(bodies, space_size=2e11)
    anim = Animation(bodies, simulation)
    anim.show()