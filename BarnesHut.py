import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 6.67430e-11  # gravitational constant
mass = 5e26 # masses of the body (on the order of the mass of venus)
dt = 3600 # time step (seconds)
theta = 0.5 

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.array([0.0, 0.0])  # 2D force vector

class QuadTreeNode:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.center_of_mass = np.array([0.0, 0.0])
        self.mass = 0.0
        self.body = None  # To store one body if this is a leaf node
        self.children = []  # Stores child nodes if this is an internal node

    def is_leaf(self):
        return len(self.children) == 0

    def insert(self, body):
        if self.is_leaf():
            if self.body is None:
                self.body = body
                self.mass = body.mass
                self.center_of_mass = body.position * body.mass
            else:
                # Split into four children if the node contains more than one body
                self.subdivide()
                self.insert(self.body)  # Reinsert the existing body
                self.body = None  # No longer a leaf
                self.insert(body)
        else:
            # Insert body into appropriate child node
            for child in self.children:
                if child.contains(body.position):
                    child.insert(body)
                    break
        
        # Update the center of mass and mass of the current node
        self.update_com()

    def subdivide(self):
        # Create 4 sub-nodes for a 2D quadtree (in a rectangular region)
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
        # Update the center of mass of this node
        if self.body is not None:
            self.center_of_mass = self.body.position * self.body.mass
            self.mass = self.body.mass
        else:
            total_mass = 0
            weighted_position = np.array([0.0, 0.0])
            for child in self.children:
                total_mass += child.mass
                weighted_position += child.center_of_mass
            self.mass = total_mass
            self.center_of_mass = weighted_position / total_mass

    def calculate_force(self, body, theta):
        # Compute the force on a body due to the node (or a child)
        if self.is_leaf():
            if self.body is not body:
                return self.calculate_pairwise_force(body, self.body)
            else:
                return np.array([0.0, 0.0])
        else:
            distance = np.linalg.norm(body.position - self.center_of_mass)
            size = max(self.x_max - self.x_min, self.y_max - self.y_min)
            if size / distance < theta:
                # Treat this node as a single body
                return self.calculate_pairwise_force(body, self)
            else:
                # Otherwise, recurse through the children
                total_force = np.array([0.0, 0.0])
                for child in self.children:
                    total_force += child.calculate_force(body, theta)
                return total_force

    def calculate_pairwise_force(self, body1, body2):
        # Calculate the force between two bodies or body and node
        if isinstance(body2, QuadTreeNode):
            diff_vector = body1.position - body2.center_of_mass
            distance = np.linalg.norm(diff_vector)
            if np.abs(distance) > 6e6:  # Prevent division by zero or small distance
                f_mag = G * body1.mass * body2.mass / (distance**2)
                f = f_mag * diff_vector / np.linalg.norm(diff_vector)
                return f
            else:
                return np.array([0.0, 0.0])  # No force if too close
        else:
            diff_vector = body1.position - body2.position
            distance = np.linalg.norm(diff_vector)
            if np.abs(distance) > 6e6:  # Prevent division by zero or small distance
                f_mag = G * body1.mass * body2.mass / (distance**2)
                f = f_mag * diff_vector / np.linalg.norm(diff_vector)
                return f
            else:
                return np.array([0.0, 0.0])  # no force if too close
            

class Animation:
    def __init__(self, bodies, steps=100, interval=50):
        self.bodies = bodies
        self.steps = steps
        self.interval = interval
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1e11, 1e11)
        self.ax.set_ylim(-1e11, 1e11)
        self.scatters = [self.ax.plot([], [], 'wo', markersize=2)[0] for _ in bodies]
        

        # removing axes for the animation plot
        self.ax.set_xticks([])  
        self.ax.set_yticks([]) 
        self.ax.set_xlabel('')  
        self.ax.set_ylabel('') 

        self.ani = FuncAnimation(self.fig, self.update, frames=99999, interval=self.interval, repeat=True)

    def update(self, frame):
        Body.move(self.bodies)
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        
        return self.scatters
    
    def show(self):
        plt.show()

#################################################################################
# simulate, and plot the force calculations as a function of n (number of bodies)

np.random.seed(42)
bodies = [
    Body(
        position=np.random.uniform(-1e11, 1e11, 2),
        velocity=np.random.uniform(-3e3, 3e3, 2),
        mass=mass
    ) for _ in range(100)
]

anim = Animation(bodies)
anim.show()