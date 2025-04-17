import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 2.959122082855911e-4
dt = 1/24
theta = 0.5
e = 1e-1

class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)

class QuadTree:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.center_of_mass = np.array([0.0, 0.0])
        self.total_mass = 0.0
        self.bodies = []
        self.children = None
    
    def insert(self, body):
        if not self.bodies and self.total_mass == 0:
            self.bodies.append(body)
            self.center_of_mass = body.position
            self.total_mass = body.mass
            return
        
        if self.children is None:
            self.subdivide()
        
        self.total_mass += body.mass 
        self.center_of_mass = (self.center_of_mass * (self.total_mass - body.mass) 
                               + body.position * body.mass) / self.total_mass 
        
        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return
    
    def subdivide(self):
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
        return self.x_min <= position[0] < self.x_max and self.y_min <= position[1] < self.y_max

    def draw(self, ax):
        """Recursively draw the boundaries of each node."""
        ax.plot([self.x_min, self.x_max], [self.y_min, self.y_min], 'gray', lw=0.5)
        ax.plot([self.x_min, self.x_max], [self.y_max, self.y_max], 'gray', lw=0.5)
        ax.plot([self.x_min, self.x_min], [self.y_min, self.y_max], 'gray', lw=0.5)
        ax.plot([self.x_max, self.x_max], [self.y_min, self.y_max], 'gray', lw=0.5)
        if self.children:
            for child in self.children:
                child.draw(ax)

# Generate and plot the first quadtree
np.random.seed(28)
bodies = [
    Body(
        position=np.random.uniform(-2, 2, 2),
        velocity=np.random.uniform(-0.05, 0.05, 2),
        mass=np.random.uniform(0.1, 1),
    )
    for _ in range(20)
]

# Initialize root quadtree
space_size = 2  # Same as the random range above
root = QuadTree(-space_size, space_size, -space_size, space_size)
for body in bodies:
    root.insert(body)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor('black')
ax.set_xlim(-space_size, space_size)
ax.set_ylim(-space_size, space_size)
ax.set_xticks([])
ax.set_yticks([])

# Draw bodies
for body in bodies:
    ax.plot(body.position[0], body.position[1], 'wo', markersize=body.mass * 3)

# Draw quadtree grid
root.draw(ax)
plt.tight_layout()
plt.savefig("figures/quadtree_plot.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

