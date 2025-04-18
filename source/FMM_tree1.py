import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 2.959122082855911e-4  # Gravitational constant in AU^3 M_sun^-1 day^-2
dt = 1/24  # Time step (in units of days, e.g., 1/24 is equivalent to steps of an hour)


class Body:
    """Represents a celestial body with position, velocity, mass, and net force."""
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)  # in AU
        self.velocity = np.array(velocity, dtype=float)  # in AU/day
        self.mass = mass  # in solar masses (M_sun)
        self.force = np.zeros(2)  # in AU/day^2, initialised to zero


class Node:
    def __init__(self, bounds, particles=None):
        # Bounds in the form (xmin, ymin, xmax, ymax)
        self.bounds = bounds
        self.particles = particles if particles else []
        self.children = []  # Child nodes (subdivided nodes)
        self.center_of_mass = None
        self.total_mass = 0

    def is_leaf(self):
        return len(self.children) == 0


class Simulation:
    """Handles the physics of the n-body simulation using FMM."""
    def __init__(self, bodies, max_depth=10, max_particles_per_node=10, theta=0.5):
        self.bodies = bodies
        self.max_depth = max_depth
        self.max_particles_per_node = max_particles_per_node
        self.theta = theta
        self.root_node = self.build_tree(bodies)

    def build_tree(self, bodies):
        """Builds a quadtree to organize the bodies."""
        bounds = (min(body.position[0] for body in bodies),
                  min(body.position[1] for body in bodies),
                  max(body.position[0] for body in bodies),
                  max(body.position[1] for body in bodies))
        return self._build_tree(bodies, bounds)

    def _build_tree(self, particles, bounds, depth=0):
        """Recursively builds the quadtree."""
        if len(particles) <= self.max_particles_per_node or depth >= self.max_depth:
            return Node(bounds, particles)

        # Split the current bounding box into 4 quadrants
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        quadrants = [
            (bounds[0], bounds[1], cx, cy),  # bottom-left
            (cx, bounds[1], bounds[2], cy),  # bottom-right
            (bounds[0], cy, cx, bounds[3]),  # top-left
            (cx, cy, bounds[2], bounds[3])   # top-right
        ]
        
        quadrants_particles = [[] for _ in range(4)]
        for particle in particles:
            for i, q in enumerate(quadrants):
                if q[0] <= particle.position[0] < q[2] and q[1] <= particle.position[1] < q[3]:
                    quadrants_particles[i].append(particle)
                    break
        
        children = []
        for i, q_particles in enumerate(quadrants_particles):
            if len(q_particles) > 0:
                children.append(self._build_tree(q_particles, quadrants[i], depth + 1))

        node = Node(bounds)
        node.children = children
        return node

    def plot_quadtree(self, ax):
        """Plots the quadtree structure."""
        self._plot_quadtree(self.root_node, ax)

    def _plot_quadtree(self, node, ax):
        """Recursively plots the quadtree structure."""
        # Plot the current node's bounding box
        x_min, y_min, x_max, y_max = node.bounds
        ax.plot([x_min, x_max], [y_min, y_min], 'w-', lw=1)  # bottom edge
        ax.plot([x_min, x_max], [y_max, y_max], 'w-', lw=1)  # top edge
        ax.plot([x_min, x_min], [y_min, y_max], 'w-', lw=1)  # left edge
        ax.plot([x_max, x_max], [y_min, y_max], 'w-', lw=1)  # right edge

        # Recur for children if they exist
        for child in node.children:
            self._plot_quadtree(child, ax)


class QuadtreePlot:
    """Handles the visualization of the quadtree structure."""
    def __init__(self, bodies, simulation):
        self.bodies = bodies
        self.simulation = simulation

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_facecolor("black")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', 'box')  # Ensures the aspect ratio is square

    def plot(self):
        """Plots the quadtree structure along with bodies."""
        # Plot the quadtree structure
        self.simulation.plot_quadtree(self.ax)

        # Plot the bodies
        body_positions = np.array([body.position for body in self.bodies])
        self.ax.scatter(body_positions[:, 0], body_positions[:, 1], c='yellow', s=50, label="Bodies", edgecolor='black')

        # Add a grid and labels
        self.ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.set_title("Quadtree Structure and Bodies", color='white', fontsize=14)

        plt.show()

###############################################################################################

if __name__ == "__main__":
    np.random.seed(42)
    bodies = [
        Body(
            position=np.random.uniform(-1, 1, 2),  # in AU
            velocity=np.random.uniform(0.0, 0.0, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(1000)
    ]

    simulation = Simulation(bodies)
    quadtree_plot = QuadtreePlot(bodies, simulation)
    quadtree_plot.plot()
