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
    def __init__(self, bodies):
        self.bodies = bodies
        self.max_depth = self.calculate_depth(len(bodies))  # Calculate dynamic depth based on number of bodies
        self.root_node = self.build_tree(bodies)

    def calculate_depth(self, num_bodies):
        """Calculate the depth based on the number of bodies."""
        # Calculate depth such that there are about 4 particles per leaf node
        ideal_leaf_nodes = num_bodies / 4
        depth = int(np.ceil(np.log(ideal_leaf_nodes) / np.log(4)))
        return max(depth, 1)  # Ensure that the depth is at least 1

    def build_tree(self, bodies):
        """Builds a balanced quadtree with fixed depth."""
        bounds = (min(body.position[0] for body in bodies),
                  min(body.position[1] for body in bodies),
                  max(body.position[0] for body in bodies),
                  max(body.position[1] for body in bodies))
        return self._build_balanced_tree(bodies, bounds, self.max_depth)

    def _build_balanced_tree(self, particles, bounds, depth):
        """Builds a balanced quadtree with the specified fixed depth."""
        if depth == 0 or len(particles) == 0:
            return Node(bounds, particles)

        # Split the current bounding box into 4 quadrants
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        quadrants = [
            (bounds[0], bounds[1], cx, cy),  # bottom-left
            (cx, bounds[1], bounds[2], cy),  # bottom-right
            (bounds[0], cy, cx, bounds[3]),  # top-left
            (cx, cy, bounds[2], bounds[3])   # top-right
        ]

        # At the deepest level, the leaf nodes are created at this level.
        if depth == 1:
            return self._create_leaf_nodes(particles, quadrants)

        # Recur for each quadrant
        children = []
        for q in quadrants:
            children.append(self._build_balanced_tree(particles, q, depth - 1))

        # Create a node at this level
        node = Node(bounds)
        node.children = children
        return node

    def _create_leaf_nodes(self, particles, quadrants):
        """Create leaf nodes with particles assigned to the corresponding quadrants."""
        leaf_nodes = []
        for q in quadrants:
            # Filter particles that fall within the current quadrant
            particles_in_quadrant = [p for p in particles if q[0] <= p.position[0] < q[2] and q[1] <= p.position[1] < q[3]]
            leaf_nodes.append(Node(q, particles_in_quadrant))
        return leaf_nodes


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
        self._plot_quadtree(self.simulation.root_node, self.ax)

        # Plot the bodies
        body_positions = np.array([body.position for body in self.bodies])
        self.ax.scatter(body_positions[:, 0], body_positions[:, 1], c='yellow', s=50, label="Bodies", edgecolor='black')

        # Add a grid and labels
        self.ax.grid(True, which='both', color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.set_title("Balanced Quadtree Structure and Bodies", color='white', fontsize=14)

        plt.show()

    def _plot_quadtree(self, node, ax):
        """Recursively plots the quadtree structure."""
        # Check if node is a leaf (node with no children)
        if isinstance(node, list):  # If the node is a list of leaf nodes
            for leaf in node:
                self._plot_quadtree(leaf, ax)
        else:
            # Plot the current node's bounding box
            x_min, y_min, x_max, y_max = node.bounds
            ax.plot([x_min, x_max], [y_min, y_min], 'w-', lw=1)  # bottom edge
            ax.plot([x_min, x_max], [y_max, y_max], 'w-', lw=1)  # top edge
            ax.plot([x_min, x_min], [y_min, y_max], 'w-', lw=1)  # left edge
            ax.plot([x_max, x_max], [y_min, y_max], 'w-', lw=1)  # right edge

            # Recur for children if they exist
            for child in node.children:
                self._plot_quadtree(child, ax)


if __name__ == "__main__":
    np.random.seed(42)
    bodies = [
        Body(
            position=np.random.uniform(-1, 1, 2),  # in AU
            velocity=np.random.uniform(0.0, 0.0, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(6)  # Example: 3000 bodies
    ]

    simulation = Simulation(bodies)  # Depth is calculated dynamically based on the number of bodies
    quadtree_plot = QuadtreePlot(bodies, simulation)
    quadtree_plot.plot()
