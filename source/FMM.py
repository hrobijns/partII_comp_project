import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
        self.multipole_expansion = None  # This will hold the multipole expansion of the node

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
        self.compute_center_of_mass_and_total_mass(self.root_node)
        self.multipole_expansion(self.root_node)

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

    def compute_center_of_mass_and_total_mass(self, node):
        """Computes the center of mass and total mass for each node."""
        if node.is_leaf():
            total_mass = sum([body.mass for body in node.particles])
            center_of_mass = np.mean([body.position for body in node.particles], axis=0)
        else:
            total_mass = 0
            weighted_sum = np.zeros(2)
            for child in node.children:
                self.compute_center_of_mass_and_total_mass(child)
                total_mass += child.total_mass
                weighted_sum += child.center_of_mass * child.total_mass
            center_of_mass = weighted_sum / total_mass

        node.center_of_mass = center_of_mass
        node.total_mass = total_mass

    def multipole_expansion(self, node):
        """Computes the multipole expansion for each node."""
        if node.is_leaf():
            node.multipole_expansion = (node.total_mass, node.center_of_mass)
        else:
            for child in node.children:
                self.multipole_expansion(child)

            total_mass = node.total_mass
            center_of_mass = node.center_of_mass
            node.multipole_expansion = (total_mass, center_of_mass)

    def compute_forces(self):
        """Computes the forces using the FMM method."""
        for body in self.bodies:
            body.force = np.zeros(2)  # Clear previous forces
            self._compute_force_on_particle(self.root_node, body)

    def _compute_force_on_particle(self, node, body):
        """Recursively computes the gravitational force on a given body."""
        r = body.position - node.center_of_mass
        r_mag = np.linalg.norm(r)
        
        if node.is_leaf():
            # Direct summation for leaf nodes (individual particles)
            for other_body in node.particles:
                if other_body != body:
                    diff_vector = other_body.position - body.position
                    distance = np.linalg.norm(diff_vector)
                    force_magnitude = G * body.mass * other_body.mass / (distance**2 + 1e-1**2)
                    force_vector = force_magnitude * diff_vector / distance
                    body.force += force_vector
        else:
            # Far-field approximation using multipole expansion
            total_mass, center_of_mass = node.multipole_expansion
            if r_mag > self.theta * (node.bounds[2] - node.bounds[0]):
                # Use multipole expansion if the node is far enough
                force_vector = total_mass * r / r_mag**3
                body.force += force_vector
            else:
                # Otherwise, recursively calculate forces from child nodes
                for child in node.children:
                    self._compute_force_on_particle(child, body)

    def move(self):
        """Updates the positions and velocities of all bodies."""
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
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-2, 2)  # in AU
        self.ax.set_ylim(-2, 2)  # in AU
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # create a scatter point for each body (size scaled by mass)
        self.scatters = [
            self.ax.plot([], [], "wo", markersize=body.mass * 2)[0] 
            for body in bodies
        ]

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True
        )

    def update(self, frame):
        """Updates the animation frame-by-frame."""
        self.simulation.move()
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters

    def show(self):
        plt.show()

###############################################################################################

if __name__ == "__main__":
    np.random.seed(42)
    bodies = [
        Body(
            position=np.random.uniform(-2, 2, 2),  # in AU
            velocity=np.random.uniform(-0.05, 0.05, 2),  # in AU/day
            mass=np.random.uniform(0.1, 1),  # in M_sun
        )
        for _ in range(10)
    ]

    simulation = Simulation(bodies)
    anim = Animation(bodies, simulation)
    anim.show()
