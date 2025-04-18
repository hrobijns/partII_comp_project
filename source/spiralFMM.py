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
        self.force = np.zeros(2)  # in AU/day^2, initialized to zero


class Node:
    def __init__(self, bounds, particles=None):
        self.bounds = bounds
        self.particles = particles if particles else []
        self.children = []  # Child nodes (subdivided nodes)
        self.center_of_mass = None
        self.total_mass = 0
        self.multipole_expansion = None
        self.outer_expansion = None
        self.inner_expansion = None

    def is_leaf(self):
        return len(self.children) == 0


class Simulation:
    """Handles the physics of the n-body simulation using FMM."""
    def __init__(self, bodies, black_hole, max_depth=10, max_particles_per_node=3, theta=0.5):
        self.bodies = bodies
        self.max_depth = max_depth
        self.max_particles_per_node = max_particles_per_node
        self.theta = theta
        self.black_hole = black_hole
        self.root_node = self.build_tree(bodies)
        self.compute_center_of_mass_and_total_mass(self.root_node)
        self.multipole_expansion(self.root_node)
        self.build_outer(self.root_node)  # Step 2: Compute Outer for each node
        self.build_inner(self.root_node)  # Step 3: Compute Inner for each node
        self.add_nearest_neighbor_contributions()  # Step 4: Add nearest neighbor contributions

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

    def build_outer(self, node):
        """Computes the Outer(n) for each node using a post-order traversal."""
        if node.is_leaf():
            node.outer_expansion = self.compute_outer_for_leaf(node)
        else:
            node.outer_expansion = (0, 0)  # Initialize outer expansion
            for child in node.children:
                self.build_outer(child)
                # Combine the outer expansions of the children
                node.outer_expansion = self.combine_outer(node.outer_expansion, child.outer_expansion, child.center_of_mass)

    def compute_outer_for_leaf(self, node):
        """Computes the outer expansion for a leaf node."""
        total_mass = sum([body.mass for body in node.particles])
        center_of_mass = np.mean([body.position for body in node.particles], axis=0)
        return total_mass, center_of_mass

    def combine_outer(self, outer1, outer2, center2):
        """Combines two outer expansions, shifting the center of the second expansion."""
        mass1, center1 = outer1
        mass2, center2 = outer2
        combined_mass = mass1 + mass2
        combined_center = (mass1 * center1 + mass2 * center2) / combined_mass
        return combined_mass, combined_center

    def build_inner(self, node):
        """Computes the Inner(n) for each node using a pre-order traversal."""
        if node.is_leaf():
            node.inner_expansion = self.compute_inner_for_leaf(node)
        else:
            parent_node = node
            node.inner_expansion = self.compute_inner_for_parent(parent_node)
            
            for child in node.children:
                self.build_inner(child)

    def compute_inner_for_leaf(self, node):
        """Computes the inner expansion for a leaf node."""
        return (0, 0)  # Placeholder; Actual computation needed

    def compute_inner_for_parent(self, parent_node):
        """Computes the inner expansion for a parent node based on interaction set."""
        return (0, 0)  # Placeholder; Actual computation needed

    def add_nearest_neighbor_contributions(self):
        """Adds the contributions of nearest neighbors to each leaf node's inner expansion."""
        for node in self.iterate_leaf_nodes(self.root_node):
            for body in node.particles:
                self.add_forces_from_neighbors(body)

    def iterate_leaf_nodes(self, node):
        """Recursively iterate over leaf nodes in the quadtree."""
        if node.is_leaf():
            yield node
        else:
            for child in node.children:
                yield from self.iterate_leaf_nodes(child)

    def add_forces_from_neighbors(self, body):
        """Adds the forces due to nearby particles (nearest neighbors) directly."""
        for other_body in self.bodies:
            if other_body != body:
                diff_vector = other_body.position - body.position
                distance = np.linalg.norm(diff_vector)
                force_magnitude = G * body.mass * other_body.mass / (distance**2 + 1e-1**2)
                force_vector = force_magnitude * diff_vector / distance
                body.force += force_vector

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
            for other_body in node.particles:
                if other_body != body:
                    diff_vector = other_body.position - body.position
                    distance = np.linalg.norm(diff_vector)
                    force_magnitude = G * body.mass * other_body.mass / (distance**2 + 1e-1**2)
                    force_vector = force_magnitude * diff_vector / distance
                    body.force += force_vector
        else:
            total_mass, center_of_mass = node.multipole_expansion
            r_mag = np.linalg.norm(r)
            if r_mag > self.theta * (node.bounds[2] - node.bounds[0]):
                force = self.compute_multipole_force(body, node)
                body.force += force
            else:
                for child in node.children:
                    self._compute_force_on_particle(child, body)

    def compute_multipole_force(self, body, node):
        """Computes the force using the multipole expansion for a node."""
        force = np.zeros(2)
        return force

    def move(self):
        for body in self.bodies:
            if body is not self.black_hole:
                body.velocity += 0.5 * (body.force / body.mass) * dt
                body.position += body.velocity * dt
                

        self.compute_forces()

        for body in self.bodies:
            if body is not self.black_hole:
                body.velocity += 0.5 * (body.force / body.mass) * dt

class Animation:
    """Handles visualization and formatting of the simulation using Matplotlib."""
    def __init__(self, bodies, simulation, black_hole, steps=100, interval=50):
        self.bodies = bodies
        self.simulation = simulation
        self.steps = steps
        self.interval = interval
        self.black_hole = black_hole

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-2, 2)  # in AU
        self.ax.set_ylim(-2, 2)  # in AU
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # create a scatter point for each body (size scaled by mass)
        self.scatters = [
            self.ax.plot([], [], "wo", markersize=body.mass * 0.5)[0]
            for body in bodies
        ]

        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.steps, interval=self.interval, repeat=True
        )

    def update(self, frame):
        """Updates the animation frame-by-frame."""
        print(f"Rendering frame {frame + 1}/{self.steps}")
        self.simulation.move()  # Perform one step of the simulation
        for scatter, body in zip(self.scatters, self.bodies):
            scatter.set_data(body.position[0], body.position[1])
        return self.scatters

    def save(self):
        # Save the animation to the specified output file
        self.ani.save('figures/spiralFMM.mp4', writer='ffmpeg', dpi=300)

    def show(self):
        plt.show()


def generate_spiral_galaxy(n_bodies, arms=4, arm_strength=0.7, spread=0.3, radius=1.5):
    bodies = []
    for i in range(n_bodies):
        # Radial distribution, denser near center
        r = radius * np.random.power(2.5)
        
        # Spiral angle: looser connection to radial distance
        base_theta = r * 4  # logarithmic spiral
        arm_offset = ((i % arms) / arms) * 2 * np.pi
        theta_noise = np.random.normal(scale=spread * (1 - arm_strength))
        theta = base_theta + arm_offset + theta_noise
        
        # Add perpendicular noise to spread stars around the arm
        offset_radius = r + np.random.normal(scale=spread * arm_strength)

        x = offset_radius * np.cos(theta)
        y = offset_radius * np.sin(theta)
        position = np.array([x, y])

        # Calculate orbital velocity around center (black hole mass = 100 M_sun)
        distance = np.linalg.norm(position)
        speed = np.sqrt(G * 100 / (distance + 0.1))
        direction = np.array([-position[1], position[0]])  # Perpendicular to position vector
        direction /= np.linalg.norm(direction)  # Normalize direction
        velocity = direction * speed  # Apply the velocity magnitude

        # Star mass
        mass = np.random.uniform(0.05, 0.5)
        bodies.append(Body(position, velocity, mass))
    
    return bodies


#####################################################################################################

if __name__ == "__main__":
    np.random.seed(42)

    # Add central black hole
    black_hole = Body(position=[0, 0], velocity=[0, 0], mass=100.0)

    # Generate spiral galaxy
    stars = generate_spiral_galaxy(n_bodies=100)
    bodies = [black_hole] + stars


    simulation = Simulation(bodies, black_hole=black_hole)
    anim = Animation(bodies, simulation, black_hole=black_hole, steps=10, interval=20)
    anim.show()
