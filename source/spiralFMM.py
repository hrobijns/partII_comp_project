import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmath
from math import comb as binomial

G = 2.959122082855911e-4  # AU^3 / M_sun / day^2
dt = 1 / 24  # 1 hour time steps (in days)


class Body:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(2)


class Node:
    def __init__(self, bounds, particles=None):
        self.bounds = bounds
        self.particles = particles if particles else []
        self.children = []
        self.center_of_mass = None
        self.total_mass = 0
        self.outer_expansion = None
        self.inner_expansion = None
        self.parent = None

    def is_leaf(self):
        return len(self.children) == 0


class Simulation:
    def __init__(self, bodies, black_hole, max_depth=10, max_particles_per_node=3, theta=0.5, expansion_order=2):
        self.bodies = bodies
        self.max_depth = max_depth
        self.max_particles_per_node = max_particles_per_node
        self.theta = theta
        self.expansion_order = expansion_order
        self.rebuild_tree()
        self.black_hole = black_hole

    def rebuild_tree(self):
        self.root_node = self.build_tree(self.bodies)
        self.compute_center_of_mass_and_total_mass(self.root_node)
        self.build_outer(self.root_node)
        self.build_inner(self.root_node)
        self.compute_forces()

    def build_tree(self, bodies):
        margin = 1e-3
        min_x = min(body.position[0] for body in bodies) - margin
        max_x = max(body.position[0] for body in bodies) + margin
        min_y = min(body.position[1] for body in bodies) - margin
        max_y = max(body.position[1] for body in bodies) + margin
        return self._build_tree(bodies, (min_x, min_y, max_x, max_y))

    def _build_tree(self, particles, bounds, depth=0):
        if len(particles) <= self.max_particles_per_node or depth >= self.max_depth:
            return Node(bounds, particles)

        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        quadrants = [
            (bounds[0], bounds[1], cx, cy),
            (cx, bounds[1], bounds[2], cy),
            (bounds[0], cy, cx, bounds[3]),
            (cx, cy, bounds[2], bounds[3])
        ]

        children = []
        for q in quadrants:
            q_particles = [p for p in particles if q[0] <= p.position[0] < q[2] and q[1] <= p.position[1] < q[3]]
            if q_particles:
                child = self._build_tree(q_particles, q, depth + 1)
                child.parent = None
                children.append(child)

        node = Node(bounds)
        node.children = children
        for child in children:
            child.parent = node
        return node

    def compute_center_of_mass_and_total_mass(self, node):
        if node.is_leaf():
            total_mass = sum(b.mass for b in node.particles)
            if total_mass > 0:
                x = sum(b.mass * b.position[0] for b in node.particles) / total_mass
                y = sum(b.mass * b.position[1] for b in node.particles) / total_mass
                node.center_of_mass = np.array([x, y])
                node.total_mass = total_mass
        else:
            total_mass = 0
            center = np.zeros(2)
            for child in node.children:
                self.compute_center_of_mass_and_total_mass(child)
                if child.total_mass > 0:
                    total_mass += child.total_mass
                    center += child.total_mass * np.array(child.center_of_mass)
            if total_mass > 0:
                node.center_of_mass = center / total_mass
                node.total_mass = total_mass

    def build_outer(self, node):
        node.center = 0.5 * (np.array([node.bounds[0], node.bounds[1]]) + np.array([node.bounds[2], node.bounds[3]]))
        p = self.expansion_order
        node.outer_expansion = [np.zeros(2) for _ in range(p + 1)]  # monopole, dipole, etc.

        if node.is_leaf():
            for body in node.particles:
                r = body.position - node.center
                node.outer_expansion[0] += body.mass * np.ones(2)  # Monopole treated as scalar but stored vector for uniformity
                if p >= 1:
                    node.outer_expansion[1] += body.mass * r
        else:
            for child in node.children:
                self.build_outer(child)
                d = child.center - node.center
                node.outer_expansion[0] += child.outer_expansion[0]
                if p >= 1:
                    node.outer_expansion[1] += child.outer_expansion[1] + child.outer_expansion[0] * d

    def build_inner(self, node, parent_inner=None, parent_center=None):
        node.center = 0.5 * (np.array([node.bounds[0], node.bounds[1]]) + np.array([node.bounds[2], node.bounds[3]]))
        node.inner_expansion = np.zeros(2)

        if parent_inner is not None:
            node.inner_expansion += parent_inner  # inherited downward

        if node.parent:
            for neighbor in self.get_well_separated_nodes(node):
                r = node.center - neighbor.center
                r_norm = np.linalg.norm(r) + 1e-8
                if r_norm == 0:
                    continue
                m = neighbor.outer_expansion[0][0]  # scalar mass from vector storage
                d = neighbor.outer_expansion[1]
                node.inner_expansion += -G * (
                    m * r / r_norm**3 - np.dot(d, r) * r / r_norm**5
                )

        for child in node.children:
            self.build_inner(child, node.inner_expansion, node.center)

    def get_well_separated_nodes(self, node):
        if not hasattr(node, 'parent') or node.parent is None:
            return []

        parent = node.parent
        neighbors = self.get_adjacent_nodes(parent)

        interaction_list = []
        for nbr in neighbors:
            if nbr.is_leaf():
                if nbr != node and not self.is_adjacent(nbr, node):
                    interaction_list.append(nbr)
            else:
                for child in nbr.children:
                    if not self.is_adjacent(child, node):
                        interaction_list.append(child)
        return interaction_list

    def get_adjacent_nodes(self, node):
        result = []
        if not hasattr(node, 'parent') or node.parent is None:
            return result

        for sibling in node.parent.children:
            if sibling != node and self.are_bounds_adjacent(node.bounds, sibling.bounds):
                result.append(sibling)

        for parent_nbr in self.get_adjacent_nodes(node.parent):
            if not parent_nbr.is_leaf():
                for child in parent_nbr.children:
                    if self.are_bounds_adjacent(node.bounds, child.bounds):
                        result.append(child)
        return result

    def is_adjacent(self, node1, node2):
        return self.are_bounds_adjacent(node1.bounds, node2.bounds)

    def are_bounds_adjacent(self, b1, b2):
        ax0, ay0, ax1, ay1 = b1
        bx0, by0, bx1, by1 = b2
        dx = max(0, max(bx0 - ax1, ax0 - bx1))
        dy = max(0, max(by0 - ay1, ay0 - by1))
        return dx <= 0 and dy <= 0

    def compute_forces(self):
        for body in self.bodies:
            node = self.find_leaf(self.root_node, body)
            r = body.position - node.center
            force = node.inner_expansion.copy()
            force += self.direct_interactions(body, node)
            body.force = force

    def direct_interactions(self, body, node):
        force = np.zeros(2)
        for other in node.particles:
            if other is not body:
                r_vec = other.position - body.position
                r = np.linalg.norm(r_vec) + 1e-8
                force += G * other.mass * r_vec / r**3
        return force

    def find_leaf(self, node, body):
        if node.is_leaf():
            return node
        x, y = body.position
        for child in node.children:
            bx0, by0, bx1, by1 = child.bounds
            if bx0 <= x < bx1 and by0 <= y < by1:
                return self.find_leaf(child, body)
        return node

    def move(self):
        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt
            body.position += body.velocity * dt

        self.rebuild_tree()

        for body in self.bodies:
            body.velocity += 0.5 * (body.force / body.mass) * dt


class Animation:
    def __init__(self, bodies, simulation, black_hole, steps=100, interval=50, output_file="figures/spiralBH.mp4"):
        self.bodies = bodies
        self.simulation = simulation
        self.black_hole = black_hole
        self.steps = steps
        self.interval = interval
        self.output_file = output_file  # Define the output_file argument here
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.scatters = []

        for body in bodies:
            if body is not self.black_hole:
                scatter = self.ax.plot([], [], 'wo', markersize=1, markeredgewidth=0)[0]
                self.scatters.append(scatter)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # This makes sure the animation object is retained
        self.ani = FuncAnimation(self.fig, self.update, frames=self.steps,
                                 interval=self.interval, repeat=True)

        # Call tight_layout to ensure minimal whitespace
        self.fig.tight_layout()

    def update(self, frame):
        print(f"Rendering frame {frame + 1}/{self.steps}")
        self.simulation.move()
        idx = 0
        for body in self.bodies:
            if body is not self.black_hole:
                self.scatters[idx].set_data(body.position[0], body.position[1])
                idx += 1
        return self.scatters
    
    def save(self):
        # Save the animation to the specified output file
        self.ani.save(self.output_file, writer='ffmpeg', dpi=300)

    def show(self):
        plt.show()


def generate_spiral_galaxy(n_bodies, arms=5, arm_strength=0.5, spread=0.1, radius=1.5):
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
        direction = np.array([position[1], -position[0]])
        direction /= np.linalg.norm(direction)
        velocity = direction * speed

        # Star mass
        mass = np.random.uniform(0.05, 0.5)
        bodies.append(Body(position, velocity, mass))
    
    return bodies


#####################################################################################################

if __name__ == "__main__":
    np.random.seed(42)

    # Add central black hole
    black_hole = Body(position=[0, 0], velocity=[0, 0], mass=1000.0)

    # Generate spiral galaxy
    stars = generate_spiral_galaxy(n_bodies=1000)
    bodies = [black_hole] + stars


    simulation = Simulation(bodies, black_hole=black_hole)
    anim = Animation(bodies, simulation, black_hole=black_hole, steps=10, interval=20)
    anim.show()
